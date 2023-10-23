# The corresponding Colab can be found here: https://colab.research.google.com/drive/1t6GN768qPsjhLlwCsdj8-ROXLf0MCsZS#scrollTo=YfKWbBugNF93
import torch
import torch.nn as nn
from datetime import datetime
from torch.nn import functional as F

# import wandb

from config import ModelConfig

# TODO: Use a different tokenizer
# TODO: Try GELU

# hyperparameters
device = "cuda" if torch.cuda.is_available() else "cpu"
dropout = 0.2

# ---------
# wandb.init(
#     project="implement gpt3 from scratch",
#     config={
#         "learning_rate": 0.02,
#         "architecture": "CNN",
#         "dataset": "CIFAR-100",
#         "epochs": 10,
#     },
# )

torch.manual_seed(1337)

"""
We don't need `Head` anymore as there is a more efficient way of calculating
attention of multiple heads without using for loop.

class Head(nn.Module):
    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, x):
        B, T, C = x.shape

        k = self.key(
            x
        )  # (B,T,h), h usually == C, if not, it usually is multi head and will concat to C.
        q = self.query(x)  # (B,T,h)
        v = self.value(x)  # (B,T,h)

        # Scale down so that the variance stays at the same to make the softmax
        # more sparsed out instead of an activation.
        wei = q @ k.transpose(-2, -1) * C**-0.5  # (B,T,T)

        # Make it a decoder block by keeping the information flow from only past
        # to the future.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)  # (B,T,T)
        wei = self.dropout(wei)

        out = wei @ v  # (B,T,h)

        return out

"""


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel

    num_heads - how many attention heads to run in parallel
    head_size - output dimension, where num_heads * head_size = C.
    """

    def __init__(self, n_embd, num_heads, block_size):
        super().__init__()

        # Option 1: Use for loop for multi head calculation.
        # This needs to be nn.ModuleList because that's the only way for pytorch
        # to recognize it as parameters so that it can perform back propagation.
        # >>>
        # self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        # self.head_size = head_size
        # <<<

        self.n_embd = n_embd

        assert n_embd % num_heads == 0
        self.num_heads = num_heads

        self.c_attn = nn.Linear(n_embd, 3 * n_embd, bias=False)
        # Register a (1,1,T,T) buffer

        # This is according to the original paper
        self.proj = nn.Linear(n_embd, n_embd)

        self.attn_dropout = nn.Dropout(dropout)
        self.dropout = nn.Dropout(dropout)

        # Use flash attention is it exist (for torch > 2.0)
        self.flash = hasattr(torch.nn.functional, "scaled_dot_product_attention")
        if not self.flash:
            print("WARNING: Using manual implemented attention")
            self.register_buffer(
                "tril",
                torch.tril(torch.ones(block_size, block_size)).view(
                    1, 1, block_size, block_size
                ),
            )

    def forward(self, x):
        # [Option 1] out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,C*head)

        # [Option 2] perform one matrix multiplication via one shot.
        B, T, C = x.shape

        # directly map the input to 3x size output, where split the last
        # dimension will give us the batched q, k, v
        q, k, v = self.c_attn(x).split(
            self.n_embd, dim=-1
        )  # 3 tuple, each of (B, T, C)

        # Then split the attention to be (B, nh, T, hs)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)

        if self.flash:
            out = torch.nn.functional.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=None,
                dropout_p=dropout if self.training else 0,
                is_causal=True,
            )
        else:
            print("using manual implementation of attention")
            # Calculate wei in a batch, shape: (B, nh, T, T)
            wei = q @ k.transpose(-2, -1) * (k.size(-1) ** -0.5)
            wei = wei.masked_fill(self.tril[:, :, :T, :T] == 0, float("-inf"))
            wei = F.softmax(wei, dim=-1)
            wei = self.attn_dropout(wei)
            out = wei @ v  # This is (B, nh, T, hs)

        # We need to then make it (B, T, C)
        # 1. transpose -> (B, T, nh, hs)
        # 2. making it contiguous memory so that we can call view
        # 3. reshape and remove the last dimension.
        out = out.transpose(1, 2).contiguous().view(B, T, C)

        # final dropout
        out = self.dropout(self.proj(out))
        return out


class FeedForward(nn.Module):
    """
    A simple linear layer followed by a non-linearity, used in a single block to
    perform calculation after all attention nodes.

    This is used at the end of the block to perform some final calculation.
    """

    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block, performs communication and computation"""

    def __init__(self, n_embd, n_head, block_size):
        super().__init__()

        # Do communication
        self.ma_head = MultiHeadAttention(n_embd, n_head, block_size)

        # Do computation
        self.ffwd = FeedForward(n_embd)

        # Perform layer norm
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Residual 1
        x = x + self.ma_head(self.ln1(x))

        # Residual 2
        x = x + self.ffwd(self.ln2(x))
        return x


class GPT(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        # Store key model hyper parameters
        self.config = config

        self.token_embedding_table = nn.Embedding(
            self.config.vocab_size, self.config.n_embd
        )  # token -> token embedding
        self.position_embedding_table = nn.Embedding(
            self.config.block_size, self.config.n_embd
        )  # pos -> pos embedding

        # Deprecated: Create a single self attention module
        # self.sa_head = Head(n_embd)

        self.blocks = nn.Sequential(
            *[
                Block(self.config.n_embd, self.config.n_head, self.config.block_size)
                for _ in range(self.config.n_layer)
            ],
        )

        self.ln_f = nn.LayerNorm(self.config.n_embd)  # final layer norm

        # This is the last output layer in the transformer paper.
        self.lm_head = nn.Linear(self.config.n_embd, self.config.vocab_size)

        print("num of parameters: %.2fM" % (self.get_num_params() / 1e6))

    def get_num_params(self, non_embedding=True):
        n_params = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n_params -= self.position_embedding_table.weight.numel()
        return n_params

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)

        x = self.ln_f(x)

        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape

            # Logits are the prediction for the next character (token) FOR ALL
            # BATCHES AND TIME, and we calculate the average cross entropy between
            # the logits and true label.
            # Note that, F.cross_entropy will calculate softmax of logits before
            # actually compare with targets.
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.block_size :]
            logits, loss = self(idx_cond)  # Logits here is of shape (B, T, C)
            logits = logits[
                :, -1, :
            ]  # Take only the last in the context so it transform to (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Concatenate on T's dimension
        return idx


# model = GPT()
# m = model.to(device)

# # Create a PyTorch optimizer
# optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

# start_time = datetime.now()
# print(f"training using device: {device}")
# for iter in range(max_iters):
#     # Every once in a while we report the loss
#     if iter % eval_interval == 0:
#         losses = estimate_loss()
#         print(
#             f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
#         )
#         wandb.log({"loss": losses})

#     xb, yb = get_batch("train")  # xb is (B, T), yb is (B, T)

#     logits, loss = model(xb, yb)
#     optimizer.zero_grad(set_to_none=True)
#     loss.backward()
#     optimizer.step()

# end_time = datetime.now()


# print("\ngenerating...")
# context = torch.zeros((1, 1), dtype=torch.long, device=device)
# print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
# print("finished!")
