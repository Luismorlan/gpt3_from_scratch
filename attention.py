# The corresponding Colab can be found here: https://colab.research.google.com/drive/1t6GN768qPsjhLlwCsdj8-ROXLf0MCsZS#scrollTo=YfKWbBugNF93
import torch
import torch.nn as nn
from datetime import datetime
from torch.nn import functional as F

# hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 1e-3
device = "cuda" if torch.cuda.is_available() else "cpu"
eval_iters = 200
n_embd = 32

# ---------

torch.manual_seed(1337)

with open("input.txt", "r", encoding="utf-8") as f:
    text = f.read()

chars = sorted(list(set(text)))
vocab_size = len(chars)

stoi = {ch: i for i, ch in enumerate(chars)}
itos = {i: ch for i, ch in enumerate(chars)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda l: "".join([itos[i] for i in l])

data = torch.tensor(encode(text), dtype=torch.long)
n = int(0.9 * len(data))
train_data = data[:n]
eval_data = data[n:]


def get_batch(split):
    data = train_data if split == "train" else eval_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i : i + block_size] for i in ix])
    y = torch.stack([data[i + 1 : i + block_size + 1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    # Set model to evaluation mode. Layers like dropout will not zero out
    # certain values
    model.eval()

    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Set model back to training mode. Layers like dropout will still have all
    # values
    model.train()
    return out


class Head(nn.Module):
    """one head of self-attention"""

    def __init__(self, head_size):
        super().__init__()
        self.key = nn.Linear(n_embd, head_size, bias=False)
        self.query = nn.Linear(n_embd, head_size, bias=False)
        self.value = nn.Linear(n_embd, head_size, bias=False)
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
        wei = q @ k.transpose(-2, -1) * C**-0.5

        # Make it a decoder block by keeping the information flow from only past
        # to the future.
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float("-inf"))

        wei = F.softmax(wei, dim=-1)  # (B,T,T)

        out = wei @ v  # (B,T,h)

        return out


class MultiHeadAttention(nn.Module):
    """multiple heads of self-attention in parallel"""

    def __init__(self, num_heads, head_size):
        super().__init__()

        # This needs to be nn.ModuleList because that's the only way for pytorch
        # to recognize it as parameters so that it can perform back propagation.
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])

        # This is according to the original paper
        self.proj = nn.Linear(n_embd, n_embd)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim=-1)  # (B,T,C*head)
        out = self.proj(out)
        return out


class FeedForward(nn.Module):
    """a simple linear layer followed by a non-linearity"""

    def __init__(self, n_emdb):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
        )

    def forward(self, x):
        return self.net(x)


class Block(nn.Module):
    """Transformer block, performs communication and computation"""

    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head

        # Do communication
        self.ma_head = MultiHeadAttention(n_head, head_size)

        # Do computation
        self.ffwd = FeedForward(n_embd)

    def forward(self, x):
        x = x + self.ma_head(x)
        x = x + self.ffwd(x)
        return x


class BigramLanguageModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_embedding_table = nn.Embedding(
            vocab_size, n_embd
        )  # token -> token embedding
        self.position_embedding_table = nn.Embedding(
            block_size, n_embd
        )  # pos -> pos embedding

        # Deprecated: Create a single self attention module
        # self.sa_head = Head(n_embd)

        self.blocks = nn.Sequential(
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
            Block(n_embd, n_head=4),
        )

        # This is the last output layer in the transformer paper.
        self.lm_head = nn.Linear(n_embd, vocab_size)

    def forward(self, idx, targets=None):
        B, T = idx.shape

        tok_emb = self.token_embedding_table(idx)  # (B,T,C)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))  # (T,C)

        x = tok_emb + pos_emb  # (B,T,C)

        x = self.blocks(x)

        logits = self.lm_head(x)  # (B,T,vocab_size)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B,T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            logits, loss = self(idx_cond)  # Logits here is of shape (B, T, C)
            logits = logits[
                :, -1, :
            ]  # Take only the last in the context so it transform to (B, C)
            probs = F.softmax(logits, dim=-1)  # (B, C)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)  # Concatenate on T's dimension
        return idx


model = BigramLanguageModel()
m = model.to(device)

# Create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

start_time = datetime.now()
print(f"training using device: {device}")
for iter in range(max_iters):
    # Every once in a while we report the loss
    if iter % eval_interval == 0:
        losses = estimate_loss()
        print(
            f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
        )

    xb, yb = get_batch("train")

    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

end_time = datetime.now()
time_diff = end_time - start_time
print(f"Traing Duration: {time_diff}")

print("generating...")
context = torch.zeros((1, 1), dtype=torch.long, device=device)
print(decode(m.generate(context, max_new_tokens=500)[0].tolist()))
print("finished!")
