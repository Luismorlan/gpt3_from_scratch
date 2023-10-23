import argparse
import json
import os

import torch
from config import ModelConfig, load_from_file
from model import GPT
import numpy as np
from datetime import datetime
from tqdm import tqdm


device = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(description="training script for transformer")
    parser.add_argument(
        "--config", help="file name in config/ directory", required=True
    )
    parser.add_argument(
        "--eval_iters",
        type=int,
        help="number of evaluation iteration to smoothen the loss",
        default=200,
    )
    parser.add_argument(
        "--batch_size", type=int, help="batch size of training", default=64
    )
    parser.add_argument(
        "--learning_rate", type=float, help="learning rate", default=3e-4
    )
    parser.add_argument(
        "--max_iters", type=int, help="max iteration in training", default=5000
    )
    parser.add_argument(
        "--eval_interval", type=int, help="print loss every x iterations", default=100
    )
    parser.add_argument(
        "--dataset", help="dataset to train the model", default="shakespeare"
    )
    return parser.parse_args()


@torch.no_grad()
def estimate_loss(model, train_data, val_data, args, config: ModelConfig):
    # Set model to evaluation mode. Layers like dropout will not zero out
    # certain values
    model.eval()

    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            X, Y = get_batch(
                split, train_data, val_data, config.block_size, args.batch_size
            )
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()

    # Set model back to training mode. Layers like dropout will still have all
    # values
    model.train()
    return out


def get_batch(split, train_data, val_data, block_size, batch_size):
    data = train_data if split == "train" else val_data
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack(
        [torch.from_numpy((data[i : i + block_size]).astype(np.int64)) for i in ix]
    )
    y = torch.stack(
        [
            torch.from_numpy((data[i + 1 : i + block_size + 1]).astype(np.int64))
            for i in ix
        ]
    )
    x, y = x.to(device), y.to(device)
    return x, y


def train(model, train_data, val_data, args, config: ModelConfig):
    start_time = datetime.now()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate)
    for iter in tqdm(range(args.max_iters)):
        if iter % args.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, args, config)
            print(
                f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}"
            )
        xb, yb = get_batch(
            "train", train_data, val_data, config.block_size, args.batch_size
        )  # xb is (B, T), yb is (B, T)
        logits, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    end_time = datetime.now()
    time_diff = end_time - start_time
    print(f"Traing Duration: {time_diff}")


def load_data(dataset: str):
    data_dir = os.path.join("data", dataset)
    train_data = np.memmap(
        os.path.join(data_dir, "train.bin"), dtype=np.uint16, mode="r"
    )
    val_data = np.memmap(os.path.join(data_dir, "val.bin"), dtype=np.uint16, mode="r")
    return train_data, val_data


def main(args):
    config = load_from_file(f"./config/{args.config}")
    train_data, val_data = load_data(args.dataset)
    model = GPT(config)
    model.to(device)
    print("start training")
    train(model, train_data, val_data, args, config)


if __name__ == "__main__":
    args = parse_args()
    main(args)
