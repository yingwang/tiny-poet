"""Train tiny-poet.

Usage:
  python train.py --config small --device cuda --iters 10000
  python train.py --config tiny --device cpu --iters 5000
"""

import argparse
import math
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def get_batch(split: str, block_size: int, batch_size: int, device: str):
    data_path = DATA_DIR / f"{split}.bin"
    data = np.memmap(data_path, dtype=np.int16, mode="r")
    ix = torch.randint(len(data) - block_size - 1, (batch_size,))
    x = torch.stack([torch.from_numpy(data[i : i + block_size].astype(np.int64)) for i in ix])
    y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + block_size].astype(np.int64)) for i in ix])
    return x.to(device), y.to(device)


@torch.no_grad()
def estimate_loss(model, block_size, batch_size, device, eval_iters=50):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split, block_size, batch_size, device)
            _, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(it: int, warmup: int, max_iters: int, max_lr: float, min_lr: float) -> float:
    """Linear warmup then cosine decay."""
    if it < warmup:
        return max_lr * (it + 1) / warmup
    if it > max_iters:
        return min_lr
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", choices=["tiny", "small", "base"], default="small")
    p.add_argument("--device", default="auto", help="cuda / mps / cpu / auto")
    p.add_argument("--iters", type=int, default=10000)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--warmup", type=int, default=200)
    p.add_argument("--eval_interval", type=int, default=500)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Device: {args.device}")

    # Load meta
    with open(DATA_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"Vocab size: {vocab_size}")

    # Build model
    config_cls = {"tiny": GPTConfig.tiny, "small": GPTConfig.small, "base": GPTConfig.base}[args.config]
    config = config_cls(vocab_size=vocab_size)
    print(f"Config: {args.config}  n_layer={config.n_layer}  n_head={config.n_head}  n_embd={config.n_embd}")

    model = GPT(config).to(args.device)
    print(f"Params: {model.num_params()/1e6:.2f}M")

    # Optimizer (AdamW with weight decay on matmul weights only)
    decay_params = [p for n, p in model.named_parameters() if p.dim() >= 2]
    no_decay_params = [p for n, p in model.named_parameters() if p.dim() < 2]
    optim_groups = [
        {"params": decay_params, "weight_decay": 0.1},
        {"params": no_decay_params, "weight_decay": 0.0},
    ]
    optimizer = torch.optim.AdamW(optim_groups, lr=args.lr, betas=(0.9, 0.95))

    start_iter = 0
    best_val_loss = float("inf")

    # Resume
    ckpt_path = CKPT_DIR / f"{args.config}.pt"
    if args.resume and ckpt_path.exists():
        print(f"Resuming from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location=args.device)
        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optimizer"])
        start_iter = ckpt["iter"] + 1
        best_val_loss = ckpt.get("best_val_loss", float("inf"))

    # Train loop
    model.train()
    t0 = time.time()
    for it in range(start_iter, args.iters):
        lr = get_lr(it, args.warmup, args.iters, args.lr, args.min_lr)
        for g in optimizer.param_groups:
            g["lr"] = lr

        X, Y = get_batch("train", config.block_size, args.batch_size, args.device)
        _, loss = model(X, Y)

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if it % args.log_interval == 0:
            dt = time.time() - t0
            print(f"iter {it:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {dt:.1f}s")

        if it > 0 and it % args.eval_interval == 0:
            losses = estimate_loss(model, config.block_size, args.batch_size, args.device)
            print(f"  >> eval: train {losses['train']:.4f}  val {losses['val']:.4f}")
            if losses["val"] < best_val_loss:
                best_val_loss = losses["val"]
                torch.save(
                    {
                        "model": model.state_dict(),
                        "optimizer": optimizer.state_dict(),
                        "config": config.__dict__,
                        "iter": it,
                        "best_val_loss": best_val_loss,
                        "meta": meta,
                    },
                    ckpt_path,
                )
                print(f"  >> saved checkpoint to {ckpt_path}")

    # Final save
    torch.save(
        {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "config": config.__dict__,
            "iter": args.iters,
            "best_val_loss": best_val_loss,
            "meta": meta,
        },
        CKPT_DIR / f"{args.config}_final.pt",
    )
    print(f"\nDone. Final checkpoint: {CKPT_DIR}/{args.config}_final.pt")


if __name__ == "__main__":
    main()
