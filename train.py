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
from typing import Optional

import numpy as np
import torch

from model import GPT, GPTConfig

DATA_DIR = Path(__file__).parent / "data"
CKPT_DIR = Path(__file__).parent / "checkpoints"
CKPT_DIR.mkdir(exist_ok=True)


def get_batch(split: str, block_size: int, batch_size: int, device: str):
    data_path = DATA_DIR / f"{split}.bin"
    data = np.memmap(data_path, dtype=np.int16, mode="r")
    num_starts = len(data) - block_size
    if num_starts <= 0:
        raise ValueError(
            f"{data_path} has {len(data)} tokens; at least {block_size + 1} are required"
        )
    ix = torch.randint(num_starts, (batch_size,))
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
    if warmup and it < warmup:
        return max_lr * (it + 1) / warmup
    if it >= max_iters:
        return min_lr
    decay_ratio = (it - warmup) / (max_iters - warmup)
    coeff = 0.5 * (1 + math.cos(math.pi * decay_ratio))
    return min_lr + coeff * (max_lr - min_lr)


def capture_rng_state() -> dict:
    """Capture generators used by batching and dropout for exact continuation."""
    state = {"torch": torch.get_rng_state()}
    if torch.cuda.is_available():
        state["cuda"] = torch.cuda.get_rng_state_all()
    mps = getattr(torch, "mps", None)
    if torch.backends.mps.is_available() and mps is not None and hasattr(mps, "get_rng_state"):
        state["mps"] = mps.get_rng_state()
    return state


def restore_rng_state(state: Optional[dict]) -> None:
    if not state:
        return
    torch.set_rng_state(state["torch"].cpu())
    if torch.cuda.is_available() and "cuda" in state:
        torch.cuda.set_rng_state_all([item.cpu() for item in state["cuda"]])
    mps = getattr(torch, "mps", None)
    if (
        torch.backends.mps.is_available()
        and mps is not None
        and hasattr(mps, "set_rng_state")
        and "mps" in state
    ):
        mps.set_rng_state(state["mps"].cpu())


def checkpoint_payload(model, optimizer, config, completed_iter, best_val_loss, meta) -> dict:
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": config.__dict__,
        "iter": completed_iter,
        "next_iter": completed_iter + 1,
        "best_val_loss": best_val_loss,
        "meta": meta,
        "rng_state": capture_rng_state(),
    }


def save_checkpoint(payload: dict, path: Path) -> None:
    """Write atomically so interruption cannot corrupt the previous checkpoint."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


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
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    if args.iters <= 0:
        p.error("--iters must be positive")
    if args.batch_size <= 0:
        p.error("--batch_size must be positive")
    if args.lr <= 0 or args.min_lr < 0 or args.min_lr > args.lr:
        p.error("learning rates must satisfy 0 <= --min_lr <= --lr and --lr > 0")
    if not 0 <= args.warmup < args.iters:
        p.error("--warmup must satisfy 0 <= warmup < iters")
    if args.eval_interval <= 0 or args.eval_iters <= 0 or args.log_interval <= 0:
        p.error("--eval_interval, --eval_iters, and --log_interval must be positive")

    # Device
    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    print(f"Device: {args.device}")
    torch.manual_seed(args.seed)

    # Load meta
    with open(DATA_DIR / "meta.pkl", "rb") as f:
        meta = pickle.load(f)
    vocab_size = meta["vocab_size"]
    print(f"Vocab size: {vocab_size}")

    latest_path = CKPT_DIR / f"{args.config}_latest.pt"
    best_path = CKPT_DIR / f"{args.config}_best.pt"
    final_path = CKPT_DIR / f"{args.config}_final.pt"
    legacy_path = CKPT_DIR / f"{args.config}.pt"

    resume_ckpt = None
    resume_path = None
    if args.resume:
        if latest_path.exists():
            resume_path = latest_path
        elif legacy_path.exists():
            resume_path = legacy_path
            print(f"No latest checkpoint found; using legacy checkpoint {legacy_path}")
        else:
            p.error(f"--resume requested, but neither {latest_path} nor {legacy_path} exists")
        resume_ckpt = torch.load(resume_path, map_location="cpu", weights_only=True)
        if resume_ckpt.get("meta") != meta:
            p.error("checkpoint tokenizer metadata does not match data/meta.pkl")

    # Build from the saved architecture when resuming, so code-default changes
    # cannot silently alter an existing run.
    if resume_ckpt is not None:
        config = GPTConfig(**resume_ckpt["config"])
    else:
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

    # Resume from latest training state. Old <config>.pt files are accepted once
    # for compatibility, but all new runs write explicit latest/best files.
    if resume_ckpt is not None:
        print(f"Resuming from {resume_path}")
        model.load_state_dict(resume_ckpt["model"])
        optimizer.load_state_dict(resume_ckpt["optimizer"])
        start_iter = resume_ckpt.get("next_iter", resume_ckpt["iter"] + 1)
        best_val_loss = resume_ckpt.get("best_val_loss", float("inf"))
        if "rng_state" in resume_ckpt:
            restore_rng_state(resume_ckpt["rng_state"])
        else:
            # Legacy checkpoints did not record RNG state. Resetting here at
            # least makes subsequent sampling deterministic.
            torch.manual_seed(args.seed)
            print("Warning: legacy checkpoint has no RNG state; continuation is not exact")
        if start_iter >= args.iters:
            p.error(
                f"checkpoint already reached iteration {start_iter}; "
                f"set --iters above {start_iter}"
            )

    # Train loop
    model.train()
    t0 = time.time()
    last_iter = start_iter - 1
    for it in range(start_iter, args.iters):
        last_iter = it
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

        if (it > 0 and it % args.eval_interval == 0) or it == args.iters - 1:
            losses = estimate_loss(
                model,
                config.block_size,
                args.batch_size,
                args.device,
                eval_iters=args.eval_iters,
            )
            print(f"  >> eval: train {losses['train']:.4f}  val {losses['val']:.4f}")
            improved = losses["val"] < best_val_loss
            if improved:
                best_val_loss = losses["val"]
            payload = checkpoint_payload(model, optimizer, config, it, best_val_loss, meta)
            save_checkpoint(payload, latest_path)
            print(f"  >> saved latest checkpoint to {latest_path}")
            if improved:
                save_checkpoint(payload, best_path)
                print(f"  >> saved best checkpoint to {best_path}")

    # Final save
    save_checkpoint(
        checkpoint_payload(model, optimizer, config, last_iter, best_val_loss, meta),
        final_path,
    )
    print(f"\nDone. Final checkpoint: {final_path}")


if __name__ == "__main__":
    main()
