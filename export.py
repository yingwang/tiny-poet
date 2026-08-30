"""Strip optimizer state from a training checkpoint for distribution.

train.py saves the optimizer alongside the weights so training can resume, which
roughly triples the file size. Inference never needs it, so this produces the
slim checkpoint that gets attached to a release.

Usage:
  python export.py --ckpt checkpoints/small.pt
  python export.py --ckpt checkpoints/small_final.pt --out small_inference.pt
"""

import argparse
from pathlib import Path

import torch

# Everything sample.py reads, plus the two scalars worth keeping as provenance.
KEEP = ("model", "config", "meta", "iter", "best_val_loss")
REQUIRED = ("model", "config", "meta")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/small.pt")
    p.add_argument("--out", default=None, help="default: <ckpt stem>_inference.pt")
    args = p.parse_args()

    src = Path(args.ckpt)
    if not src.exists():
        raise SystemExit(f"checkpoint not found: {src}")
    dst = Path(args.out) if args.out else src.with_name(f"{src.stem}_inference.pt")

    ckpt = torch.load(src, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED if k not in ckpt]
    if missing:
        raise SystemExit(f"{src} is missing required keys: {missing}")

    torch.save({k: ckpt[k] for k in KEEP if k in ckpt}, dst)

    before = src.stat().st_size / 1e6
    after = dst.stat().st_size / 1e6
    print(f"{src} ({before:.1f} MB) -> {dst} ({after:.1f} MB, {after/before:.0%} of original)")


if __name__ == "__main__":
    main()
