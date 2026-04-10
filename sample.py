"""Generate poetry from a trained tiny-poet checkpoint.

Usage:
  python sample.py --ckpt checkpoints/small.pt --prompt "春眠不觉晓" --max_tokens 60
  python sample.py --ckpt checkpoints/small.pt --prompt "蝶恋花" --temperature 0.8 --top_k 40
"""

import argparse
from pathlib import Path

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/small.pt")
    p.add_argument("--prompt", default="春")
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_samples", type=int, default=3)
    args = p.parse_args()

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = ckpt["meta"]
    stoi, itos = meta["stoi"], meta["itos"]

    # Filter prompt chars not in vocab
    prompt_ids = [stoi[c] for c in args.prompt if c in stoi]
    if not prompt_ids:
        print(f"Warning: no chars from '{args.prompt}' found in vocab, starting with '春'")
        prompt_ids = [stoi.get("春", 0)]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1} ---")
        out = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
        text = "".join(itos[int(t)] for t in out[0].tolist())
        print(text)


if __name__ == "__main__":
    main()
