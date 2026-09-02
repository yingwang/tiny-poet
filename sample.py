"""Generate poetry from a trained tiny-poet checkpoint.

Usage:
  python sample.py --ckpt checkpoints/small_best.pt --prompt "春眠不觉晓" --max_tokens 60
  python sample.py --ckpt checkpoints/small_best.pt --prompt "蝶恋花" --temperature 0.8 --top_k 40
"""

import argparse

import torch

from model import GPT, GPTConfig


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="checkpoints/small_best.pt")
    p.add_argument("--prompt", default="春")
    p.add_argument("--max_tokens", type=int, default=60)
    p.add_argument("--temperature", type=float, default=0.9)
    p.add_argument("--top_k", type=int, default=40)
    p.add_argument("--device", default="auto")
    p.add_argument("--num_samples", type=int, default=3)
    p.add_argument("--seed", type=int, default=None)
    args = p.parse_args()

    if args.max_tokens < 0:
        p.error("--max_tokens must be non-negative")
    if args.temperature < 0:
        p.error("--temperature must be non-negative; use 0 for greedy decoding")
    if args.top_k <= 0:
        p.error("--top_k must be positive")
    if args.num_samples <= 0:
        p.error("--num_samples must be positive")

    if args.device == "auto":
        if torch.cuda.is_available():
            args.device = "cuda"
        elif torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"
    if args.seed is not None:
        torch.manual_seed(args.seed)

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=True)
    config = GPTConfig(**ckpt["config"])
    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model"])
    model.eval()

    meta = ckpt["meta"]
    stoi, itos = meta["stoi"], meta["itos"]

    # A checkpoint trained on a normalized corpus expects prompts in that script;
    # convert so a traditional prompt to a simplified model is not thrown away as
    # out-of-vocabulary characters.
    prompt = args.prompt
    script = meta.get("script")
    if script in ("simplified", "traditional"):
        try:
            import opencc

            prompt = opencc.OpenCC("t2s" if script == "simplified" else "s2t").convert(prompt)
        except ImportError:
            print(f"Warning: model expects {script} characters; install opencc-python-reimplemented to convert prompts")
    if prompt != args.prompt:
        print(f"Prompt converted to {script}: {prompt}")

    # Filter prompt chars not in vocab, but report any semantic change instead
    # of silently turning (for example) "春🙂天" into "春天".
    unknown_chars = sorted({char for char in prompt if char not in stoi})
    if unknown_chars:
        print(f"Warning: dropping out-of-vocabulary prompt chars: {unknown_chars}")
    prompt_ids = [stoi[c] for c in prompt if c in stoi]
    if not prompt_ids:
        print(f"Warning: no chars from '{prompt}' found in vocab, starting with '春'")
        prompt_ids = [stoi.get("春", 0)]

    x = torch.tensor([prompt_ids], dtype=torch.long, device=args.device)

    for i in range(args.num_samples):
        print(f"\n--- Sample {i+1} ---")
        out = model.generate(x, max_new_tokens=args.max_tokens, temperature=args.temperature, top_k=args.top_k)
        text = "".join(itos[int(t)] for t in out[0].tolist())
        print(text)


if __name__ == "__main__":
    main()
