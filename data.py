"""Download and preprocess 全唐诗 + 全宋词.

Pulls from https://github.com/chinese-poetry/chinese-poetry, strips metadata,
builds a character-level vocab, and saves as binary .bin files for fast training.

Outputs:
  data/train.bin     — int16 array of token ids (train split)
  data/val.bin       — int16 array of token ids (val split)
  data/meta.pkl      — dict with itos, stoi, vocab_size
"""

import json
import os
import pickle
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
RAW_DIR = DATA_DIR / "raw"
RAW_DIR.mkdir(exist_ok=True)

REPO = "https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/master"


def fetch(url: str, dest: Path) -> bool:
    if dest.exists():
        return True
    try:
        print(f"  fetching {url.rsplit('/', 1)[-1]}...", end=" ", flush=True)
        # URL-encode path (safe for non-ASCII segments like 全唐诗)
        parsed = urllib.parse.urlsplit(url)
        safe_path = urllib.parse.quote(parsed.path)
        safe_url = urllib.parse.urlunsplit((parsed.scheme, parsed.netloc, safe_path, parsed.query, parsed.fragment))
        urllib.request.urlretrieve(safe_url, dest)
        print("ok")
        return True
    except Exception as e:
        print(f"failed: {e}")
        return False


def load_tang_poems() -> list[str]:
    """Load 全唐诗 — 58 files, poet.tang.0.json to poet.tang.57000.json."""
    print("Loading 全唐诗...")
    poems = []
    for i in range(0, 58000, 1000):
        fname = f"poet.tang.{i}.json"
        url = f"{REPO}/全唐诗/{fname}"
        local = RAW_DIR / fname
        if not fetch(url, local):
            continue
        try:
            items = json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in items:
            title = item.get("title", "").strip()
            paragraphs = item.get("paragraphs", [])
            body = "".join(paragraphs).strip()
            if title and body:
                poems.append(f"{title}\n{body}\n")
    print(f"  loaded {len(poems)} Tang poems")
    return poems


def load_song_ci() -> list[str]:
    """Load 全宋词 — files are ci.song.0.json to ci.song.21000.json."""
    print("Loading 全宋词...")
    ci = []
    for i in range(0, 22000, 1000):
        fname = f"ci.song.{i}.json"
        url = f"{REPO}/宋词/{fname}"
        local = RAW_DIR / fname
        if not fetch(url, local):
            continue
        try:
            items = json.loads(local.read_text(encoding="utf-8"))
        except Exception:
            continue
        for item in items:
            tune = item.get("rhythmic", "").strip()
            author = item.get("author", "").strip()
            paragraphs = item.get("paragraphs", [])
            body = "".join(paragraphs).strip()
            if tune and body:
                header = f"{tune}·{author}" if author else tune
                ci.append(f"{header}\n{body}\n")
    print(f"  loaded {len(ci)} Song ci")
    return ci


def build_vocab(text: str) -> tuple[dict, dict]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def main():
    tang = load_tang_poems()
    song = load_song_ci()

    if not tang and not song:
        print("ERROR: no data downloaded. Check network and try again.")
        sys.exit(1)

    # Shuffle and join with a separator
    all_poems = tang + song
    random.seed(42)
    random.shuffle(all_poems)

    text = "\n".join(all_poems)
    print(f"\nTotal characters: {len(text):,}")

    stoi, itos = build_vocab(text)
    print(f"Vocab size: {len(stoi)}")

    # Encode
    data = np.array([stoi[c] for c in text], dtype=np.int16)
    print(f"Encoded tokens: {len(data):,}")

    # 95 / 5 train / val split
    n = int(0.95 * len(data))
    train, val = data[:n], data[n:]
    print(f"Train: {len(train):,}  Val: {len(val):,}")

    train.tofile(DATA_DIR / "train.bin")
    val.tofile(DATA_DIR / "val.bin")

    with open(DATA_DIR / "meta.pkl", "wb") as f:
        pickle.dump({"stoi": stoi, "itos": itos, "vocab_size": len(stoi)}, f)

    print(f"\nSaved to {DATA_DIR}/")
    print("  train.bin  val.bin  meta.pkl")


if __name__ == "__main__":
    main()
