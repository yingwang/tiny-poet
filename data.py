"""Download and preprocess 全唐诗 + 全宋词.

Pulls from https://github.com/chinese-poetry/chinese-poetry, strips metadata,
builds a character-level vocab, and saves as binary .bin files for fast training.

Outputs:
  data/train.bin     — int16 array of token ids (train split)
  data/val.bin       — int16 array of token ids (val split)
  data/meta.pkl      — dict with itos, stoi, vocab_size
"""

import argparse
import json
import pickle
import random
import sys
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)
# Pin the corpus so different runs build the same dataset. Keep each revision in
# its own cache directory so data downloaded by older versions is never reused
# accidentally after the pin changes.
REPO_REF = "b8594f81a89752241442f2ce267d6f66f96704ee"
SPLIT_SEED = 42
TRAIN_FRACTION = 0.95
RAW_DIR = DATA_DIR / "raw" / REPO_REF[:12]
RAW_DIR.mkdir(parents=True, exist_ok=True)

REPO = f"https://raw.githubusercontent.com/chinese-poetry/chinese-poetry/{REPO_REF}"


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
        # urlretrieve streams straight into dest, so an interrupted download
        # leaves a truncated file behind. Delete it, or the next run sees
        # dest.exists() and happily skips re-fetching a corrupt file.
        dest.unlink(missing_ok=True)
        return False


def load_json(url: str, dest: Path):
    """Fetch and parse JSON, retrying one failed download or corrupt cache."""
    for attempt in (1, 2):
        if not fetch(url, dest):
            if attempt == 1:
                print(f"  ! retrying {dest.name}")
                continue
            return None
        try:
            return json.loads(dest.read_text(encoding="utf-8"))
        except Exception as e:
            dest.unlink(missing_ok=True)
            if attempt == 1:
                print(f"  ! {dest.name} was corrupt ({e}); re-downloading")
            else:
                print(f"  ! {dest.name} still unreadable ({e}); skipping it")
    return None


def _check_complete(label: str, skipped: int, expected: int, allow_partial: bool) -> None:
    if not skipped:
        return
    message = f"{skipped} of {expected} {label} files could not be read"
    print(f"  WARNING: {message}")
    if not allow_partial:
        raise RuntimeError(f"{message}; refusing to build a partial dataset")


def load_tang_poems(allow_partial: bool = False) -> list[str]:
    """Load 全唐诗 — 58 files, poet.tang.0.json to poet.tang.57000.json."""
    print("Loading 全唐诗...")
    poems = []
    skipped = 0
    for i in range(0, 58000, 1000):
        fname = f"poet.tang.{i}.json"
        url = f"{REPO}/全唐诗/{fname}"
        items = load_json(url, RAW_DIR / fname)
        if items is None:
            skipped += 1
            continue
        for item in items:
            title = item.get("title", "").strip()
            paragraphs = item.get("paragraphs", [])
            body = "".join(paragraphs).strip()
            if title and body:
                poems.append(f"{title}\n{body}\n")
    print(f"  loaded {len(poems)} Tang poems")
    _check_complete("Tang", skipped, 58, allow_partial)
    return poems


def load_song_ci(allow_partial: bool = False) -> list[str]:
    """Load 全宋词 — files are ci.song.0.json to ci.song.21000.json."""
    print("Loading 全宋词...")
    ci = []
    skipped = 0
    for i in range(0, 22000, 1000):
        fname = f"ci.song.{i}.json"
        url = f"{REPO}/宋词/{fname}"
        items = load_json(url, RAW_DIR / fname)
        if items is None:
            skipped += 1
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
    _check_complete("Song ci", skipped, 22, allow_partial)
    return ci


def build_vocab(text: str) -> tuple[dict, dict]:
    chars = sorted(set(text))
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}
    return stoi, itos


def split_poems(
    poems: list[str], train_fraction: float = TRAIN_FRACTION
) -> tuple[list[str], list[str]]:
    """Split whole works, never cutting a poem between train and validation."""
    if len(poems) < 2:
        raise ValueError("at least two poems are required for a train/validation split")
    if not 0 < train_fraction < 1:
        raise ValueError("train_fraction must be between 0 and 1")
    split_at = max(1, min(len(poems) - 1, int(train_fraction * len(poems))))
    return poems[:split_at], poems[split_at:]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--allow-partial",
        action="store_true",
        help="build even if one or more corpus files cannot be downloaded",
    )
    args = parser.parse_args()

    try:
        tang = load_tang_poems(args.allow_partial)
        song = load_song_ci(args.allow_partial)
    except RuntimeError as exc:
        print(f"ERROR: {exc}. Check the network and retry, or pass --allow-partial.")
        sys.exit(1)

    if not tang and not song:
        print("ERROR: no data downloaded. Check network and try again.")
        sys.exit(1)

    # Shuffle and join with a separator
    all_poems = tang + song
    random.Random(SPLIT_SEED).shuffle(all_poems)

    train_poems, val_poems = split_poems(all_poems)
    train_text = "\n".join(train_poems)
    val_text = "\n".join(val_poems)
    print(f"\nTotal characters: {len(train_text) + len(val_text):,}")
    print(f"Works: {len(train_poems):,} train  {len(val_poems):,} val")

    # Build one shared vocabulary, while keeping the actual sequences separate.
    stoi, itos = build_vocab(train_text + val_text)
    print(f"Vocab size: {len(stoi)}")
    if len(stoi) > np.iinfo(np.int16).max + 1:
        raise ValueError("vocabulary is too large for the int16 dataset format")

    # Encode
    train = np.array([stoi[c] for c in train_text], dtype=np.int16)
    val = np.array([stoi[c] for c in val_text], dtype=np.int16)
    print(f"Encoded tokens: {len(train) + len(val):,}")
    print(f"Train: {len(train):,}  Val: {len(val):,}")

    train.tofile(DATA_DIR / "train.bin")
    val.tofile(DATA_DIR / "val.bin")

    with open(DATA_DIR / "meta.pkl", "wb") as f:
        pickle.dump(
            {
                "stoi": stoi,
                "itos": itos,
                "vocab_size": len(stoi),
                "corpus_ref": REPO_REF,
                "split_seed": SPLIT_SEED,
                "train_fraction": TRAIN_FRACTION,
                "train_tokens": len(train),
                "val_tokens": len(val),
            },
            f,
        )

    print(f"\nSaved to {DATA_DIR}/")
    print("  train.bin  val.bin  meta.pkl")


if __name__ == "__main__":
    main()
