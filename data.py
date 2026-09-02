"""Download and preprocess 全唐诗 + 全宋词.

Pulls from https://github.com/chinese-poetry/chinese-poetry, strips metadata,
normalizes the script, drops duplicate works, builds a character-level vocab with
a cut-off for rare characters, and saves binary .bin files for fast training.

Outputs:
  data/train.bin     — int16 array of token ids (train split)
  data/val.bin       — int16 array of token ids (val split)
  data/meta.pkl      — dict with itos, stoi, vocab_size and the preprocessing choices
"""

import argparse
import collections
import json
import pickle
import random
import re
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

# The upstream corpus is not in one script: the Tang poems are traditional, the
# Song ci simplified. Left as is, a character-level model learns "traditional means
# shi, simplified means ci" and mixes both scripts within one poem. Everything is
# therefore converted to a single script before the vocabulary is built.
DEFAULT_SCRIPT = "simplified"
SCRIPTS = ("simplified", "traditional", "mixed")

# Characters seen fewer times than this in the training text are mapped to one
# unknown token. Roughly two fifths of the raw vocabulary are such characters (used
# once or twice in six million), each costing an embedding row it can never learn.
DEFAULT_MIN_COUNT = 3
UNK_TOKEN = "□"

# Punctuation and whitespace are ignored when deciding whether two works are the
# same poem; the corpus repeats a fair number of poems with different punctuation.
_NON_CONTENT = re.compile(r"[\s\u3000-\u303f\uff00-\uffef.,;:!?'\"()\[\]{}<>/\\|~`@#$%^&*_+=-]+")
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


def get_converter(script: str):
    """An OpenCC converter for the requested script, or None for "mixed"."""
    if script not in SCRIPTS:
        raise ValueError(f"script must be one of {SCRIPTS}, got {script!r}")
    if script == "mixed":
        return None
    try:
        import opencc
    except ImportError as exc:  # pragma: no cover - depends on the environment
        raise SystemExit(
            "script normalization needs the opencc package: pip install opencc-python-reimplemented "
            "(or pass --script mixed to keep the corpus as it is)"
        ) from exc
    return opencc.OpenCC("t2s" if script == "simplified" else "s2t")


def normalize_script(text: str, converter) -> str:
    return text if converter is None else converter.convert(text)


def content_key(body: str) -> str:
    """The body with everything but the characters removed, for duplicate detection."""
    return _NON_CONTENT.sub("", body)


def dedupe_works(works: list[str]) -> tuple[list[str], int]:
    """Drop later copies of a work whose body already appeared. Returns (kept, dropped).

    Works are "header\nbody\n" strings; only the body counts, so the same poem filed
    under two titles or authors is still one poem. Without this, copies of one poem
    land on both sides of the train/validation split and flatter the validation loss.
    """
    seen = set()
    kept = []
    dropped = 0
    for work in works:
        body = work.split("\n", 1)[1] if "\n" in work else work
        key = content_key(body)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        kept.append(work)
    return kept, dropped


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
            paragraphs = item.get("paragraphs", [])
            body = "".join(paragraphs).strip()
            # The header is the tune name alone. With the author appended the model
            # learned to follow every tune with an invented name.
            if tune and body:
                ci.append(f"{tune}\n{body}\n")
    print(f"  loaded {len(ci)} Song ci")
    _check_complete("Song ci", skipped, 22, allow_partial)
    return ci


def build_vocab(text: str, min_count: int = 1) -> tuple[dict, dict]:
    """Character vocabulary over `text`, with UNK_TOKEN at id 0.

    Characters seen fewer than `min_count` times are left out and encode to the
    unknown token, as does anything in other text (the validation split, a prompt)
    that never appeared here.
    """
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    counts = collections.Counter(text)
    counts.pop(UNK_TOKEN, None)
    chars = sorted(ch for ch, n in counts.items() if n >= min_count)
    itos = {0: UNK_TOKEN}
    for ch in chars:
        itos[len(itos)] = ch
    stoi = {ch: i for i, ch in itos.items()}
    return stoi, itos


def encode(text: str, stoi: dict) -> np.ndarray:
    unk = stoi[UNK_TOKEN]
    return np.array([stoi.get(c, unk) for c in text], dtype=np.int16)


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
    parser.add_argument(
        "--script",
        choices=SCRIPTS,
        default=DEFAULT_SCRIPT,
        help="convert the whole corpus to one script (default: simplified); "
        "'mixed' keeps the upstream mixture of traditional Tang and simplified Song",
    )
    parser.add_argument(
        "--min-count",
        type=int,
        default=DEFAULT_MIN_COUNT,
        help="characters seen fewer times than this in the training text become the unknown token",
    )
    args = parser.parse_args()
    if args.min_count < 1:
        parser.error("--min-count must be at least 1")
    converter = get_converter(args.script)

    try:
        tang = load_tang_poems(args.allow_partial)
        song = load_song_ci(args.allow_partial)
    except RuntimeError as exc:
        print(f"ERROR: {exc}. Check the network and retry, or pass --allow-partial.")
        sys.exit(1)

    if not tang and not song:
        print("ERROR: no data downloaded. Check network and try again.")
        sys.exit(1)

    # One script for everything, then one copy of each poem.
    if converter is not None:
        print(f"Normalizing script to {args.script}...")
    all_poems = [normalize_script(work, converter) for work in tang + song]
    all_poems, duplicates = dedupe_works(all_poems)
    print(f"Dropped {duplicates:,} duplicate works; {len(all_poems):,} remain")

    # Shuffle and join with a separator
    random.Random(SPLIT_SEED).shuffle(all_poems)

    train_poems, val_poems = split_poems(all_poems)
    train_text = "\n".join(train_poems)
    val_text = "\n".join(val_poems)
    print(f"\nTotal characters: {len(train_text) + len(val_text):,}")
    print(f"Works: {len(train_poems):,} train  {len(val_poems):,} val")

    # The vocabulary comes from the training text only; rare characters and
    # anything the validation split adds map to the unknown token.
    stoi, itos = build_vocab(train_text, min_count=args.min_count)
    print(f"Vocab size: {len(stoi)} (characters seen fewer than {args.min_count} times -> {UNK_TOKEN})")
    if len(stoi) > np.iinfo(np.int16).max + 1:
        raise ValueError("vocabulary is too large for the int16 dataset format")

    # Encode
    train = encode(train_text, stoi)
    val = encode(val_text, stoi)
    unk = stoi[UNK_TOKEN]
    print(f"Encoded tokens: {len(train) + len(val):,}")
    print(f"Train: {len(train):,}  Val: {len(val):,}")
    print(f"Unknown-token rate: train {np.mean(train == unk):.3%}  val {np.mean(val == unk):.3%}")

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
                "script": args.script,
                "min_count": args.min_count,
                "unk_token": UNK_TOKEN,
                "unk_id": unk,
                "duplicates_dropped": duplicates,
            },
            f,
        )

    print(f"\nSaved to {DATA_DIR}/")
    print("  train.bin  val.bin  meta.pkl")


if __name__ == "__main__":
    main()
