"""Build the hashed block lexicon used by stage-1 rules.

Data-driven part: normalised tokens that appear in >= MIN_SUPPORT training comments and whose
training-set toxic rate is >= MIN_PRECISION. Optional curated part: a plaintext file passed with
--extra (kept OUT of the repo) for strong slurs that are too rare in Civil Comments to be learned
(the platform was pre-moderated). Only salted SHA-256 hashes are written.
"""

from __future__ import annotations

import argparse
import collections
from pathlib import Path

import pandas as pd

from hybridmod.config import load_config
from hybridmod.rules import hash_token, normalise_token, tokens

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--min-support", type=int, default=20)
    ap.add_argument("--min-precision", type=float, default=0.85)
    ap.add_argument("--extra", type=Path, default=None)
    args = ap.parse_args()
    cfg = load_config()

    df = pd.read_parquet(ROOT / "data/processed/train.parquet", columns=["comment_text", "toxicity"])
    n = collections.Counter()
    pos = collections.Counter()
    for text, y in zip(df.comment_text, df.toxicity, strict=True):
        for t in set(tokens(text)):
            n[t] += 1
            pos[t] += int(y)
    learned = sorted(
        t for t in n if n[t] >= args.min_support and pos[t] / n[t] >= args.min_precision
    )
    extra = []
    if args.extra and args.extra.exists():
        extra = [normalise_token(w) for w in args.extra.read_text().split() if w.strip()]
    words = sorted(set(learned) | set(extra))
    out = cfg.path(cfg.rules.lexicon_path)
    lines = [
        "# salted sha256 of normalised block tokens; see scripts/build_lexicon.py",
        f"# learned={len(learned)} (support>={args.min_support}, train precision>="
        f"{args.min_precision}) curated_extra={len(set(extra) - set(learned))}",
    ]
    lines += [hash_token(w, cfg.rules.lexicon_salt) for w in words]
    out.write_text("\n".join(lines) + "\n")
    print(f"wrote {len(words)} hashes -> {out} (learned {len(learned)})")


if __name__ == "__main__":
    main()
