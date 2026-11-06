"""Download the labelled Civil Comments (WILDS) data at a pinned revision and cut fixed splits.

Source: https://huggingface.co/datasets/pietrolesci/civilcomments-wilds (mirror of the WILDS
CivilComments release, Koh et al. 2021, built from the Jigsaw "Unintended Bias in Toxicity
Classification" / Civil Comments corpus, Borkan et al. 2019, released CC0 1.0).

Splits (all deterministic, seed 42):
  train   80,000 rows sampled from the WILDS train split
  calib   10,000 rows sampled from the rest of the WILDS train split (calibration + thresholds)
  test     5,000 rows sampled from the WILDS test split, exact-text duplicates of train/calib removed
  llm     the first N rows of a fixed permutation of `test` (N set in configs/moderation.yaml)

Only the row uids are committed (data/splits/*.txt); the text is re-downloaded on demand.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = "pietrolesci/civilcomments-wilds"
REVISION = "c227534cc0a34cf21db6a0bf0edd0f9050c8b305"
FILES = {
    "train": "data/train-00000-of-00001.parquet",
    "test": "data/test-00000-of-00001.parquet",
}
ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"

KEEP = [
    "uid", "comment_text", "toxicity", "severe_toxicity", "obscene", "threat", "insult",
    "identity_attack", "sexual_explicit", "identity_any", "male", "female", "LGBTQ",
    "christian", "muslim", "other_religions", "black", "white",
]


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def download() -> dict[str, Path]:
    from huggingface_hub import hf_hub_download

    out = {}
    for split, fname in FILES.items():
        out[split] = Path(
            hf_hub_download(REPO, fname, repo_type="dataset", revision=REVISION, local_dir=RAW)
        )
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-train", type=int, default=80_000)
    ap.add_argument("--n-calib", type=int, default=10_000)
    ap.add_argument("--n-test", type=int, default=5_000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    paths = download()
    tr = pd.read_parquet(paths["train"], columns=KEEP)
    te = pd.read_parquet(paths["test"], columns=KEEP)
    tr = tr[tr.comment_text.str.strip().str.len() > 0].drop_duplicates("comment_text")
    te = te[te.comment_text.str.strip().str.len() > 0].drop_duplicates("comment_text")

    rng = np.random.default_rng(args.seed)
    idx = rng.permutation(len(tr))
    train = tr.iloc[idx[: args.n_train]]
    calib = tr.iloc[idx[args.n_train : args.n_train + args.n_calib]]

    seen = set(train.comment_text) | set(calib.comment_text)
    te = te[~te.comment_text.isin(seen)]
    test = te.iloc[rng.permutation(len(te))[: args.n_test]]
    # fixed order for the LLM sample: a permutation of the test set, first N used
    llm_order = test.uid.to_numpy()[np.random.default_rng(7).permutation(len(test))]

    PROC.mkdir(parents=True, exist_ok=True)
    SPLITS.mkdir(parents=True, exist_ok=True)
    stats = {}
    for name, df in [("train", train), ("calib", calib), ("test", test)]:
        df.reset_index(drop=True).to_parquet(PROC / f"{name}.parquet", index=False)
        (SPLITS / f"{name}_uids.txt").write_text("\n".join(map(str, df.uid)) + "\n")
        stats[name] = {
            "rows": int(len(df)),
            "toxic_rate": round(float(df.toxicity.mean()), 4),
            "identity_any_rate": round(float(df.identity_any.mean()), 4),
        }
    (SPLITS / "llm_order_uids.txt").write_text("\n".join(map(str, llm_order)) + "\n")

    prov = {
        "dataset": REPO,
        "revision": REVISION,
        "url": f"https://huggingface.co/datasets/{REPO}/tree/{REVISION}",
        "upstream": [
            "WILDS CivilComments (Koh et al., 2021) https://wilds.stanford.edu/datasets/#civilcomments",
            "Jigsaw Unintended Bias in Toxicity Classification / Civil Comments (Borkan et al., 2019)",
        ],
        "license": "CC0 1.0 (Civil Comments corpus as released by Jigsaw/Google; see "
        "https://huggingface.co/datasets/google/civil_comments). The mirror repo itself declares no "
        "license; its processing_script.py binarises labels at >= 0.5 and removes duplicates.",
        "label_definition": "toxicity and subtype columns are the fraction of crowd raters who "
        "said so, binarised at >= 0.5 by the WILDS processing.",
        "files_sha256": {k: sha256(p) for k, p in paths.items()},
        "seed": args.seed,
        "splits": stats,
    }
    (ROOT / "data" / "PROVENANCE.json").write_text(json.dumps(prov, indent=2) + "\n")
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
