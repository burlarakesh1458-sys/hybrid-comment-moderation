"""Load the prepared splits (see scripts/prepare_data.py)."""

from __future__ import annotations

import pandas as pd

from .config import ROOT

PROC = ROOT / "data" / "processed"
SPLITS = ROOT / "data" / "splits"
IDENTITY_COLS = ["male", "female", "LGBTQ", "christian", "muslim", "other_religions", "black", "white"]


def load_split(name: str) -> pd.DataFrame:
    path = PROC / f"{name}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} missing - run `make data` (scripts/prepare_data.py)")
    return pd.read_parquet(path)


def llm_sample(n: int) -> pd.DataFrame:
    """The fixed head-to-head sample: first n uids of a seeded permutation of the test split."""
    test = load_split("test").set_index("uid")
    order = [int(x) for x in (SPLITS / "llm_order_uids.txt").read_text().split()]
    return test.loc[order[:n]].reset_index()
