"""Choose the LLM confidence band on the CALIBRATION split only (no test data, no LLM results).

The band is the probability interval around the decision threshold that captures a target share
of calibration comments (default 10%: half just below the threshold, half just above). It is the
region where the calibrated classifier is least sure; outside it the ML decision is final.
The chosen values are written back into configs/moderation.yaml.
"""

from __future__ import annotations

import argparse
import json
import re

import numpy as np

from hybridmod.config import DEFAULT_CONFIG, load_config
from hybridmod.data import load_split
from hybridmod.metrics import binary_metrics


def band_for_budget(probs: np.ndarray, threshold: float, share: float) -> tuple[float, float]:
    below = np.sort(probs[probs < threshold])[::-1]
    above = np.sort(probs[probs >= threshold])
    k = int(round(len(probs) * share / 2))
    lo = float(below[min(k, len(below)) - 1]) if len(below) else threshold
    hi = float(above[min(k, len(above)) - 1]) if len(above) else threshold
    return round(lo, 4), round(hi + 1e-4, 4)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--share", type=float, default=0.10)
    ap.add_argument("--write", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    train_rep = json.loads(cfg.path("results/train_tfidf_lr.json").read_text())
    thr = train_rep["calib"]["best_threshold"]
    probs = np.load(cfg.path("results/calib_probs_tfidf_lr.npy"))
    y = load_split("calib").toxicity.to_numpy()
    lo, hi = band_for_budget(probs, thr, args.share)
    inband = (probs >= lo) & (probs < hi)
    rep = {
        "threshold": thr, "band_low": lo, "band_high": hi, "target_share": args.share,
        "calib_share_in_band": round(float(inband.mean()), 4),
        "calib_ml_metrics_inside_band": binary_metrics(y[inband], probs[inband] >= thr),
        "calib_ml_metrics_outside_band": binary_metrics(y[~inband], probs[~inband] >= thr),
    }
    print(json.dumps(rep, indent=2))
    cfg.path("results/band_tuning.json").write_text(json.dumps(rep, indent=2) + "\n")
    if args.write:
        s = DEFAULT_CONFIG.read_text()
        s = re.sub(r"(\n  threshold: )[0-9.]+", rf"\g<1>{thr}", s)
        s = re.sub(r"(\n  band_low: )[0-9.]+", rf"\g<1>{lo}", s)
        s = re.sub(r"(\n  band_high: )[0-9.]+", rf"\g<1>{hi}", s)
        DEFAULT_CONFIG.write_text(s)


if __name__ == "__main__":
    main()
