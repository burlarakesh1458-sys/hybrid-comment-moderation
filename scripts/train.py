"""Train the stage-2 classifier, pick the decision threshold on the calibration split.

  python scripts/train.py                 # baseline  -> models/tfidf_lr.joblib
  python scripts/train.py --reweight      # fairness mitigation -> models/tfidf_lr_reweighted.joblib

--reweight applies reweighing (Kamiran & Calders, 2012) on the identity-mention attribute: non-toxic
training comments that mention an identity group are up-weighted so that, in the weighted training
data, mentioning an identity group is no longer correlated with the toxic label. The attribute is
used only at training time; inference never sees it.
"""

from __future__ import annotations

import argparse
import json
import time

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

from hybridmod.classifier import ClfParams, ToxicityClassifier, fit
from hybridmod.config import load_config
from hybridmod.data import load_split
from hybridmod.metrics import best_f1_threshold, binary_metrics


def reweigh(y: np.ndarray, a: np.ndarray) -> np.ndarray:
    """w(a, y) = P(a) P(y) / P(a, y): makes y independent of a in the weighted data."""
    w = np.ones(len(y), dtype=float)
    for av in (0, 1):
        for yv in (0, 1):
            m = (a == av) & (y == yv)
            if m.any():
                w[m] = (a == av).mean() * (y == yv).mean() / m.mean()
    return w


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reweight", action="store_true")
    ap.add_argument("--C", type=float, default=2.0)
    args = ap.parse_args()
    cfg = load_config()
    tr, ca = load_split("train"), load_split("calib")
    y = tr.toxicity.to_numpy()
    sw = reweigh(y, tr.identity_any.to_numpy()) if args.reweight else None
    p = ClfParams(C=args.C)
    t0 = time.time()
    model = fit(tr.comment_text, y, p, sample_weight=sw)
    train_s = time.time() - t0
    clf = ToxicityClassifier(model)
    name = "tfidf_lr_reweighted" if args.reweight else "tfidf_lr"
    path = cfg.path(f"models/{name}.joblib")
    clf.save(path)

    prob = clf.predict_proba(list(ca.comment_text))
    thr, f1 = best_f1_threshold(ca.toxicity, prob)
    report = {
        "model": name, "params": {k: v for k, v in p.__dict__.items() if k != "extra"},
        "reweight": args.reweight, "train_rows": len(tr), "train_seconds": round(train_s, 1),
        "model_mb": round(path.stat().st_size / 1e6, 1),
        "calib": {
            "rows": len(ca), "roc_auc": round(roc_auc_score(ca.toxicity, prob), 4),
            "avg_precision": round(average_precision_score(ca.toxicity, prob), 4),
            "best_threshold": thr, **binary_metrics(ca.toxicity, prob >= thr),
        },
    }
    if sw is not None:
        report["weights"] = {f"identity={a},toxic={t}": round(float(sw[(tr.identity_any == a)
                             & (tr.toxicity == t)][0]), 3) for a in (0, 1) for t in (0, 1)}
    out = cfg.path(f"results/train_{name}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n")
    np.save(cfg.path(f"results/calib_probs_{name}.npy"), prob.astype(np.float32))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
