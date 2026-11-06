"""Evaluation helpers: binary metrics, latency percentiles, cost projection, subgroup FPR."""

from __future__ import annotations

import numpy as np


def binary_metrics(y_true, y_pred) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    tp = int(((p == 1) & (y == 1)).sum())
    fp = int(((p == 1) & (y == 0)).sum())
    fn = int(((p == 0) & (y == 1)).sum())
    tn = int(((p == 0) & (y == 0)).sum())
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * prec * rec / (prec + rec) if prec + rec else 0.0
    fpr = fp / (fp + tn) if fp + tn else 0.0
    return {"n": int(len(y)), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
            "precision": round(prec, 4), "recall": round(rec, 4), "f1": round(f1, 4),
            "fpr": round(fpr, 4)}


def percentiles(values, qs=(50, 95)) -> dict:
    v = np.asarray(list(values), dtype=float)
    if v.size == 0:
        return {f"p{q}": None for q in qs}
    return {f"p{q}": round(float(np.percentile(v, q)), 3) for q in qs}


def cost_per_10k(prompt_tokens: float, completion_tokens: float, n_comments: int,
                 price_in_per_m: float, price_out_per_m: float) -> float:
    """Project the measured token usage on `n_comments` to 10,000 comments (USD)."""
    if n_comments == 0:
        return 0.0
    per_comment = (prompt_tokens * price_in_per_m + completion_tokens * price_out_per_m) / 1e6
    return round(per_comment / n_comments * 10_000, 4)


def best_f1_threshold(y_true, probs, grid=None) -> tuple[float, float]:
    y = np.asarray(y_true).astype(int)
    pr = np.asarray(probs, dtype=float)
    grid = np.round(np.arange(0.05, 0.96, 0.01), 2) if grid is None else grid
    best = (0.5, -1.0)
    for t in grid:
        f1 = binary_metrics(y, pr >= t)["f1"]
        if f1 > best[1]:
            best = (float(t), f1)
    return best


def subgroup_fpr(y_true, y_pred, mask) -> dict:
    y = np.asarray(y_true).astype(int)
    p = np.asarray(y_pred).astype(int)
    m = np.asarray(mask).astype(bool)
    out = {}
    for name, sel in (("mentions_identity", m), ("no_identity", ~m)):
        neg = sel & (y == 0)
        out[name] = {
            "n": int(sel.sum()), "negatives": int(neg.sum()),
            "fp": int((p[neg] == 1).sum()),
            "fpr": round(float((p[neg] == 1).mean()), 4) if neg.any() else None,
            "fnr": round(float((p[sel & (y == 1)] == 0).mean()), 4) if (sel & (y == 1)).any() else None,
        }
    a, b = out["mentions_identity"]["fpr"], out["no_identity"]["fpr"]
    out["fpr_gap"] = round(a - b, 4) if a is not None and b is not None else None
    out["fpr_ratio"] = round(a / b, 2) if a is not None and b else None
    return out
