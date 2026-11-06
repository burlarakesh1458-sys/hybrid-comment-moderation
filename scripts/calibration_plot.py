"""Reliability diagram: why the classifier is calibrated before its scores define the LLM band.

Trains the same TF-IDF + LR without calibration for comparison (not saved), then plots both
against the committed calibrated model on the test split. Writes results/calibration.json and
results/charts/calibration.png.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from sklearn.calibration import calibration_curve  # noqa: E402
from sklearn.metrics import brier_score_loss, roc_auc_score  # noqa: E402

from hybridmod.classifier import ClfParams, ToxicityClassifier, fit  # noqa: E402
from hybridmod.config import load_config  # noqa: E402
from hybridmod.data import load_split  # noqa: E402


def ece(y, p, bins=10) -> float:
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    return float(sum(abs(y[idx == b].mean() - p[idx == b].mean()) * (idx == b).mean()
                     for b in range(bins) if (idx == b).any()))


def main() -> None:
    cfg = load_config()
    tr, te = load_split("train"), load_split("test")
    y = te.toxicity.to_numpy()
    cal = ToxicityClassifier.load(cfg.path(cfg.classifier.model_path)).predict_proba(list(te.comment_text))
    raw_model = fit(tr.comment_text, tr.toxicity, ClfParams(C=2.0, calibration="none"))
    raw = raw_model.predict_proba(list(te.comment_text))[:, 1]
    rep = {}
    fig, ax = plt.subplots(figsize=(5.6, 5))
    ax.plot([0, 1], [0, 1], "k:", lw=1, label="perfect")
    for name, p, c in (("uncalibrated LR", raw, "#9ca3af"), ("isotonic-calibrated LR", cal, "#2563eb")):
        fx, fy = calibration_curve(y, p, n_bins=10, strategy="quantile")
        ax.plot(fy, fx, "o-", color=c, label=f"{name} (ECE {ece(y, p):.3f})")
        rep[name] = {"brier": round(brier_score_loss(y, p), 4), "ece": round(ece(y, p), 4),
                     "roc_auc": round(roc_auc_score(y, p), 4), "mean_p": round(float(p.mean()), 4)}
    rep["test_toxic_rate"] = round(float(y.mean()), 4)
    ax.set_xlabel("mean predicted P(toxic)")
    ax.set_ylabel("observed toxic rate")
    ax.set_title(f"Reliability on the test split (n={len(y)})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(cfg.path("results/charts/calibration.png"), dpi=130)
    cfg.path("results/calibration.json").write_text(json.dumps(rep, indent=2) + "\n")
    print(json.dumps(rep, indent=2))


if __name__ == "__main__":
    main()
