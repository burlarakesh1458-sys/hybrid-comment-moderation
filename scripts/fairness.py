"""False-positive analysis for comments that mention identity groups.

Compares, on the full test split (and the LLM sample where the LLM-for-everything scenario
exists):
  * ML baseline                       (models/tfidf_lr.joblib, calib threshold)
  * ML + reweighing mitigation        (models/tfidf_lr_reweighted.joblib, its own calib threshold)
  * hybrid pipeline                   (from results/predictions_test.csv, if present)
  * LLM for everything                (sample only, from results/predictions_sample.csv)

Identity mentions use the Civil Comments human identity annotations (WILDS binarised columns),
never inferred from the text.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

from hybridmod.classifier import ToxicityClassifier  # noqa: E402
from hybridmod.config import load_config  # noqa: E402
from hybridmod.data import IDENTITY_COLS, load_split  # noqa: E402
from hybridmod.metrics import binary_metrics, subgroup_fpr  # noqa: E402


def per_group(df: pd.DataFrame, flag: np.ndarray, min_neg: int = 30) -> dict:
    out = {}
    y = df.toxicity.to_numpy()
    for g in IDENTITY_COLS:
        neg = (df[g].to_numpy() == 1) & (y == 0)
        if neg.sum() >= min_neg:
            out[g] = {"negatives": int(neg.sum()), "fpr": round(float(flag[neg].mean()), 4)}
    none = (df.identity_any.to_numpy() == 0) & (y == 0)
    out["no_identity"] = {"negatives": int(none.sum()), "fpr": round(float(flag[none].mean()), 4)}
    return out


def main() -> None:
    cfg = load_config()
    test = load_split("test")
    y = test.toxicity.to_numpy()
    ident = test.identity_any.to_numpy() == 1
    rep: dict = {"test_n": len(test), "identity_share": round(float(ident.mean()), 4),
                 "toxic_rate_identity": round(float(y[ident].mean()), 4),
                 "toxic_rate_no_identity": round(float(y[~ident].mean()), 4), "systems": {}}

    flags = {}
    for name, model_file, train_json in [
        ("ml_baseline", "models/tfidf_lr.joblib", "results/train_tfidf_lr.json"),
        ("ml_reweighted", "models/tfidf_lr_reweighted.joblib", "results/train_tfidf_lr_reweighted.json"),
    ]:
        thr = json.loads(cfg.path(train_json).read_text())["calib"]["best_threshold"]
        p = ToxicityClassifier.load(cfg.path(model_file)).predict_proba(list(test.comment_text))
        flags[name] = p >= thr
        rep["systems"][name] = {"threshold": thr, "overall": binary_metrics(y, flags[name]),
                                "identity": subgroup_fpr(y, flags[name], ident),
                                "per_group": per_group(test, flags[name])}
    pt = cfg.path("results/predictions_test.csv")
    if pt.exists():
        pr = pd.read_csv(pt)
        assert (pr.uid.to_numpy() == test.uid.to_numpy()).all()
        f = pr.hybrid_flag.to_numpy().astype(bool)
        flags["hybrid"] = f
        rep["systems"]["hybrid"] = {"overall": binary_metrics(y, f),
                                    "identity": subgroup_fpr(y, f, ident),
                                    "per_group": per_group(test, f)}
    ps = cfg.path("results/predictions_sample.csv")
    if ps.exists():
        s = pd.read_csv(ps)
        si = s.identity_any.to_numpy() == 1
        rep["sample"] = {"n": len(s)}
        for col in ["ml_flag", "llm_flag", "hybrid_flag"]:
            f = s[col].to_numpy().astype(bool)
            rep["sample"][col.replace("_flag", "")] = {
                "overall": binary_metrics(s.toxicity, f),
                "identity": subgroup_fpr(s.toxicity, f, si)}

    cfg.path("results/fairness.json").write_text(json.dumps(rep, indent=2) + "\n")

    groups = [g for g in IDENTITY_COLS + ["no_identity"]
              if g in rep["systems"]["ml_baseline"]["per_group"]]
    names = list(flags)
    colors = {"ml_baseline": "#6b7280", "ml_reweighted": "#2563eb", "hybrid": "#16a34a"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(names)
    x = np.arange(len(groups))
    for k, n in enumerate(names):
        vals = [rep["systems"][n]["per_group"][g]["fpr"] for g in groups]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=n, color=colors.get(n))
    ax.set_xticks(x, [f"{g}\n(n={rep['systems']['ml_baseline']['per_group'][g]['negatives']})"
                      for g in groups], fontsize=8)
    ax.set_ylabel("false-positive rate on non-toxic comments")
    ax.set_title(f"False positives by identity mentioned (test split, n={len(test)})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.path("results/charts/fpr_by_identity.png"), dpi=130)
    print(json.dumps({k: {"f1": v["overall"]["f1"], **{kk: v["identity"][kk] for kk in
                                                      ("fpr_gap", "fpr_ratio")},
                          "fpr_id": v["identity"]["mentions_identity"]["fpr"],
                          "fpr_no": v["identity"]["no_identity"]["fpr"]}
                      for k, v in rep["systems"].items()}, indent=2))


if __name__ == "__main__":
    main()
