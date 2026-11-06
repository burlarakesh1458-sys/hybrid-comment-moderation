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
        # hybrid predictions may cover a subset of the test split: compare like with like
        pr = pd.read_csv(pt)
        pos = pd.Series(np.arange(len(test)), index=test.uid).loc[pr.uid].to_numpy()
        sub = test.iloc[pos].reset_index(drop=True)
        ys, idn = sub.toxicity.to_numpy(), sub.identity_any.to_numpy() == 1
        rep["hybrid_subset"] = {"n": len(sub)}
        for name, f in [("ml_baseline", flags["ml_baseline"][pos]),
                        ("ml_reweighted", flags["ml_reweighted"][pos]),
                        ("hybrid", pr.hybrid_flag.to_numpy().astype(bool))]:
            rep["hybrid_subset"][name] = {"overall": binary_metrics(ys, f),
                                          "identity": subgroup_fpr(ys, f, idn),
                                          "per_group": per_group(sub, f)}
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

    src = rep.get("hybrid_subset") or rep["systems"]
    n_chart = rep["hybrid_subset"]["n"] if "hybrid_subset" in rep else len(test)
    names = [k for k in ("ml_baseline", "ml_reweighted", "hybrid") if k in src]
    groups = [g for g in IDENTITY_COLS + ["no_identity"] if g in src["ml_baseline"]["per_group"]
              and all(g in src[n]["per_group"] for n in names)]
    colors = {"ml_baseline": "#6b7280", "ml_reweighted": "#2563eb", "hybrid": "#16a34a"}
    fig, ax = plt.subplots(figsize=(11, 4.2))
    w = 0.8 / len(names)
    x = np.arange(len(groups))
    for k, n in enumerate(names):
        vals = [src[n]["per_group"][g]["fpr"] for g in groups]
        ax.bar(x + k * w - 0.4 + w / 2, vals, w, label=n, color=colors.get(n))
    ax.set_xticks(x, [f"{g}\n(n={src['ml_baseline']['per_group'][g]['negatives']})"
                      for g in groups], fontsize=8)
    ax.set_ylabel("false-positive rate on non-toxic comments")
    ax.set_title(f"False positives on non-toxic comments, by identity mentioned (test, n={n_chart})")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(cfg.path("results/charts/fpr_by_identity.png"), dpi=130)
    def brief(d):
        return {k: {"f1": v["overall"]["f1"], "fpr_id": v["identity"]["mentions_identity"]["fpr"],
                    "fpr_no": v["identity"]["no_identity"]["fpr"],
                    "gap": v["identity"]["fpr_gap"], "ratio": v["identity"]["fpr_ratio"]}
                for k, v in d.items() if isinstance(v, dict) and "overall" in v}
    print(json.dumps({"full_test": brief(rep["systems"]),
                      "hybrid_subset": brief(rep.get("hybrid_subset", {})),
                      "sample": brief(rep.get("sample", {}))}, indent=2))


if __name__ == "__main__":
    main()
