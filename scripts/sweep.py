"""Confidence-band sweep: LLM cost vs F1 trade-off.

Runs on the fixed LLM sample, where EVERY comment already has a cached LLM verdict, so any band
can be simulated exactly without new API calls. This is analysis, not selection: the production
band was fixed beforehand on the calibration split (scripts/tune_band.py).

Two views:
  1. budget curve - band widths defined on the calibration split (share of comments routed:
     0% = ML only ... 100% = LLM for everything after rules), applied to the sample.
  2. grid - every (band_low, band_high) pair on a 0.05 grid; Pareto front of cost vs F1.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import tune_band  # noqa: E402  (scripts/ on sys.path when run as a script)

from hybridmod.classifier import ToxicityClassifier  # noqa: E402
from hybridmod.config import load_config  # noqa: E402
from hybridmod.data import llm_sample  # noqa: E402
from hybridmod.llm import JsonlCache, cache_key  # noqa: E402
from hybridmod.metrics import binary_metrics  # noqa: E402
from hybridmod.rules import RuleEngine  # noqa: E402


def main() -> None:
    cfg = load_config()
    c = cfg.classifier
    df = llm_sample(cfg.experiment.llm_sample_size)
    y = df.toxicity.to_numpy()
    texts = list(df.comment_text)
    rules = RuleEngine.from_file(cfg.rules, cfg.path(cfg.rules.lexicon_path))
    racts = [rules.evaluate(t).action for t in texts]
    clf = ToxicityClassifier.load(cfg.path(c.model_path))
    p = clf.predict_proba(texts)
    cache = JsonlCache(cfg.path(cfg.llm.cache_path))
    recs = [cache.get(cache_key(cfg.llm.model, t)) for t in texts]
    assert all(r and r.get("verdict") for r in recs), "run scripts/run_llm_sample.py first"
    llm_v = np.array([r["verdict"]["violates"] for r in recs])
    tok_cost = np.array([(r["prompt_tokens"] * cfg.llm.price_input_per_m
                          + r["completion_tokens"] * cfg.llm.price_output_per_m) / 1e6 for r in recs])
    ract = np.array(racts)
    passes = ract == "pass"
    rule_flag = ract == "block"

    def simulate(lo: float, hi: float) -> dict:
        band = passes & (p >= lo) & (p < hi)
        flag = np.where(passes, np.where(band, llm_v, p >= c.threshold), rule_flag)
        m = binary_metrics(y, flag)
        return {"band_low": lo, "band_high": hi, "routed_share": round(float(band.mean()), 4),
                "cost_per_10k": round(float(tok_cost[band].sum() / len(y) * 10_000), 4),
                "f1": m["f1"], "precision": m["precision"], "recall": m["recall"],
                "fp": m["fp"], "fn": m["fn"]}

    calib_p = np.load(cfg.path("results/calib_probs_tfidf_lr.npy"))
    budget = []
    for share in [0, 0.02, 0.05, 0.075, 0.10, 0.15, 0.20, 0.30, 0.40, 0.60, 0.80, 1.0]:
        if share == 0:
            lo, hi = c.threshold, c.threshold
        elif share >= 1.0:
            lo, hi = 0.0, 1.01
        else:
            lo, hi = tune_band.band_for_budget(calib_p, c.threshold, share)
        budget.append({"calib_share": share, **simulate(lo, hi)})

    grid_vals = np.round(np.arange(0.0, 1.0001, 0.05), 2)
    grid = [simulate(float(lo), float(hi)) for lo in grid_vals for hi in grid_vals if hi > lo]
    grid.sort(key=lambda r: r["cost_per_10k"])
    pareto, best = [], -1.0
    for r in grid:
        if r["f1"] > best:
            pareto.append(r)
            best = r["f1"]
    chosen = simulate(c.band_low, c.band_high)
    ml_only = simulate(c.threshold, c.threshold)
    llm_all_flag = llm_v
    llm_all = {"f1": binary_metrics(y, llm_all_flag)["f1"],
               "cost_per_10k": round(float(tok_cost.sum() / len(y) * 10_000), 4)}

    out = {"n": len(y), "chosen": chosen, "ml_only": ml_only, "llm_all": llm_all,
           "budget_curve": budget, "pareto": pareto, "grid_size": len(grid)}
    cfg.path("results/band_sweep.json").write_text(json.dumps(out, indent=2) + "\n")

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    ax = axes[0]
    xs = [b["routed_share"] * 100 for b in budget]
    ax.plot(xs, [b["f1"] for b in budget], "o-", color="#2563eb", label="rules + ML + LLM band")
    ax.axhline(llm_all["f1"], color="#dc2626", ls="--", label=f"LLM for everything (F1 {llm_all['f1']:.3f})")
    ax.axhline(ml_only["f1"], color="#6b7280", ls=":", label=f"ML only (F1 {ml_only['f1']:.3f})")
    ax.scatter([chosen["routed_share"] * 100], [chosen["f1"]], s=140, facecolors="none",
               edgecolors="#16a34a", linewidths=2, zorder=5, label="chosen band (tuned on calib)")
    ax.set_xlabel("% of comments sent to the LLM")
    ax.set_ylabel("F1 (toxic)")
    ax.set_title(f"LLM budget vs F1 (n={len(y)} sample)")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")

    ax = axes[1]
    ax.scatter([g["cost_per_10k"] for g in grid], [g["f1"] for g in grid], s=10, color="#93c5fd",
               label=f"all {len(grid)} (low, high) bands")
    ax.plot([q["cost_per_10k"] for q in pareto], [q["f1"] for q in pareto], "-o", ms=3,
            color="#1d4ed8", label="Pareto front")
    ax.scatter([chosen["cost_per_10k"]], [chosen["f1"]], s=140, facecolors="none",
               edgecolors="#16a34a", linewidths=2, zorder=5, label="chosen band")
    ax.scatter([llm_all["cost_per_10k"]], [llm_all["f1"]], marker="x", s=80, color="#dc2626",
               label="LLM for everything")
    ax.set_xlabel("estimated LLM cost, USD per 10k comments")
    ax.set_ylabel("F1 (toxic)")
    ax.set_title("Cost vs F1 for every confidence band")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=8, loc="lower right")
    fig.tight_layout()
    path = cfg.path("results/charts/band_sweep.png")
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    print(json.dumps({"chosen": chosen, "ml_only": ml_only, "llm_all": llm_all,
                      "best_on_grid": max(grid, key=lambda r: r["f1"])}, indent=2))


if __name__ == "__main__":
    main()
