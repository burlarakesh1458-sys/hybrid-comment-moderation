"""Compare the moderation scenarios on the same held-out comments.

  (a) ml_only     calibrated TF-IDF + LR at the calibration-tuned threshold
  (a2) rules_ml   stage-1 rules + ML (ablation: the hybrid without the LLM)
  (b) llm_all     every comment sent to the LLM (only on the fixed LLM sample - rate limits)
  (c) hybrid      rules -> ML -> LLM for the low-confidence band -> human queue

Two evaluation sets:
  sample  the fixed N-comment LLM sample (all four scenarios, head-to-head)
  test    the 5,000-comment test split, or its first --test-n comments in the fixed permutation
          (a superset of the sample) when the LLM budget cannot cover the full split

LLM verdicts come from the on-disk cache; with --allow-llm, missing band verdicts are fetched
(resumable). Outputs results/metrics.json, results/results_table.md, results/predictions_*.csv.
"""

from __future__ import annotations

import argparse
import json
import math
import sys

import numpy as np
import pandas as pd

from hybridmod.classifier import ToxicityClassifier
from hybridmod.config import load_config
from hybridmod.data import llm_sample, load_split
from hybridmod.llm import LLMReviewer, RateLimited, cache_key
from hybridmod.metrics import binary_metrics, cost_per_10k, percentiles
from hybridmod.pipeline import HybridModerator
from hybridmod.rules import RuleEngine


class OfflineTransport:
    def __call__(self, messages):
        raise RateLimited(1e9, "offline: verdict not in cache (run with --allow-llm)")


def bootstrap_f1(y, pred, n=1000, seed=0) -> list[float]:
    rng = np.random.default_rng(seed)
    y, pred = np.asarray(y), np.asarray(pred)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        vals.append(binary_metrics(y[i], pred[i])["f1"])
    return [round(float(np.percentile(vals, 2.5)), 4), round(float(np.percentile(vals, 97.5)), 4)]


def summarise(name, y, flagged, latencies, n_llm_comments, prompt_tok, compl_tok, human, cfg,
              extra=None, ci=False) -> dict:
    n = len(y)
    bs = cfg.llm.batch_size
    out = {
        "scenario": name, **binary_metrics(y, flagged),
        "latency_ms": percentiles(latencies),
        "llm_comments": int(n_llm_comments),
        "llm_requests_at_batch": int(math.ceil(n_llm_comments / bs)) if n_llm_comments else 0,
        "llm_comments_per_10k": round(n_llm_comments / n * 10_000, 1),
        "llm_requests_per_10k": round(n_llm_comments / n * 10_000 / bs, 1),
        "prompt_tokens": round(float(prompt_tok)), "completion_tokens": round(float(compl_tok)),
        "est_cost_usd_per_10k": cost_per_10k(prompt_tok, compl_tok, n, cfg.llm.price_input_per_m,
                                             cfg.llm.price_output_per_m),
        "human_review": int(human), "human_review_per_10k": round(human / n * 10_000, 1),
    }
    if ci:
        out["f1_95ci"] = bootstrap_f1(y, flagged)
    out.update(extra or {})
    return out


def run_moderator(mod: HybridModerator, df: pd.DataFrame, cfg, name: str, ci=False):
    ds = mod.moderate_many(list(df.comment_text), per_item_timing=True)
    y = df.toxicity.to_numpy()
    flagged = np.array([d.flagged for d in ds])
    routed = [d for d in ds if d.llm_called]
    pt = ct = 0.0
    if routed and mod.reviewer is not None:
        for d, t in zip(ds, df.comment_text, strict=True):
            if d.llm_called and d.stage == "llm":
                rec = mod.reviewer.cache.get(cache_key(cfg.llm.model, t))
                if rec:
                    pt += rec["prompt_tokens"]
                    ct += rec["completion_tokens"]
    stages = pd.Series([d.stage for d in ds]).value_counts().to_dict()
    actions = pd.Series([d.action for d in ds]).value_counts().to_dict()
    res = summarise(name, y, flagged, [d.latency_ms for d in ds], len(routed), pt, ct,
                    sum(d.human_review for d in ds), cfg,
                    {"stages": stages, "actions": actions,
                     "llm_failures": sum(d.stage == "llm_fallback" for d in ds)}, ci=ci)
    return res, ds


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--allow-llm", action="store_true", help="fetch missing band verdicts")
    ap.add_argument("--model", default=None, help="override classifier path")
    ap.add_argument("--tag", default="")
    ap.add_argument("--test-n", type=int, default=None,
                    help="evaluate the 'test' set on the first N comments of the fixed test "
                         "permutation (nests the LLM sample); default = all 5,000")
    args = ap.parse_args()
    cfg = load_config()
    rules = RuleEngine.from_file(cfg.rules, cfg.path(cfg.rules.lexicon_path))
    clf = ToxicityClassifier.load(args.model or cfg.path(cfg.classifier.model_path))
    transport = None if args.allow_llm else OfflineTransport()
    reviewer = LLMReviewer(cfg.llm, transport=transport, cache_path=cfg.path(cfg.llm.cache_path),
                           max_rate_wait_s=3600 if args.allow_llm else 0)

    test = llm_sample(args.test_n) if args.test_n else load_split("test")
    sample = llm_sample(cfg.experiment.llm_sample_size)
    missing = [t for t in sample.comment_text if not reviewer.cache.get(cache_key(cfg.llm.model, t))]
    if missing:
        sys.exit(f"{len(missing)} LLM-sample comments not cached yet; run scripts/run_llm_sample.py")

    ml = HybridModerator(cfg, rules, clf, None, use_rules=False, use_llm=False)
    rml = HybridModerator(cfg, rules, clf, None, use_rules=True, use_llm=False)
    hyb = HybridModerator(cfg, rules, clf, reviewer, use_rules=True, use_llm=True)

    results: dict = {"config": {
        "classifier": cfg.classifier.model_dump(), "llm_model": cfg.llm.model,
        "llm_batch_size": cfg.llm.batch_size,
        "price_per_m": [cfg.llm.price_input_per_m, cfg.llm.price_output_per_m],
        "price_source": f"https://console.groq.com/docs/model/{cfg.llm.model}",
        "sample_n": len(sample), "test_n": len(test)}}

    # ---- head-to-head on the LLM sample ---------------------------------------------------
    y_s = sample.toxicity.to_numpy()
    r_ml_s, d_ml_s = run_moderator(ml, sample, cfg, "ml_only", ci=True)
    r_rml_s, d_rml_s = run_moderator(rml, sample, cfg, "rules_ml", ci=True)
    llm_res = reviewer.review(list(sample.comment_text))
    llm_flag = np.array([bool(r.verdict and r.verdict.violates) for r in llm_res])
    llm_hr = sum(bool(r.verdict and r.verdict.violates
                      and r.verdict.category.value in cfg.human_review.penalty_categories)
                 for r in llm_res)
    r_llm_s = summarise(
        "llm_all", y_s, llm_flag, [r.latency_ms for r in llm_res], len(llm_res),
        sum(r.prompt_tokens for r in llm_res), sum(r.completion_tokens for r in llm_res),
        llm_hr, cfg, {"llm_failures": sum(r.verdict is None for r in llm_res),
                      "categories": pd.Series([r.verdict.category.value for r in llm_res
                                               if r.verdict]).value_counts().to_dict()}, ci=True)
    r_hyb_s, d_hyb_s = run_moderator(hyb, sample, cfg, "hybrid", ci=True)
    results["sample"] = [r_ml_s, r_rml_s, r_llm_s, r_hyb_s]

    pd.DataFrame({
        "uid": sample.uid, "toxicity": y_s, "identity_any": sample.identity_any,
        "p_ml": [d.p_toxic for d in d_ml_s], "ml_flag": [d.flagged for d in d_ml_s],
        "rules_ml_flag": [d.flagged for d in d_rml_s], "llm_flag": llm_flag,
        "llm_category": [r.verdict.category.value if r.verdict else None for r in llm_res],
        "hybrid_flag": [d.flagged for d in d_hyb_s], "hybrid_stage": [d.stage for d in d_hyb_s],
        "hybrid_action": [d.action for d in d_hyb_s],
        "hybrid_human_review": [d.human_review for d in d_hyb_s],
    }).to_csv(cfg.path(f"results/predictions_sample{args.tag}.csv"), index=False)

    # ---- full test split --------------------------------------------------------------------
    r_ml_t, d_ml_t = run_moderator(ml, test, cfg, "ml_only", ci=True)
    r_rml_t, d_rml_t = run_moderator(rml, test, cfg, "rules_ml", ci=True)
    try:
        r_hyb_t, d_hyb_t = run_moderator(hyb, test, cfg, "hybrid", ci=True)
        results["test"] = [r_ml_t, r_rml_t, r_hyb_t]
        pd.DataFrame({
            "uid": test.uid, "toxicity": test.toxicity, "identity_any": test.identity_any,
            "p_ml": [d.p_toxic for d in d_ml_t], "ml_flag": [d.flagged for d in d_ml_t],
            "rules_ml_flag": [d.flagged for d in d_rml_t],
            "hybrid_flag": [d.flagged for d in d_hyb_t], "hybrid_stage": [d.stage for d in d_hyb_t],
            "hybrid_action": [d.action for d in d_hyb_t],
            "hybrid_category": [d.category for d in d_hyb_t],
            "hybrid_human_review": [d.human_review for d in d_hyb_t],
            "hybrid_review_reason": [d.review_reason for d in d_hyb_t],
            "rule": [d.rule for d in d_hyb_t],
        }).to_csv(cfg.path(f"results/predictions_test{args.tag}.csv"), index=False)
    except Exception as e:  # pragma: no cover - only when band verdicts are missing offline
        results["test"] = [r_ml_t, r_rml_t]
        results["test_hybrid_error"] = str(e)
    results["llm_stats_this_run"] = reviewer.stats.__dict__ | {
        "call_latencies_ms": percentiles(reviewer.stats.call_latencies_ms)}

    cfg.path(f"results/metrics{args.tag}.json").write_text(json.dumps(results, indent=2) + "\n")
    cfg.path(f"results/results_table{args.tag}.md").write_text(render_table(results))
    print(render_table(results))


def render_table(res: dict) -> str:
    cols = ["scenario", "n", "f1", "precision", "recall", "fp", "fn", "p50 ms", "p95 ms",
            "LLM comments", "LLM req", "$ / 10k", "human / 10k"]
    lines = []
    for key, title in (("sample", f"Head-to-head on the fixed LLM sample (n={res['config']['sample_n']})"),
                       ("test", (f"Full test split (n={res['config']['test_n']})"
                                 if res['config']['test_n'] >= 5000 else
                                 f"Test subset, first {res['config']['test_n']} of the 5,000 in "
                                 "the fixed permutation"))):
        lines += [f"### {title}", "", "| " + " | ".join(cols) + " |",
                  "|" + "---|" * len(cols)]
        for r in res.get(key, []):
            ci = r.get("f1_95ci")
            f1 = f"{r['f1']:.3f}" + (f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci else "")
            lines.append("| " + " | ".join(str(x) for x in [
                r["scenario"], r["n"], f1, f"{r['precision']:.3f}", f"{r['recall']:.3f}", r["fp"],
                r["fn"], r["latency_ms"]["p50"], r["latency_ms"]["p95"], r["llm_comments"],
                r["llm_requests_at_batch"], f"{r['est_cost_usd_per_10k']:.4f}",
                r["human_review_per_10k"]]) + " |")
        lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
