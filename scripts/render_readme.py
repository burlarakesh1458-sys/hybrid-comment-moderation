"""Fill the generated blocks of README.md from results/*.json, so README numbers are never typed
by hand. Blocks look like <!-- GEN:name --> ... <!-- /GEN:name -->."""

from __future__ import annotations

import json
import re

from hybridmod.config import ROOT


def load(name):
    p = ROOT / "results" / name
    return json.loads(p.read_text()) if p.exists() else None


def fmt_ms(v):
    return "-" if v is None else (f"{v:,.0f}" if v >= 100 else f"{v:.1f}")


def scenario_table(rows) -> str:
    head = ("| scenario | F1 [95% CI] | precision | recall | false pos. | false neg. | "
            "latency p50 / p95 (ms) | LLM comments | LLM requests | est. $ per 10k | "
            "human reviews per 10k |\n|---|---|---|---|---|---|---|---|---|---|---|")
    names = {"ml_only": "(a) ML only", "rules_ml": "rules + ML (ablation)",
             "llm_all": "(b) LLM for everything", "hybrid": "(c) hybrid"}
    lines = [head]
    for r in rows:
        ci = r.get("f1_95ci")
        lines.append(
            f"| {names.get(r['scenario'], r['scenario'])} | **{r['f1']:.3f}**"
            + (f" [{ci[0]:.2f}, {ci[1]:.2f}]" if ci else "")
            + f" | {r['precision']:.3f} | {r['recall']:.3f} | {r['fp']} | {r['fn']} | "
            f"{fmt_ms(r['latency_ms']['p50'])} / {fmt_ms(r['latency_ms']['p95'])} | "
            f"{r['llm_comments']} | {r['llm_requests_at_batch']} | "
            f"${r['est_cost_usd_per_10k']:.3f} | {r['human_review_per_10k']:.0f} |")
    return "\n".join(lines)


def blocks() -> dict[str, str]:
    out = {}
    m = load("metrics.json")
    if m:
        c = m["config"]
        out["sample_table"] = f"Fixed LLM sample, **n = {c['sample_n']}** comments.\n\n" + \
            scenario_table(m["sample"])
        if m.get("test"):
            out["test_table"] = f"Test comments, **n = {c['test_n']}** (LLM sample included).\n\n" + \
                scenario_table(m["test"])
    f = load("fairness.json")
    if f:
        rows = ["| system | eval set | F1 | FPR, identity mentioned | FPR, no identity | gap | ratio |",
                "|---|---|---|---|---|---|---|"]

        def add(name, d, where):
            i = d["identity"]
            rows.append(f"| {name} | {where} | {d['overall']['f1']:.3f} | "
                        f"{i['mentions_identity']['fpr']:.3f} | {i['no_identity']['fpr']:.3f} | "
                        f"{i['fpr_gap']:+.3f} | {i['fpr_ratio']:.2f}x |")
        for k, d in f["systems"].items():
            add(k, d, f"test n={f['test_n']}")
        for k in ("ml_baseline", "ml_reweighted", "hybrid"):
            if f.get("hybrid_subset", {}).get(k):
                add(k, f["hybrid_subset"][k], f"test n={f['hybrid_subset']['n']}")
        for k in ("ml", "llm", "hybrid"):
            if f.get("sample", {}).get(k):
                add({"ml": "ml_baseline", "llm": "llm_all"}.get(k, k), f["sample"][k],
                    f"sample n={f['sample']['n']}")
        out["fairness_table"] = "\n".join(rows)
    s = load("band_sweep.json")
    if s:
        rows = ["| calib share routed | band [low, high) | share routed on sample | F1 | FP | FN | "
                "est. $ per 10k |", "|---|---|---|---|---|---|---|"]
        for b in s["budget_curve"]:
            rows.append(f"| {b['calib_share']:.0%} | [{b['band_low']:.3f}, {b['band_high']:.3f}) | "
                        f"{b['routed_share']:.1%} | {b['f1']:.3f} | {b['fp']} | {b['fn']} | "
                        f"${b['cost_per_10k']:.3f} |")
        out["sweep_table"] = "\n".join(rows)
    return out


def main() -> None:
    readme = ROOT / "README.md"
    text = readme.read_text()
    for name, body in blocks().items():
        pat = re.compile(rf"(<!-- GEN:{name} -->)(.*?)(<!-- /GEN:{name} -->)", re.S)
        text = pat.sub(lambda m, b=body: f"{m.group(1)}\n{b}\n{m.group(3)}", text)
    readme.write_text(text)
    print("README blocks:", ", ".join(blocks()))


if __name__ == "__main__":
    main()
