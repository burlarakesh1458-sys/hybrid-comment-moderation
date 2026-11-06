"""Side comparison of every LLM that has cached verdicts on a common prefix of the LLM sample.

The run that produced the cache started on openai/gpt-oss-120b and switched to
openai/gpt-oss-safeguard-20b when the free-tier daily token cap for gpt-oss-120b was exhausted
(see README). This script compares the models only on the comments BOTH have judged.
"""

from __future__ import annotations

import json

import numpy as np

from hybridmod.config import load_config
from hybridmod.data import llm_sample
from hybridmod.llm import JsonlCache, cache_key
from hybridmod.metrics import binary_metrics

PRICES = {  # USD per 1M tokens, Groq model pages read 2026-11-06
    "openai/gpt-oss-120b": (0.15, 0.60),
    "openai/gpt-oss-safeguard-20b": (0.075, 0.30),
    "openai/gpt-oss-20b": (0.075, 0.30),
}


def main() -> None:
    cfg = load_config()
    cache = JsonlCache(cfg.path(cfg.llm.cache_path))
    models = sorted({r["model"] for r in cache.data.values()})
    df = llm_sample(cfg.experiment.llm_sample_size)
    texts = list(df.comment_text)
    have = {m: [cache.get(cache_key(m, t)) for t in texts] for m in models}
    # longest common prefix where every model has a verdict
    n = 0
    while n < len(texts) and all(have[m][n] for m in models):
        n += 1
    y = df.toxicity.to_numpy()[:n]
    out = {"models": models, "common_prefix_n": n, "results": {}}
    flags = {}
    for m in models:
        recs = have[m][:n]
        f = np.array([r["verdict"]["violates"] for r in recs])
        flags[m] = f
        pin, pout = PRICES.get(m, (None, None))
        pt = sum(r["prompt_tokens"] for r in recs)
        ct = sum(r["completion_tokens"] for r in recs)
        out["results"][m] = {
            **binary_metrics(y, f),
            "tokens_per_comment": round((pt + ct) / max(n, 1), 1),
            "est_cost_usd_per_10k": round((pt * pin + ct * pout) / 1e6 / max(n, 1) * 1e4, 4)
            if pin else None,
            "median_call_latency_ms": float(np.median([r["latency_ms"] for r in recs])) if n else None,
            "batch_size": recs[0]["batch_size"] if n else None,
        }
    if len(models) == 2 and n:
        a, b = models
        out["agreement"] = round(float((flags[a] == flags[b]).mean()), 4)
    cfg.path("results/llm_model_comparison.json").write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
