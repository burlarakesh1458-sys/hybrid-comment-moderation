"""Scenario (b) data collection: send every comment of the fixed LLM sample to the LLM.

Resumable: verdicts are appended to the JSONL cache as they arrive; re-running only pays for
comments not cached yet. Writes results/llm_sample_run.json with call / token / failure counts
for THIS invocation (the cache keeps per-comment token shares for cost projection).
"""

from __future__ import annotations

import argparse
import json
import time

from hybridmod.config import load_config
from hybridmod.data import llm_sample
from hybridmod.llm import LLMReviewer


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=None)
    ap.add_argument("--chunk", type=int, default=64, help="progress print / flush granularity")
    args = ap.parse_args()
    cfg = load_config()
    n = args.n or cfg.experiment.llm_sample_size
    df = llm_sample(n)
    rev = LLMReviewer(cfg.llm, cache_path=cfg.path(cfg.llm.cache_path), max_rate_wait_s=3600)
    texts = list(df.comment_text)
    t0 = time.time()
    failed = 0
    for s in range(0, len(texts), args.chunk):
        res = rev.review(texts[s:s + args.chunk])
        failed += sum(r.verdict is None for r in res)
        st = rev.stats
        print(f"{min(s + args.chunk, len(texts))}/{len(texts)} calls={st.calls} "
              f"hits={st.cache_hits} retries={st.retries} 429s={st.rate_limited} "
              f"fail={failed} tok_in={st.prompt_tokens} tok_out={st.completion_tokens} "
              f"{time.time() - t0:.0f}s", flush=True)
    out = cfg.path("results/llm_sample_run.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    st = rev.stats
    out.write_text(json.dumps({
        "model": cfg.llm.model, "n": n, "batch_size": cfg.llm.batch_size,
        "calls_this_run": st.calls, "cache_hits_this_run": st.cache_hits,
        "retries": st.retries, "rate_limited": st.rate_limited, "failures": failed,
        "prompt_tokens_this_run": st.prompt_tokens,
        "completion_tokens_this_run": st.completion_tokens,
        "wall_s": round(time.time() - t0, 1),
    }, indent=2) + "\n")


if __name__ == "__main__":
    main()
