# Resume bullets: Hybrid AI Comment Moderation

Every number below is measured and traceable to `results/metrics.json`, `results/fairness.json`,
`results/band_sweep.json` or `results/calibration.json`. Data: Civil Comments (WILDS), human labels.

- Built a 4-stage comment-moderation pipeline (hashed-lexicon rules -> isotonic-calibrated
  TF-IDF/logistic regression -> LLM review of only low-confidence comments -> SQLite human-review
  queue) with a FastAPI service, Docker image and 48 offline pytest tests.
- Routed only 9.2% of comments to the LLM (`gpt-oss-safeguard-20b` on Groq) and lifted F1 from
  0.617 (ML only) to 0.633 on 3,000 held-out test comments, cutting false positives 188 -> 153,
  at a projected $0.036 per 10k comments.
- Ran a head-to-head on 500 comments: sending everything to the LLM scored F1 0.429 vs 0.624 for
  the classifier and 0.622 for the hybrid, at ~10x the hybrid's projected API cost ($0.374 vs
  $0.035 per 10k) and ~280x the classifier's median latency (936 ms vs 3.3 ms).
- Mapped the LLM budget-vs-quality curve with cached verdicts: F1 peaked at 0.645 with 6.6% of
  comments routed and fell to 0.440 at 98.6% routed.
- Halved expected calibration error with isotonic calibration (ECE 0.028 -> 0.014, ROC-AUC 0.918
  -> 0.919), so the LLM routing band set on calibration data routed the same 9.2% on test data.
- Audited fairness on the 5,000-comment test split: the baseline hid 13.3% of non-toxic comments
  mentioning an identity group vs 3.5% of others (3.8x); training-data reweighing cut this to 9.7%
  (2.8x) for a 1.1-point F1 cost (0.603 -> 0.592).
