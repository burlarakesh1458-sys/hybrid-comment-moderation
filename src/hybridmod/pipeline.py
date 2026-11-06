"""The hybrid moderation pipeline: rules -> calibrated classifier -> LLM band -> human queue.

Comment-level actions:
  allow      publish
  hide       remove from public view automatically (reversible, no account consequence)
  spam_hold  hold for spam review (not counted as a toxicity flag in evaluation)

`human_review=True` means the decision could justify an account penalty (strike / suspension) or
the automated stages could not decide; the comment is queued for a person, who makes the final
call. Automated stages never issue account penalties.
"""

from __future__ import annotations

import time
from dataclasses import asdict, dataclass

from .config import Config
from .llm import LLMReviewer
from .rules import RuleEngine


@dataclass
class Decision:
    action: str
    stage: str  # rules | ml | llm | llm_fallback
    p_toxic: float | None = None
    rule: str | None = None
    category: str | None = None
    llm_confidence: float | None = None
    reason: str | None = None
    human_review: bool = False
    review_reason: str | None = None
    latency_ms: float = 0.0
    llm_called: bool = False

    @property
    def flagged(self) -> bool:
        return self.action == "hide"

    def to_dict(self) -> dict:
        return asdict(self) | {"flagged": self.flagged}


class HybridModerator:
    def __init__(self, cfg: Config, rules: RuleEngine | None, classifier, reviewer: LLMReviewer | None,
                 queue=None, use_rules: bool = True, use_llm: bool = True,
                 band_to_human_without_llm: bool = False):
        self.cfg = cfg
        self.rules = rules
        self.clf = classifier
        self.reviewer = reviewer
        self.queue = queue
        self.use_rules = use_rules and rules is not None
        self.use_llm = use_llm and reviewer is not None and cfg.llm.enabled
        # degraded mode (LLM configured but unavailable): band comments keep the ML decision and
        # are queued for a person instead of being silently auto-decided
        self.band_to_human = band_to_human_without_llm and not self.use_llm

    # ---- helpers -------------------------------------------------------------------------
    def _ml_decision(self, p: float) -> Decision:
        c = self.cfg.classifier
        flagged = p >= c.threshold
        d = Decision("hide" if flagged else "allow", "ml", p_toxic=p)
        if flagged and p >= self.cfg.human_review.penalty_min_p_toxic:
            d.human_review, d.review_reason = True, "high_confidence_ml"
        return d

    def _in_band(self, p: float) -> bool:
        c = self.cfg.classifier
        return c.band_low <= p < c.band_high

    def _apply_verdict(self, d: Decision, res) -> Decision:
        hr = self.cfg.human_review
        d.llm_called = True
        if res is None or res.verdict is None:
            # LLM unavailable/invalid after retries: fall back to the ML threshold, ask a human
            fb = self._ml_decision(d.p_toxic or 0.0)
            d.action, d.stage = fb.action, "llm_fallback"
            if hr.queue_llm_failures:
                d.human_review, d.review_reason = True, "llm_failure"
            return d
        v = res.verdict
        d.stage, d.category, d.llm_confidence, d.reason = "llm", v.category.value, v.confidence, v.reason
        d.action = "hide" if v.violates else "allow"
        if v.violates and v.category.value in hr.penalty_categories:
            d.human_review, d.review_reason = True, f"penalty_category:{v.category.value}"
        d.latency_ms += res.latency_ms
        return d

    def _enqueue(self, text: str, d: Decision, comment_id: str | None) -> None:
        if self.queue is not None and d.human_review:
            self.queue.enqueue(comment_id, text, d)

    # ---- public API ----------------------------------------------------------------------
    def moderate_many(self, texts: list[str], comment_ids: list[str] | None = None,
                      per_item_timing: bool = False) -> list[Decision]:
        n = len(texts)
        decisions: list[Decision | None] = [None] * n
        ml_idx: list[int] = []
        t_rules = [0.0] * n
        for i, t in enumerate(texts):
            t0 = time.perf_counter()
            r = self.rules.evaluate(t) if self.use_rules else None
            t_rules[i] = (time.perf_counter() - t0) * 1000
            if r is None or r.action == "pass":
                ml_idx.append(i)
            elif r.action == "allow":
                decisions[i] = Decision("allow", "rules", rule=r.rule)
            elif r.action == "spam":
                decisions[i] = Decision("spam_hold", "rules", rule=r.rule)
            else:  # block
                d = Decision("hide", "rules", rule=r.rule)
                if self.cfg.human_review.queue_rule_blocks:
                    d.human_review, d.review_reason = True, "rule_block"
                decisions[i] = d

        if ml_idx:
            if per_item_timing:
                probs, ml_ms = [], []
                for i in ml_idx:
                    t0 = time.perf_counter()
                    probs.append(float(self.clf.predict_proba([texts[i]])[0]))
                    ml_ms.append((time.perf_counter() - t0) * 1000)
            else:
                t0 = time.perf_counter()
                probs = [float(p) for p in self.clf.predict_proba([texts[i] for i in ml_idx])]
                ml_ms = [(time.perf_counter() - t0) * 1000 / len(ml_idx)] * len(ml_idx)
            band: list[int] = []
            for i, p, ms in zip(ml_idx, probs, ml_ms, strict=True):
                if self.use_llm and self._in_band(p):
                    decisions[i] = Decision("allow", "ml", p_toxic=p, latency_ms=ms)
                    band.append(i)
                else:
                    d = self._ml_decision(p)
                    d.latency_ms = ms
                    if self.band_to_human and self._in_band(p) and not d.human_review:
                        d.human_review, d.review_reason = True, "llm_unavailable"
                    decisions[i] = d
            if band:
                results = self.reviewer.review([texts[i] for i in band])
                for i, res in zip(band, results, strict=True):
                    decisions[i] = self._apply_verdict(decisions[i], res)

        out: list[Decision] = []
        for i, d in enumerate(decisions):
            assert d is not None
            d.latency_ms = round(d.latency_ms + t_rules[i], 3)
            self._enqueue(texts[i], d, comment_ids[i] if comment_ids else None)
            out.append(d)
        return out

    def moderate(self, text: str, comment_id: str | None = None) -> Decision:
        return self.moderate_many([text], [comment_id] if comment_id else None)[0]
