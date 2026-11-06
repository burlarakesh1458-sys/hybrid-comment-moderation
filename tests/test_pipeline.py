from hybridmod.llm import LLMReviewer
from hybridmod.pipeline import HybridModerator
from hybridmod.review_queue import ReviewQueue

from .conftest import FakeTransport, verdicts_from_keywords


def make(cfg, rules, clf, transport=None, queue=None, **kw):
    rev = LLMReviewer(cfg.llm, transport=transport or FakeTransport(default=verdicts_from_keywords),
                      cache_path=None)
    return HybridModerator(cfg, rules, clf, rev, queue=queue, **kw), rev


def test_rules_short_circuit_before_ml(cfg, rules, stub_clf):
    mod, rev = make(cfg, rules, stub_clf)
    d = mod.moderate("badword [p=0.01]")
    assert d.action == "hide" and d.stage == "rules" and d.p_toxic is None
    assert d.human_review and d.review_reason == "rule_block"
    assert mod.moderate("check http://a http://b http://c").action == "spam_hold"
    assert not rev.stats.calls


def test_confident_scores_skip_llm(cfg, rules, stub_clf):
    mod, rev = make(cfg, rules, stub_clf)
    low, high = mod.moderate_many(["fine [p=0.05]", "awful [p=0.95]"])
    assert (low.action, low.stage) == ("allow", "ml")
    assert (high.action, high.stage) == ("hide", "ml")
    assert high.human_review and high.review_reason == "high_confidence_ml"
    assert rev.stats.calls == 0


def test_band_goes_to_llm_and_llm_decides(cfg, rules, stub_clf):
    mod, rev = make(cfg, rules, stub_clf)
    ok, bad = mod.moderate_many(["just a question? [p=0.6]", "that is a dumb take [p=0.35]"])
    assert ok.stage == "llm" and ok.action == "allow" and ok.category == "question"
    assert bad.stage == "llm" and bad.action == "hide" and bad.category == "insult"
    assert not bad.human_review  # insult hides the comment but is not penalty-eligible
    assert rev.stats.calls == 1  # both band comments in one batch


def test_penalty_categories_go_to_human_queue(cfg, rules, stub_clf):
    q = ReviewQueue(":memory:")
    mod, _ = make(cfg, rules, stub_clf, queue=q)
    d = mod.moderate("I will kill you [p=0.5]", comment_id="c1")
    assert d.action == "hide" and d.human_review and d.review_reason == "penalty_category:threat"
    items = q.list()
    assert len(items) == 1 and items[0]["comment_id"] == "c1" and items[0]["category"] == "threat"


def test_llm_failure_falls_back_to_ml_and_human(cfg, rules, stub_clf):
    mod, _ = make(cfg, rules, stub_clf, transport=FakeTransport(default="garbage"))
    d = mod.moderate("unclear [p=0.55]")
    assert d.stage == "llm_fallback" and d.action == "hide"  # 0.55 >= threshold 0.5
    assert d.human_review and d.review_reason == "llm_failure"


def test_ml_only_mode_ignores_band(cfg, rules, stub_clf):
    mod, rev = make(cfg, rules, stub_clf, use_rules=False, use_llm=False)
    d1, d2 = mod.moderate_many(["badword [p=0.2]", "x [p=0.55]"])
    assert d1.stage == "ml" and d1.action == "allow"  # rules off
    assert d2.stage == "ml" and d2.action == "hide"
    assert rev.stats.calls == 0


def test_band_edges_are_half_open(cfg, rules, stub_clf):
    mod, _ = make(cfg, rules, stub_clf)
    lo, hi = mod.moderate_many(["a [p=0.3]", "b [p=0.7]"])
    assert lo.stage == "llm" and hi.stage == "ml"


def test_decision_serialises_with_flag(cfg, rules, stub_clf):
    mod, _ = make(cfg, rules, stub_clf)
    d = mod.moderate("x [p=0.9]").to_dict()
    assert d["flagged"] is True and d["latency_ms"] >= 0


def test_degraded_mode_sends_band_to_human(cfg, rules, stub_clf):
    mod = HybridModerator(cfg, rules, stub_clf, None, band_to_human_without_llm=True)
    band, sure = mod.moderate_many(["meh [p=0.4]", "fine [p=0.1]"])
    assert band.stage == "ml" and band.action == "allow"
    assert band.human_review and band.review_reason == "llm_unavailable"
    assert not sure.human_review
