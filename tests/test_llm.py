import json

import httpx
import pytest
from pydantic import ValidationError

from hybridmod.llm import (
    CommentVerdict,
    LLMReviewer,
    RateLimited,
    RequestTooLarge,
    build_user_prompt,
    parse_batch,
)

from .conftest import FakeTransport, verdicts_from_keywords


def test_verdict_schema_enforces_consistency():
    CommentVerdict(id=1, category="insult", violates=True, confidence=0.8)
    with pytest.raises(ValidationError):
        CommentVerdict(id=1, category="insult", violates=False, confidence=0.8)
    with pytest.raises(ValidationError):
        CommentVerdict(id=1, category="appreciation", violates=True, confidence=0.8)


def test_verdict_schema_rejects_bad_category_and_confidence():
    with pytest.raises(ValidationError):
        CommentVerdict(id=1, category="spam", violates=False, confidence=0.5)
    with pytest.raises(ValidationError):
        CommentVerdict(id=1, category="question", violates=False, confidence=1.5)


def test_parse_batch_reports_missing_and_invalid():
    raw = json.dumps({"results": [
        {"id": 0, "category": "question", "violates": False, "confidence": 0.9},
        {"id": 1, "category": "threat", "violates": False, "confidence": 0.9},
    ]})
    ok, errs = parse_batch(raw, [0, 1, 2])
    assert set(ok) == {0}
    assert any("missing" in e for e in errs) and any("item 1" in e for e in errs)


def test_parse_batch_non_json():
    ok, errs = parse_batch("Sure! here you go", [0])
    assert ok == {} and "not valid JSON" in errs[0]


def test_parse_batch_accepts_bare_list():
    raw = json.dumps([{"id": 0, "category": "appreciation", "violates": False, "confidence": 1}])
    ok, errs = parse_batch(raw, [0])
    assert 0 in ok and not errs


def test_prompt_contains_ids_and_truncates():
    p = build_user_prompt([(0, "a" * 5000), (1, "hello")], max_chars=100)
    lines = p.splitlines()
    assert len(lines) == 3 and json.loads(lines[1])["comment"] == "a" * 100


def test_review_batches_and_caches(cfg, tmp_path):
    tr = FakeTransport(default=verdicts_from_keywords)
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=tmp_path / "c.jsonl")
    texts = ["is this ok?", "kill them", "dumb take", "nice", "I hate group x"]
    res = rev.review(texts)
    assert [r.verdict.category.value for r in res] == [
        "question", "threat", "insult", "neutral_ok", "identity_attack"]
    assert len(tr.calls) == 2  # batch_size=4 -> 2 requests
    # second reviewer instance re-reads the cache from disk: zero new calls
    tr2 = FakeTransport(default=verdicts_from_keywords)
    rev2 = LLMReviewer(cfg.llm, transport=tr2, cache_path=tmp_path / "c.jsonl")
    res2 = rev2.review(texts)
    assert not tr2.calls and rev2.stats.cache_hits == 5 and all(r.cached for r in res2)


def test_retry_only_reasks_invalid_items(cfg):
    def first(items):  # id 1 inconsistent, id 2 missing
        return json.dumps({"results": [
            {"id": 0, "category": "question", "violates": False, "confidence": 0.9},
            {"id": 1, "category": "threat", "violates": False, "confidence": 0.9}]})

    tr = FakeTransport(script=[first, verdicts_from_keywords])
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=None)
    res = rev.review(["why?", "kill", "fine"])
    assert all(r.verdict is not None for r in res)
    assert len(FakeTransport.items(tr.calls[1])) == 2  # only ids 1 and 2 re-asked
    assert "invalid" in tr.calls[1][-1]["content"]
    assert rev.stats.retries == 1


def test_gives_up_after_max_retries(cfg):
    tr = FakeTransport(default="not json")
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=None)
    res = rev.review(["a", "b"])
    assert all(r.verdict is None for r in res)
    assert len(tr.calls) == cfg.llm.max_retries + 1 and rev.stats.failures == 2


def test_rate_limit_waits_without_consuming_retries(cfg):
    slept = []
    tr = FakeTransport(script=[RateLimited(3), RateLimited(2), verdicts_from_keywords])
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=None, sleep=slept.append)
    res = rev.review(["hello"])
    assert res[0].verdict is not None and rev.stats.rate_limited == 2
    assert slept == [3.5, 2.5] and rev.stats.retries == 0


def test_transport_error_is_retried(cfg):
    tr = FakeTransport(script=[httpx.ConnectError("boom"), verdicts_from_keywords])
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=None)
    assert rev.review(["hello"])[0].verdict is not None


def test_too_large_batch_is_split(cfg):
    def respond(items):
        if len(items) > 2:
            raise RequestTooLarge("413")
        return verdicts_from_keywords(items)

    class SplitTransport(FakeTransport):
        def __call__(self, messages):
            self.calls.append(messages)
            return respond(self.items(messages)), {"prompt_tokens": 10, "completion_tokens": 5}

    tr = SplitTransport()
    rev = LLMReviewer(cfg.llm, transport=tr, cache_path=None)
    res = rev.review(["a", "b", "kill", "d"])  # batch_size 4 -> 413 -> 2 + 2
    assert all(r.verdict is not None for r in res) and rev.stats.splits == 1
    assert res[2].verdict.category.value == "threat"
