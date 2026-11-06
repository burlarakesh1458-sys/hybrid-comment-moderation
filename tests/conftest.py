from __future__ import annotations

import json
import re

import numpy as np
import pytest

from hybridmod.config import Config
from hybridmod.rules import RuleEngine, hash_token


class StubClassifier:
    """Deterministic P(toxic) from markers in the text: [p=0.42] sets it explicitly."""

    def predict_proba(self, texts):
        out = []
        for t in texts:
            m = re.search(r"\[p=([0-9.]+)\]", t)
            out.append(float(m.group(1)) if m else 0.01)
        return np.array(out)


class FakeTransport:
    """Scripted OpenAI-compatible transport. `script` is a list of callables or strings; each call
    pops one. A callable receives the parsed (id, comment) items and returns the raw content."""

    def __init__(self, script=None, default=None):
        self.script = list(script or [])
        self.default = default
        self.calls: list[list[dict]] = []

    @staticmethod
    def items(messages):
        user = messages[1]["content"].splitlines()[1:]
        return [json.loads(line) for line in user]

    def __call__(self, messages):
        self.calls.append(messages)
        step = self.script.pop(0) if self.script else self.default
        if isinstance(step, Exception):
            raise step
        raw = step(self.items(messages)) if callable(step) else step
        return raw, {"prompt_tokens": 100, "completion_tokens": 40}


def verdicts_from_keywords(items):
    """'hate' -> identity_attack, 'kill' -> threat, 'dumb' -> insult, else neutral_ok."""
    res = []
    for it in items:
        c = it["comment"].lower()
        if "kill" in c:
            cat = "threat"
        elif "hate" in c:
            cat = "identity_attack"
        elif "dumb" in c:
            cat = "insult"
        elif "?" in c:
            cat = "question"
        else:
            cat = "neutral_ok"
        res.append({"id": it["id"], "category": cat,
                    "violates": cat in {"threat", "identity_attack", "insult", "toxicity"},
                    "confidence": 0.9, "reason": "test"})
    return json.dumps({"results": res})


@pytest.fixture
def cfg(tmp_path) -> Config:
    c = Config()
    c.classifier.threshold = 0.5
    c.classifier.band_low = 0.3
    c.classifier.band_high = 0.7
    c.llm.batch_size = 4
    c.llm.max_retries = 2
    c.llm.cache_path = str(tmp_path / "cache.jsonl")
    c.human_review.queue_path = str(tmp_path / "q.sqlite")
    return c


@pytest.fixture
def rules(cfg) -> RuleEngine:
    # test lexicon: a harmless placeholder word stands in for a real slur
    return RuleEngine(cfg.rules, {hash_token("badword", cfg.rules.lexicon_salt)})


@pytest.fixture
def stub_clf():
    return StubClassifier()
