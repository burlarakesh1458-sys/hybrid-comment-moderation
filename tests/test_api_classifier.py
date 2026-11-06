import numpy as np
from fastapi.testclient import TestClient

from hybridmod.api import create_app
from hybridmod.classifier import ClfParams, ToxicityClassifier, fit
from hybridmod.llm import LLMReviewer
from hybridmod.pipeline import HybridModerator
from hybridmod.review_queue import ReviewQueue

from .conftest import FakeTransport, verdicts_from_keywords


def client(cfg, rules, stub_clf, transport=None):
    rev = LLMReviewer(cfg.llm, transport=transport or FakeTransport(default=verdicts_from_keywords),
                      cache_path=None)
    mod = HybridModerator(cfg, rules, stub_clf, rev, queue=ReviewQueue(":memory:"))
    return TestClient(create_app(moderator=mod, cfg=cfg))


def test_health_and_config(cfg, rules, stub_clf):
    with client(cfg, rules, stub_clf) as c:
        assert c.get("/healthz").json() == {"status": "ok", "llm_enabled": True,
                                            "rules_enabled": True}
        conf = c.get("/v1/config").json()
        assert conf["classifier"]["threshold"] == 0.5 and "api_key_env" not in conf["llm"]


def test_moderate_single(cfg, rules, stub_clf):
    with client(cfg, rules, stub_clf) as c:
        r = c.post("/v1/moderate", json={"text": "lovely [p=0.02]", "comment_id": "a"}).json()
        assert r["action"] == "allow" and r["stage"] == "ml" and r["flagged"] is False


def test_moderate_batch_summary(cfg, rules, stub_clf):
    with client(cfg, rules, stub_clf) as c:
        body = {"comments": [{"text": "ok [p=0.01]"}, {"text": "badword"},
                             {"text": "I will kill [p=0.5]"}, {"text": "why? [p=0.4]"}]}
        r = c.post("/v1/moderate/batch", json=body).json()
        assert r["summary"] == {"n": 4, "hidden": 2, "human_review": 2, "llm_calls_routed": 2}


def test_validation_errors(cfg, rules, stub_clf):
    with client(cfg, rules, stub_clf) as c:
        assert c.post("/v1/moderate/batch", json={"comments": []}).status_code == 422
        assert c.post("/v1/moderate", json={"text": "x" * 10_001}).status_code == 422


def test_review_queue_flow(cfg, rules, stub_clf):
    with client(cfg, rules, stub_clf) as c:
        c.post("/v1/moderate", json={"text": "I will kill you [p=0.45]", "comment_id": "z"})
        items = c.get("/v1/review-queue").json()["items"]
        assert len(items) == 1 and items[0]["review_reason"] == "penalty_category:threat"
        iid = items[0]["id"]
        r = c.post(f"/v1/review-queue/{iid}/resolve",
                   json={"human_action": "penalize", "reviewer": "mod1"})
        assert r.status_code == 200 and r.json()["status"] == "resolved"
        again = c.post(f"/v1/review-queue/{iid}/resolve",
                       json={"human_action": "allow", "reviewer": "mod1"})
        assert again.status_code == 409
        assert c.post("/v1/review-queue/999/resolve",
                      json={"human_action": "allow", "reviewer": "m"}).status_code == 404
        assert c.post(f"/v1/review-queue/{iid}/resolve",
                      json={"human_action": "nuke", "reviewer": "m"}).status_code == 422


def test_tiny_classifier_trains_and_calibrates(tmp_path):
    rng = np.random.default_rng(0)
    good = ["thanks for the thoughtful article", "interesting point about the budget",
            "i agree with the council decision", "the weather was nice today"]
    bad = ["you are a pathetic idiot", "shut up moron", "what a stupid clown",
           "idiot people like you"]
    X = [good[i % 4] + f" {rng.integers(1000)}" for i in range(60)] + \
        [bad[i % 4] + f" {rng.integers(1000)}" for i in range(60)]
    y = [0] * 60 + [1] * 60
    p = ClfParams(word_max_features=500, char_hash_features=2**12, calibration="sigmoid", min_df=1)
    clf = ToxicityClassifier(fit(X, y, p))
    probs = clf.predict_proba(["you idiot moron", "thanks, interesting article"])
    assert probs[0] > 0.5 > probs[1]
    clf.save(tmp_path / "m.joblib")
    again = ToxicityClassifier.load(tmp_path / "m.joblib").predict_proba(["you idiot moron"])
    assert abs(again[0] - probs[0]) < 1e-9


def test_sample_weights_are_accepted():
    X = ["bad idiot"] * 10 + ["good day"] * 10
    y = [1] * 10 + [0] * 10
    p = ClfParams(word_max_features=50, char_hash_features=2**10, calibration="none", min_df=1)
    model = fit(X, y, p, sample_weight=np.r_[np.ones(10), np.full(10, 3.0)])
    assert model.predict_proba(["good day"])[0, 1] < 0.5
