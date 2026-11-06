import numpy as np
import pytest

from hybridmod.config import ClassifierCfg, load_config
from hybridmod.metrics import (
    best_f1_threshold,
    binary_metrics,
    cost_per_10k,
    percentiles,
    subgroup_fpr,
)
from hybridmod.pipeline import Decision
from hybridmod.review_queue import ReviewQueue


def test_queue_enqueue_resolve_and_stats(tmp_path):
    q = ReviewQueue(tmp_path / "q.sqlite")
    d = Decision("hide", "llm", p_toxic=0.4, category="threat", human_review=True,
                 review_reason="penalty_category:threat")
    i = q.enqueue("c9", "text", d)
    assert q.get(i)["status"] == "pending"
    r = q.resolve(i, "penalize", "mod@team", "clear threat")
    assert r["status"] == "resolved" and r["human_action"] == "penalize"
    assert q.stats()["resolved"] == 1 and q.list() == []


def test_queue_rejects_bad_actions_and_double_resolve():
    q = ReviewQueue()
    i = q.enqueue(None, "t", Decision("hide", "rules", review_reason="rule_block"))
    with pytest.raises(ValueError):
        q.resolve(i, "ban_forever", "me")
    q.resolve(i, "allow", "me")
    with pytest.raises(ValueError):
        q.resolve(i, "hide", "me")
    with pytest.raises(KeyError):
        q.resolve(999, "hide", "me")


def test_binary_metrics_known_values():
    m = binary_metrics([1, 1, 0, 0, 1], [1, 0, 1, 0, 1])
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (2, 1, 1, 1)
    assert m["precision"] == pytest.approx(0.6667, abs=1e-4)
    assert m["f1"] == pytest.approx(0.6667, abs=1e-4)


def test_binary_metrics_degenerate():
    m = binary_metrics([0, 0], [0, 0])
    assert m["f1"] == 0.0 and m["precision"] == 0.0


def test_percentiles_and_cost():
    assert percentiles([1, 2, 3, 4, 100])["p50"] == 3
    assert percentiles([])["p95"] is None
    # 1M prompt + 1M completion tokens over 10k comments at $0.15/$0.60 -> $0.75 per 10k
    assert cost_per_10k(1e6, 1e6, 10_000, 0.15, 0.60) == pytest.approx(0.75)
    assert cost_per_10k(1e6, 0, 5_000, 0.15, 0.60) == pytest.approx(0.30)


def test_best_threshold_finds_separator():
    y = np.array([0] * 50 + [1] * 50)
    p = np.r_[np.linspace(0, 0.3, 50), np.linspace(0.6, 1, 50)]
    t, f1 = best_f1_threshold(y, p)
    assert 0.3 < t <= 0.6 and f1 == 1.0


def test_subgroup_fpr_gap():
    y = np.array([0, 0, 0, 0, 1, 1])
    pred = np.array([1, 0, 0, 0, 1, 1])
    mask = np.array([1, 1, 0, 0, 1, 0])
    r = subgroup_fpr(y, pred, mask)
    assert r["mentions_identity"]["fpr"] == 0.5 and r["no_identity"]["fpr"] == 0.0
    assert r["fpr_gap"] == 0.5


def test_config_loads_repo_yaml_and_validates_band():
    cfg = load_config()
    c = cfg.classifier
    assert 0 <= c.band_low <= c.threshold <= c.band_high <= 1
    with pytest.raises(ValueError):
        ClassifierCfg(band_low=0.8, band_high=0.2)


def test_config_env_override(tmp_path, monkeypatch):
    p = tmp_path / "m.yaml"
    p.write_text("classifier:\n  threshold: 0.42\n  band_low: 0.1\n  band_high: 0.9\n")
    monkeypatch.setenv("MODERATION_CONFIG", str(p))
    assert load_config().classifier.threshold == 0.42
