"""Typed config loaded from configs/moderation.yaml (path overridable with MODERATION_CONFIG)."""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, model_validator

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = ROOT / "configs" / "moderation.yaml"


class RulesCfg(BaseModel):
    enabled: bool = True
    lexicon_path: str = "configs/lexicon_sha256.txt"
    lexicon_salt: str = "hybridmod-v1"
    allow_no_letters: bool = True
    allow_short_benign_max_words: int = 4
    spam_min_links: int = 3
    spam_max_repeat_ratio: float = 0.5


class ClassifierCfg(BaseModel):
    model_path: str = "models/tfidf_lr.joblib"
    threshold: float = Field(0.5, ge=0, le=1)
    band_low: float = Field(0.2, ge=0, le=1)
    band_high: float = Field(0.8, ge=0, le=1)

    @model_validator(mode="after")
    def _band_ok(self) -> ClassifierCfg:
        if not self.band_low <= self.band_high:
            raise ValueError("band_low must be <= band_high")
        return self


class LLMCfg(BaseModel):
    enabled: bool = True
    provider: str = "groq"
    model: str = "openai/gpt-oss-20b"
    base_url: str = "https://api.groq.com/openai/v1"
    api_key_env: str = "GROQ_API_KEY"
    batch_size: int = Field(8, ge=1, le=50)
    max_retries: int = Field(3, ge=0, le=10)
    timeout_s: float = 60
    reasoning_effort: str | None = "low"
    cache_path: str = "results/llm_cache.jsonl"
    price_input_per_m: float = 0.075
    price_output_per_m: float = 0.30


class HumanReviewCfg(BaseModel):
    queue_path: str = "results/review_queue.sqlite"
    penalty_categories: list[str] = ["threat", "identity_attack"]
    penalty_min_p_toxic: float = 0.9
    queue_rule_blocks: bool = True
    queue_llm_failures: bool = True


class ExperimentCfg(BaseModel):
    llm_sample_size: int = 600


class Config(BaseModel):
    rules: RulesCfg = RulesCfg()
    classifier: ClassifierCfg = ClassifierCfg()
    llm: LLMCfg = LLMCfg()
    human_review: HumanReviewCfg = HumanReviewCfg()
    experiment: ExperimentCfg = ExperimentCfg()

    def path(self, rel: str) -> Path:
        """Resolve a config path relative to the repo root (absolute paths pass through)."""
        p = Path(rel)
        return p if p.is_absolute() else ROOT / p


def load_config(path: str | os.PathLike | None = None) -> Config:
    path = Path(path or os.environ.get("MODERATION_CONFIG", DEFAULT_CONFIG))
    data = yaml.safe_load(path.read_text()) if path.exists() else {}
    return Config.model_validate(data or {})
