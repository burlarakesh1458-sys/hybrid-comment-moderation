"""Stage 2: TF-IDF (word + char n-grams) + logistic regression with calibrated probabilities.

Calibration uses CalibratedClassifierCV(cv=3) on the training split itself, so the separate
calibration split stays untouched for choosing the decision threshold and the LLM band.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer, TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import FeatureUnion, Pipeline


@dataclass
class ClfParams:
    word_max_features: int = 100_000
    char_hash_features: int = 2**19
    word_ngram: tuple[int, int] = (1, 2)
    char_ngram: tuple[int, int] = (3, 5)
    C: float = 4.0
    class_weight: str | None = None
    calibration: str = "isotonic"  # isotonic | sigmoid | none
    min_df: int = 2
    extra: dict = field(default_factory=dict)


def build_features(p: ClfParams) -> FeatureUnion:
    return FeatureUnion(
        [
            (
                "word",
                TfidfVectorizer(
                    lowercase=True, ngram_range=p.word_ngram, max_features=p.word_max_features,
                    min_df=p.min_df, sublinear_tf=True, strip_accents="unicode",
                    token_pattern=r"(?u)\b\w+\b", dtype=np.float32,
                ),
            ),
            (
                # hashed char n-grams: constant memory (no vocabulary), robust to misspellings
                "char",
                Pipeline([
                    ("hash", HashingVectorizer(
                        lowercase=True, analyzer="char_wb", ngram_range=p.char_ngram,
                        n_features=p.char_hash_features, alternate_sign=False, norm=None,
                        dtype=np.float32,
                    )),
                    ("tfidf", TfidfTransformer(sublinear_tf=True)),
                ]),
            ),
        ]
    )


def build_model(p: ClfParams) -> Pipeline:
    lr = LogisticRegression(C=p.C, class_weight=p.class_weight, max_iter=2000, solver="liblinear")
    clf = lr if p.calibration == "none" else CalibratedClassifierCV(lr, method=p.calibration, cv=3)
    return Pipeline([("features", build_features(p)), ("clf", clf)])


def fit(texts, y, p: ClfParams, sample_weight=None) -> Pipeline:
    model = build_model(p)
    kw = {"clf__sample_weight": sample_weight} if sample_weight is not None else {}
    model.fit(list(texts), np.asarray(y), **kw)
    return model


class ToxicityClassifier:
    """Thin wrapper so the pipeline/API depend on predict_proba only (easy to stub in tests)."""

    def __init__(self, model):
        self.model = model

    @classmethod
    def load(cls, path: str | Path) -> ToxicityClassifier:
        return cls(joblib.load(path))

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, path, compress=3)

    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return self.model.predict_proba(list(texts))[:, 1]
