"""Text feature implementations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import TfidfVectorizer

from .base import FeatureBlock


@dataclass
class TfidfTextBlock(FeatureBlock):
    """Baseline TF-IDF vectorizer."""

    column: str = "text"
    max_features: int | None = 25000
    ngram_range: tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: float = 0.9
    sublinear_tf: bool = True
    lowercase: bool = True
    analyzer: str = "word"

    def build(self) -> TransformerMixin:
        return TfidfVectorizer(
            input="content",
            max_features=self.max_features,
            ngram_range=self.ngram_range,
            min_df=self.min_df,
            max_df=self.max_df,
            sublinear_tf=self.sublinear_tf,
            lowercase=self.lowercase,
            analyzer=self.analyzer,
        )


class SentenceTransformerBlock(FeatureBlock):
    """Sentence transformer embeddings with optional pooling."""

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        column: str = "text",
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self._model_name = model_name
        self._column = column
        self._batch_size = batch_size
        self._normalize = normalize_embeddings

    @property
    def column(self) -> str:
        return self._column

    def build(self) -> TransformerMixin:
        return _SentenceTransformerEncoder(
            model_name=self._model_name,
            batch_size=self._batch_size,
            normalize_embeddings=self._normalize,
        )

    @property
    def name(self) -> str:
        return "sentence_transformer"


class _SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """Thin wrapper around sentence-transformers to integrate with sklearn."""

    def __init__(
        self,
        model_name: str,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
    ) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self._model = None

    def fit(self, X: Iterable[str], y=None):  # type: ignore[override]
        self._ensure_model()
        return self

    def transform(self, X: Iterable[str]):  # type: ignore[override]
        model = self._ensure_model()
        embeddings = model.encode(
            list(X),
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=self.normalize_embeddings,
        )
        return np.asarray(embeddings)

    def _ensure_model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name)
        return self._model
