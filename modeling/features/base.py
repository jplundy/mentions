"""Base feature interfaces for modeling."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Protocol

from sklearn.base import TransformerMixin


class SupportsTransform(Protocol):
    """Protocol for transformers used in pipelines."""

    def fit(self, X, y=None):  # type: ignore[override]
        ...

    def transform(self, X):  # type: ignore[override]
        ...


class FeatureBlock(ABC):
    """Base class for modular feature blocks."""

    @abstractmethod
    def build(self) -> TransformerMixin:
        """Return a scikit-learn compatible transformer."""

    @property
    @abstractmethod
    def column(self) -> str | Iterable[str]:
        """The column(s) consumed by the feature block."""

    @property
    def name(self) -> str:
        """Human-readable name used in column transformers."""

        return self.__class__.__name__.lower()
