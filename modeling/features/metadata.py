"""Categorical metadata feature implementations."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable, List

from sklearn.preprocessing import OneHotEncoder

from .base import FeatureBlock


@dataclass
class CategoricalMetadataBlock(FeatureBlock):
    """One-hot encode categorical metadata columns."""

    columns: List[str] = field(default_factory=list)
    drop: str | None = None
    sparse: bool = True
    handle_unknown: str = "ignore"

    @property
    def column(self) -> Iterable[str]:
        return self.columns

    def build(self) -> OneHotEncoder:
        encoder_kwargs = {
            "drop": self.drop,
            "handle_unknown": self.handle_unknown,
        }
        try:
            encoder = OneHotEncoder(sparse_output=self.sparse, **encoder_kwargs)
        except TypeError:
            encoder_kwargs["sparse"] = self.sparse
            encoder = OneHotEncoder(**encoder_kwargs)
        return encoder

    @property
    def name(self) -> str:
        return "categorical_metadata"
