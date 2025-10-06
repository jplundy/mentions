"""Time-aware validation strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from .config import ValidationConfig


@dataclass
class SplitIndices:
    """Indices for a train/validation split."""

    train: np.ndarray
    validate: np.ndarray


class RollingTimeSplit:
    """Rolling time series split with minimum training events."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> Iterator[SplitIndices]:
        df_sorted = df.sort_values(self.config.date_column).reset_index(drop=True)
        splitter = TimeSeriesSplit(
            n_splits=self.config.n_splits,
            test_size=max(1, len(df_sorted) // (self.config.n_splits + 1)),
        )
        for train_index, test_index in splitter.split(df_sorted):
            if len(train_index) < self.config.minimum_train_events:
                continue
            yield SplitIndices(train=train_index, validate=test_index)


class LeaveOneEventOut:
    """Leave-one-event-out cross validation based on event groups."""

    def __init__(self, config: ValidationConfig) -> None:
        self.config = config

    def split(self, df: pd.DataFrame) -> Iterator[SplitIndices]:
        groups = df[self.config.group_column].astype("category").cat.codes.values
        unique_groups = np.unique(groups)
        for group in unique_groups:
            validate_mask = groups == group
            train_mask = ~validate_mask
            train_index = np.where(train_mask)[0]
            validate_index = np.where(validate_mask)[0]
            if len(train_index) < self.config.minimum_train_events:
                continue
            yield SplitIndices(train=train_index, validate=validate_index)


def build_validator(config: ValidationConfig):
    """Create a validator instance based on configuration."""

    if config.strategy == "rolling":
        return RollingTimeSplit(config)
    if config.strategy == "leave_one_event_out":
        return LeaveOneEventOut(config)
    raise ValueError(f"Unsupported validation strategy: {config.strategy}")
