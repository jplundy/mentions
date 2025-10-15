"""Dataset loading utilities for modeling experiments."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import pandas as pd

from .config import ExperimentConfig


@dataclass
class NewsMetadata:
    """Metadata describing the news feature snapshot used in an experiment."""

    snapshot_hash: Optional[str]
    provenance_by_event: Dict[str, list]


@dataclass
class DatasetBundle:
    """Container bundling the modeling frame with optional auxiliary metadata."""

    frame: pd.DataFrame
    news: Optional[NewsMetadata] = None


class DatasetLoader:
    """Load the baseline dataset and merge optional auxiliary features."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config

    def load(self) -> DatasetBundle:
        base = self._load_table(self.config.dataset_path)
        news_metadata: Optional[NewsMetadata] = None

        if self.config.news_features_path:
            news_frame = self._load_table(self.config.news_features_path)
            news_metadata = self._extract_news_metadata(news_frame)
            base = self._merge_news_features(base, news_frame)

        return DatasetBundle(frame=base, news=news_metadata)

    def _load_table(self, path: Path) -> pd.DataFrame:
        if path.suffix == ".parquet":
            return pd.read_parquet(path)
        if path.suffix in {".csv", ".tsv"}:
            return pd.read_csv(path)
        raise ValueError(f"Unsupported dataset format: {path.suffix}")

    def _merge_news_features(self, base: pd.DataFrame, news: pd.DataFrame) -> pd.DataFrame:
        if "event_id" not in news.columns:
            raise ValueError("News feature table must include an event_id column")
        news_features = news.copy()
        metadata_columns = {"news_snapshot_hash", "news_provenance"}
        available_metadata = metadata_columns & set(news_features.columns)
        news_features = news_features.drop(columns=list(available_metadata))
        merged = base.merge(news_features, on="event_id", how="left", suffixes=("", "_news"))
        return merged

    def _extract_news_metadata(self, news: pd.DataFrame) -> NewsMetadata:
        snapshot_hash = None
        if "news_snapshot_hash" in news.columns:
            unique_hashes = (
                news["news_snapshot_hash"].dropna().astype(str).unique().tolist()
            )
            if unique_hashes:
                snapshot_hash = unique_hashes[0]
        provenance_map: Dict[str, list] = {}
        if "news_provenance" in news.columns:
            for event_id, value in news.set_index("event_id")["news_provenance"].items():
                if pd.isna(value):
                    continue
                parsed = value
                if isinstance(value, str):
                    try:
                        parsed = json.loads(value)
                    except json.JSONDecodeError:
                        parsed = value
                provenance_map[str(event_id)] = parsed  # type: ignore[assignment]
        return NewsMetadata(snapshot_hash=snapshot_hash, provenance_by_event=provenance_map)


def load_dataset(config: ExperimentConfig) -> DatasetBundle:
    """Convenience helper mirroring the previous load_dataset signature."""

    loader = DatasetLoader(config)
    return loader.load()
