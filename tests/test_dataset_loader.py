import json

import pytest

pd = pytest.importorskip("pandas")

from modeling.config import ExperimentConfig
from modeling.dataset import DatasetLoader


def test_dataset_loader_merges_news_features(tmp_path):
    base_path = tmp_path / "segments.parquet"
    base_frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "segment_id": "evt-1-0001",
                "text": "Opening remarks",
                "target__powell": 1,
            },
            {
                "event_id": "evt-2",
                "segment_id": "evt-2-0001",
                "text": "Q&A",
                "target__powell": 0,
            },
        ]
    )
    base_frame.to_parquet(base_path, index=False)

    news_path = tmp_path / "news.parquet"
    provenance = json.dumps([
        {"article_id": "a1", "url": "https://example.com"}
    ])
    news_frame = pd.DataFrame(
        [
            {
                "event_id": "evt-1",
                "news_article_count": 3,
                "news_unique_sources": 2,
                "news_snapshot_hash": "abc123",
                "news_provenance": provenance,
            }
        ]
    )
    news_frame.to_parquet(news_path, index=False)

    config = ExperimentConfig(dataset_path=base_path, target_column="target__powell")
    config.news_features_path = news_path

    loader = DatasetLoader(config)
    bundle = loader.load()

    assert "news_article_count" in bundle.frame.columns
    merged_row = bundle.frame.loc[bundle.frame["event_id"] == "evt-1"].iloc[0]
    assert merged_row["news_article_count"] == 3
    assert bundle.news is not None
    assert bundle.news.snapshot_hash == "abc123"
    assert bundle.news.provenance_by_event["evt-1"][0]["article_id"] == "a1"
