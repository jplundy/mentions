import json
from hashlib import sha256
from pathlib import Path

import pytest

pd = pytest.importorskip("pandas")

from news.news_features import NewsFeatureBuilder


def _write_jsonl(path: Path, records):
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record))
            handle.write("\n")


def test_news_feature_builder_generates_counts(tmp_path):
    raw_dir = tmp_path / "raw"
    raw_dir.mkdir()
    article_path = raw_dir / "wsj.jsonl"
    article = {
        "article_id": "wsj-1",
        "event_id": "2024-03",
        "source": "wsj",
        "headline": "Powell expected to mention inflation",
        "url": "https://example.com/article",
        "body_text": "Analysts debate inflation outlook and inflation policy.",
        "published_at": "2024-03-15T12:00:00Z",
    }
    _write_jsonl(article_path, [article])

    builder = NewsFeatureBuilder(["inflation"])
    output_path = tmp_path / "features.parquet"
    artifact = builder.build([article_path], output_path)

    frame = pd.read_parquet(output_path)
    assert frame.loc[0, "event_id"] == "2024-03"
    assert frame.loc[0, "news_article_count"] == 1
    assert frame.loc[0, "news_unique_sources"] == 1
    assert frame.loc[0, "news_target__inflation_count"] == 2
    assert artifact.provenance_by_event["2024-03"][0]["url"] == "https://example.com/article"

    expected_hash = sha256()
    expected_hash.update((json.dumps(article) + "\n").encode("utf-8"))
    assert artifact.snapshot_hash == expected_hash.hexdigest()
