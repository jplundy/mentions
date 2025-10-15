"""Feature generation from ingested news articles."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from hashlib import sha256
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import pandas as pd

from pipeline.dataset import DatasetPublisher, _compile_target_pattern

logger = logging.getLogger(__name__)


@dataclass
class NewsFeatureArtifact:
    """Artifact produced when building news-derived features."""

    features_path: Path
    snapshot_hash: str
    provenance_by_event: Dict[str, List[Dict[str, str]]]


class NewsFeatureBuilder:
    """Build aggregate features from raw article JSONL dumps."""

    def __init__(self, target_terms: Sequence[str]):
        self.target_terms = list(target_terms)
        self._patterns = {
            term: _compile_target_pattern(term)
            for term in self.target_terms
        }
        self._target_columns = {
            term: self._news_target_column(term)
            for term in self.target_terms
        }

    def build(
        self,
        article_files: Iterable[Path],
        output_path: Path,
    ) -> NewsFeatureArtifact:
        article_paths = list(article_files)
        records, provenance_by_event = self._collect_records(article_paths)
        if not records:
            logger.warning("No news articles discovered; emitting empty feature table")
            prototype = self._initialize_row("event")
            prototype.pop("_sources", None)
            frame = pd.DataFrame(columns=prototype.keys())
        else:
            frame = pd.DataFrame(records)
            frame.sort_values("event_id", inplace=True, ignore_index=True)
        snapshot_hash = self._compute_snapshot_hash(article_paths)
        if "news_snapshot_hash" not in frame.columns:
            frame["news_snapshot_hash"] = snapshot_hash
        else:
            frame["news_snapshot_hash"].fillna(snapshot_hash, inplace=True)
        if "news_provenance" not in frame.columns and "event_id" in frame.columns:
            frame["news_provenance"] = frame["event_id"].map(
                lambda event: json.dumps(provenance_by_event.get(event, []), ensure_ascii=False)
            )
        output_path.parent.mkdir(parents=True, exist_ok=True)
        frame.to_parquet(output_path, index=False)
        return NewsFeatureArtifact(
            features_path=output_path,
            snapshot_hash=snapshot_hash,
            provenance_by_event=provenance_by_event,
        )

    def build_from_directory(self, raw_directory: Path, output_path: Path) -> NewsFeatureArtifact:
        article_files = sorted(raw_directory.glob("**/*.jsonl"))
        return self.build(article_files, output_path)

    def _collect_records(
        self, article_files: Iterable[Path]
    ) -> Tuple[List[Dict[str, object]], Dict[str, List[Dict[str, str]]]]:
        aggregated: Dict[str, Dict[str, object]] = {}
        provenance: Dict[str, List[Dict[str, str]]] = {}
        seen_articles: set[str] = set()

        for path in article_files:
            if not path.exists():
                logger.warning("Skipping missing article dump %s", path)
                continue
            for line in path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Failed to parse article from %s: %s", path, exc)
                    continue
                article_id = str(record.get("article_id", ""))
                if not article_id:
                    logger.warning("Article without id encountered in %s", path)
                    continue
                if article_id in seen_articles:
                    continue
                seen_articles.add(article_id)
                event_id = str(record.get("event_id", ""))
                if not event_id:
                    logger.warning("Article %s missing event_id; skipping", article_id)
                    continue
                aggregated.setdefault(event_id, self._initialize_row(event_id))
                provenance.setdefault(event_id, [])
                normalized = self._normalize_article(record)
                self._update_row(aggregated[event_id], normalized)
                provenance[event_id].append(
                    {
                        "article_id": normalized["article_id"],
                        "source": normalized["source"],
                        "headline": normalized["headline"],
                        "url": normalized.get("url", ""),
                        "published_at": normalized.get("published_at", ""),
                    }
                )

        rows = []
        for row in aggregated.values():
            row.pop("_sources", None)
            rows.append(row)
        return rows, provenance

    def _initialize_row(self, event_id: str) -> Dict[str, object]:
        row: Dict[str, object] = {
            "event_id": event_id,
            "news_article_count": 0,
            "news_unique_sources": 0,
            "news_average_body_length": 0.0,
            "news_first_published_at": None,
            "news_latest_published_at": None,
        }
        for column in self._target_columns.values():
            row[column] = 0
        return row

    def _normalize_article(self, record: Dict[str, object]) -> Dict[str, object]:
        body_text = str(record.get("body_text", ""))
        headline = str(record.get("headline", ""))
        source = str(record.get("source", ""))
        published_at = record.get("published_at")
        published_iso = None
        if published_at:
            published_iso = self._coerce_datetime(published_at)
        normalized = {
            "article_id": str(record.get("article_id", "")),
            "event_id": str(record.get("event_id", "")),
            "source": source,
            "headline": headline,
            "url": str(record.get("url", "")),
            "body_text": body_text,
            "published_at": published_iso,
        }
        counts = self._count_targets(headline + "\n" + body_text)
        normalized.update(counts)
        normalized["body_length"] = len(body_text)
        return normalized

    def _update_row(self, row: Dict[str, object], article: Dict[str, object]) -> None:
        row["news_article_count"] = int(row.get("news_article_count", 0)) + 1
        sources: set[str] = set(row.get("_sources", []))
        sources.add(article["source"])
        row["_sources"] = list(sources)
        row["news_unique_sources"] = len(sources)
        current_length = float(row.get("news_average_body_length", 0.0))
        count = int(row["news_article_count"])
        row["news_average_body_length"] = (
            (current_length * (count - 1) + article["body_length"]) / count
        )
        published_at = article.get("published_at")
        if published_at:
            first = row.get("news_first_published_at")
            latest = row.get("news_latest_published_at")
            row["news_first_published_at"] = (
                published_at if first is None or published_at < first else first
            )
            row["news_latest_published_at"] = (
                published_at if latest is None or published_at > latest else latest
            )
        for term, column in self._target_columns.items():
            row[column] = int(row.get(column, 0)) + int(article.get(column, 0))

    def _count_targets(self, text: str) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        text_lower = text.lower()
        for term, pattern in self._patterns.items():
            column = self._target_columns[term]
            if pattern is None:
                counts[column] = text_lower.count(term.lower())
            else:
                counts[column] = len(pattern.findall(text))
        return counts

    def _coerce_datetime(self, value: object) -> str:
        if isinstance(value, datetime):
            return value.astimezone().isoformat()
        if isinstance(value, (int, float)):
            return datetime.fromtimestamp(float(value)).isoformat()
        text = str(value)
        try:
            return datetime.fromisoformat(text).isoformat()
        except ValueError:
            logger.debug("Unable to parse datetime %s; storing raw value", text)
            return text

    def _news_target_column(self, term: str) -> str:
        base = DatasetPublisher._target_column_name(term)
        return f"news_{base}_count"

    def _compute_snapshot_hash(self, article_files: Iterable[Path]) -> str:
        digest = sha256()
        for path in sorted(article_files):
            if not path.exists():
                continue
            with path.open("r", encoding="utf-8") as handle:
                for line in handle:
                    if not line.strip():
                        continue
                    digest.update(line.encode("utf-8"))
        return digest.hexdigest()
