"""News ingestion utilities."""

from __future__ import annotations

import importlib
import json
import logging
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence

from .config import NewsIngestConfig, NewsSourceConfig

logger = logging.getLogger(__name__)


@dataclass
class NewsArticle:
    """Normalized representation of a news article."""

    article_id: str
    event_id: str
    source: str
    published_at: str
    headline: str
    url: str
    body_text: str
    tags: List[str] = field(default_factory=list)
    provenance: Optional[str] = None
    extras: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        data = {
            "article_id": self.article_id,
            "event_id": self.event_id,
            "source": self.source,
            "published_at": self.published_at,
            "headline": self.headline,
            "url": self.url,
            "body_text": self.body_text,
            "tags": self.tags,
        }
        if self.provenance is not None:
            data["provenance"] = self.provenance
        data.update(self.extras)
        return data


@dataclass
class NewsIngestReport:
    """Summary of a single ingestion run."""

    run_id: str
    output_files: List[Path]
    totals: Dict[str, int]
    window: Dict[str, Optional[str]]


class NewsIngestor:
    """Coordinate loading articles from configured sources and persisting them."""

    def __init__(self, config: NewsIngestConfig) -> None:
        self.config = config

    def ingest(
        self,
        *,
        since: Optional[datetime] = None,
        until: Optional[datetime] = None,
        limit: Optional[int] = None,
        dry_run: bool = False,
    ) -> NewsIngestReport:
        run_id = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        output_files: List[Path] = []
        totals: Dict[str, int] = {}

        logger.info("Starting news ingestion run %s", run_id)
        for source in self.config.iter_sources():
            logger.info("Fetching articles for source %s", source.name)
            loader = self._resolve_loader(source)
            article_dicts = list(
                self._load_articles(loader, source, since=since, until=until, limit=limit)
            )
            normalized = [self._normalize_article(item, source) for item in article_dicts]
            totals[source.name] = len(normalized)
            if dry_run:
                continue
            if not normalized:
                logger.info("No new articles for %s", source.name)
                continue
            output_path = self._write_articles(run_id, source.name, normalized)
            output_files.append(output_path)
            logger.info(
                "Wrote %s articles for %s to %s", len(normalized), source.name, output_path
            )

        manifest_path = self._write_manifest(run_id, output_files, totals, since, until, dry_run)
        if manifest_path:
            output_files.append(manifest_path)
        window = {
            "since": since.astimezone(timezone.utc).isoformat() if since else None,
            "until": until.astimezone(timezone.utc).isoformat() if until else None,
        }
        return NewsIngestReport(run_id=run_id, output_files=output_files, totals=totals, window=window)

    def _resolve_loader(self, source: NewsSourceConfig) -> Callable[..., Iterable[Dict[str, Any]]]:
        module_name, attr = source.loader.rsplit(".", 1)
        module = importlib.import_module(module_name)
        loader = getattr(module, attr)
        if not callable(loader):
            raise TypeError(f"Configured loader {source.loader!r} is not callable")
        return loader

    def _load_articles(
        self,
        loader: Callable[..., Iterable[Dict[str, Any]]],
        source: NewsSourceConfig,
        *,
        since: Optional[datetime],
        until: Optional[datetime],
        limit: Optional[int],
    ) -> Iterable[Dict[str, Any]]:
        kwargs = dict(source.params)
        if since:
            kwargs.setdefault("since", since)
        if until:
            kwargs.setdefault("until", until)
        if limit is not None:
            kwargs.setdefault("limit", limit)
        results = loader(**kwargs)
        if isinstance(results, dict):
            raise TypeError(
                f"Loader {source.loader!r} returned a dict; expected an iterable of article dicts"
            )
        for item in results:
            if not isinstance(item, dict):
                raise TypeError(
                    f"Loader {source.loader!r} yielded a non-dict article: {type(item).__name__}"
                )
            yield item

    def _normalize_article(
        self, article: Dict[str, Any], source: NewsSourceConfig
    ) -> NewsArticle:
        required = [
            "article_id",
            "event_id",
            "source",
            "published_at",
            "headline",
            "url",
            "body_text",
        ]
        missing = [key for key in required if not article.get(key)]
        if missing:
            raise ValueError(
                f"Article from {source.name!r} missing required fields: {', '.join(missing)}"
            )
        tags = article.get("tags") or []
        if isinstance(tags, str):
            tags = [tags]
        provenance = article.get("provenance")
        extras = {
            key: value
            for key, value in article.items()
            if key not in set(required + ["tags", "provenance"])
        }
        extras.setdefault("ingested_at", datetime.utcnow().replace(tzinfo=timezone.utc).isoformat())
        extras.setdefault("ingest_source", source.name)
        return NewsArticle(
            article_id=str(article["article_id"]),
            event_id=str(article["event_id"]),
            source=str(article["source"]),
            published_at=str(article["published_at"]),
            headline=str(article["headline"]),
            url=str(article["url"]),
            body_text=str(article["body_text"]),
            tags=[str(tag) for tag in tags],
            provenance=str(provenance) if provenance is not None else None,
            extras=extras,
        )

    def _write_articles(self, run_id: str, source_name: str, articles: Sequence[NewsArticle]) -> Path:
        source_dir = self.config.raw_directory / source_name
        source_dir.mkdir(parents=True, exist_ok=True)
        output_path = source_dir / f"{run_id}.jsonl"
        with output_path.open("w", encoding="utf-8") as handle:
            for article in articles:
                handle.write(json.dumps(article.to_dict(), ensure_ascii=False))
                handle.write("\n")
        return output_path

    def _write_manifest(
        self,
        run_id: str,
        output_files: List[Path],
        totals: Dict[str, int],
        since: Optional[datetime],
        until: Optional[datetime],
        dry_run: bool,
    ) -> Optional[Path]:
        manifest = {
            "run_id": run_id,
            "generated_at": datetime.utcnow().replace(tzinfo=timezone.utc).isoformat(),
            "dry_run": dry_run,
            "window": {
                "since": since.astimezone(timezone.utc).isoformat() if since else None,
                "until": until.astimezone(timezone.utc).isoformat() if until else None,
            },
            "totals": totals,
            "outputs": [str(path) for path in output_files],
        }
        if dry_run:
            logger.info("Dry run manifest: %s", json.dumps(manifest, indent=2))
            return None
        manifest_path = self.config.manifest_directory / f"{run_id}_manifest.json"
        self.config.manifest_directory.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        return manifest_path
