"""Command line interface for the news ingestion workflow."""

from __future__ import annotations

import argparse
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from news import NewsIngestConfig, NewsIngestor

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
logger = logging.getLogger("mentions.news")


def _parse_timestamp(value: Optional[str]) -> Optional[datetime]:
    if value is None:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except ValueError as exc:  # pragma: no cover - defensive guard
        raise argparse.ArgumentTypeError(f"Invalid ISO-8601 timestamp: {value}") from exc
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Schedule and execute news ingestion jobs")
    parser.add_argument("config", type=Path, help="Path to news ingestion YAML/JSON config")
    parser.add_argument(
        "--since",
        type=str,
        default=None,
        help="Optional ISO-8601 lower bound for publication timestamps",
    )
    parser.add_argument(
        "--until",
        type=str,
        default=None,
        help="Optional ISO-8601 upper bound for publication timestamps",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of articles to request per source",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Resolve loaders and print planned outputs without writing to disk",
    )
    return parser


def run_ingestion(args: argparse.Namespace) -> None:
    config = NewsIngestConfig.from_file(args.config)
    config.ensure_directories()
    since = _parse_timestamp(args.since)
    until = _parse_timestamp(args.until)
    ingestor = NewsIngestor(config)
    report = ingestor.ingest(since=since, until=until, limit=args.limit, dry_run=args.dry_run)

    for source, total in sorted(report.totals.items()):
        logger.info("%s: %s articles", source, total)
    logger.info("Manifest window: %s", report.window)
    if report.output_files:
        logger.info("Artifacts: %s", ", ".join(str(path) for path in report.output_files))


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()
    run_ingestion(args)


if __name__ == "__main__":
    main()
