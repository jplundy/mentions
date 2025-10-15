"""Reference loaders for news ingestion configs."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, Optional


def load_local_jsonl(
    *,
    path: str,
    since: Optional[datetime] = None,
    until: Optional[datetime] = None,
    limit: Optional[int] = None,
) -> Iterable[Dict[str, object]]:
    """Yield article dictionaries from a local JSONL file.

    The loader is useful for smoke-testing the ingestion CLI or replaying historical
    corpora fetched by other systems.
    """

    file_path = Path(path)
    if not file_path.exists():
        return []

    def _iter() -> Iterator[Dict[str, object]]:
        count = 0
        with file_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                record = json.loads(line)
                published_at = _coerce_datetime(record.get("published_at"))
                if since and published_at and published_at < since:
                    continue
                if until and published_at and published_at > until:
                    continue
                yield record
                count += 1
                if limit is not None and count >= limit:
                    break

    return list(_iter())


def _coerce_datetime(value: object) -> Optional[datetime]:
    if value is None:
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(float(value))
    try:
        return datetime.fromisoformat(str(value))
    except ValueError:
        return None
