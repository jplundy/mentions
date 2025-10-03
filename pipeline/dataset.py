"""Dataset publishing utilities for segmented transcripts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    import pandas as pd
except Exception:  # pragma: no cover
    pd = None

try:  # pragma: no cover - optional dependency
    import pyarrow as pa
    import pyarrow.parquet as pq
except Exception:  # pragma: no cover
    pa = None
    pq = None

from .inventory import TranscriptInventory, TranscriptRecord
from .segmentation import Segment

logger = logging.getLogger(__name__)


@dataclass
class ValidationResult:
    """Summary of validation checks performed on the dataset."""

    checks: Dict[str, bool] = field(default_factory=dict)
    notes: List[str] = field(default_factory=list)

    def record(self, name: str, passed: bool, note: Optional[str] = None) -> None:
        self.checks[name] = passed
        if note:
            self.notes.append(note)

    def as_dict(self) -> Dict[str, object]:
        return {"checks": self.checks, "notes": self.notes}


@dataclass
class DatasetPublisher:
    """Publish cleaned transcript segments to persistent storage."""

    inventory: TranscriptInventory
    output_dir: Path
    target_words: Iterable[str]

    def __post_init__(self) -> None:
        # Normalize the target words to a list to avoid exhausting generators.
        self.target_words = [word for word in self.target_words]
        mapping = {}
        seen_columns = set()
        for word in self.target_words:
            column = self._target_column_name(word)
            if column in seen_columns:
                raise ValueError(
                    f"Duplicate target column {column!r} generated from target words"
                )
            mapping[word] = column
            seen_columns.add(column)
        self._target_mapping = mapping

    def _segment_rows(
        self, record: TranscriptRecord, segments: Iterable[Segment]
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        target_pairs = [(word, word.lower()) for word in self.target_words]
        for segment in segments:
            text_lower = segment.text.lower()
            target_hits = {
                self._target_mapping[original]: lower in text_lower
                for original, lower in target_pairs
            }
            row = {
                "event_id": record.event_id,
                "segment_id": segment.segment_id,
                "event_type": record.event_type,
                "event_date": record.event_date.date().isoformat(),
                "speaker": segment.speaker,
                "text": segment.text,
                "start_char": segment.start_char,
                "end_char": segment.end_char,
                "source_url": record.source_url,
                "provenance": record.provenance,
            }
            row.update(record.extra)
            row.update(target_hits)
            rows.append(row)
        return rows

    def build_rows(self, segmented_records: Dict[str, List[Segment]]) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        for record in self.inventory:
            segments = segmented_records.get(record.event_id, [])
            rows.extend(self._segment_rows(record, segments))
        return rows

    def validate(self, rows: List[Dict[str, object]]) -> ValidationResult:
        result = ValidationResult()
        if not rows:
            result.record("non_empty", False, "No rows produced from segmentation")
            return result
        segment_ids = [row["segment_id"] for row in rows]
        unique_segment_ids = set(segment_ids)
        result.record("unique_segment_ids", len(segment_ids) == len(unique_segment_ids))
        empty_text_segments = [row for row in rows if not str(row.get("text", "")).strip()]
        result.record("non_empty_text", not empty_text_segments)
        return result

    def publish(
        self,
        rows: List[Dict[str, object]],
        version: Optional[str] = None,
        validation: Optional[ValidationResult] = None,
    ) -> Path:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        version = version or timestamp
        dataset_path = self.output_dir / f"segments_{version}.parquet"
        manifest_path = self.output_dir / f"segments_{version}.json"
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._write_parquet(dataset_path, rows)
        manifest = {
            "version": version,
            "generated_at": timestamp,
            "num_rows": len(rows),
            "validation": validation.as_dict() if validation else None,
            "target_columns": self._target_mapping,
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        logger.info("Published dataset to %s", dataset_path)
        return dataset_path

    def _write_parquet(self, path: Path, rows: List[Dict[str, object]]) -> None:
        if pd is not None:
            frame = pd.DataFrame(rows)
            frame.to_parquet(path, index=False)
            return
        if pq is not None and pa is not None:
            table = pa.Table.from_pylist(rows)
            pq.write_table(table, path)
            return
        raise RuntimeError("Neither pandas nor pyarrow is available to write Parquet files")

    @staticmethod
    def _target_column_name(original: str) -> str:
        normalized = re.sub(r"[^a-z0-9]+", "_", original.lower()).strip("_")
        if not normalized:
            normalized = "term"
        return f"target__{normalized}"
