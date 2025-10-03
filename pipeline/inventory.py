"""Transcript inventory management."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional

from .config import PipelineConfig


@dataclass
class TranscriptRecord:
    """Metadata for a single transcript PDF."""

    event_id: str
    event_type: str
    event_date: datetime
    speakers: List[str]
    pdf_path: Path
    source_url: Optional[str] = None
    provenance: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, str]:
        base = {
            "event_id": self.event_id,
            "event_type": self.event_type,
            "event_date": self.event_date.date().isoformat(),
            "speakers": ", ".join(self.speakers),
            "pdf_path": str(self.pdf_path),
        }
        if self.source_url:
            base["source_url"] = self.source_url
        if self.provenance:
            base["provenance"] = self.provenance
        base.update(self.extra)
        return base


class TranscriptInventory:
    """Load and validate transcript metadata and associated PDF paths."""

    def __init__(self, records: Iterable[TranscriptRecord]):
        self._records = list(records)
        self._validate_unique_ids()

    def _validate_unique_ids(self) -> None:
        seen = set()
        for record in self._records:
            if record.event_id in seen:
                raise ValueError(f"Duplicate event_id found in inventory: {record.event_id}")
            seen.add(record.event_id)

    def __iter__(self) -> Iterator[TranscriptRecord]:
        yield from self._records

    def __len__(self) -> int:
        return len(self._records)

    def to_rows(self) -> List[Dict[str, str]]:
        return [record.as_dict() for record in self._records]

    @classmethod
    def from_config(cls, config: PipelineConfig) -> "TranscriptInventory":
        path = config.inventory_path
        if not path.exists():
            raise FileNotFoundError(f"Inventory file not found: {path}")
        if path.suffix.lower() in {".json", ".jsonl"}:
            records = cls._load_json(path, config)
        elif path.suffix.lower() in {".csv"}:
            records = cls._load_csv(path, config)
        else:
            raise ValueError(f"Unsupported inventory format: {path.suffix}")
        return cls(records)

    @staticmethod
    def _parse_date(value: str) -> datetime:
        for fmt in ("%Y-%m-%d", "%m/%d/%Y", "%Y%m%d"):
            try:
                return datetime.strptime(value, fmt)
            except ValueError:
                continue
        raise ValueError(f"Unable to parse event_date: {value}")

    @staticmethod
    def _normalize_speakers(value: str | Iterable[str]) -> List[str]:
        if isinstance(value, str):
            cleaned = [speaker.strip() for speaker in value.split(",") if speaker.strip()]
        else:
            cleaned = [str(speaker).strip() for speaker in value if str(speaker).strip()]
        return cleaned

    @classmethod
    def _load_json(cls, path: Path, config: PipelineConfig) -> List[TranscriptRecord]:
        records: List[TranscriptRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        for entry in data:
            record = cls._record_from_entry(entry, config)
            records.append(record)
        return records

    @classmethod
    def _load_csv(cls, path: Path, config: PipelineConfig) -> List[TranscriptRecord]:
        records: List[TranscriptRecord] = []
        with path.open("r", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            for row in reader:
                record = cls._record_from_entry(row, config)
                records.append(record)
        return records

    @classmethod
    def _record_from_entry(cls, entry: Dict[str, str], config: PipelineConfig) -> TranscriptRecord:
        overrides = config.metadata_overrides.get(entry.get("event_id", ""), {})
        merged = {**entry, **overrides}
        pdf_path = config.resolve_path(config.pdf_directory / merged["pdf_file"])
        if not pdf_path.exists():
            raise FileNotFoundError(f"PDF file missing for event {merged['event_id']}: {pdf_path}")
        event_date = cls._parse_date(merged["event_date"])
        speakers = cls._normalize_speakers(merged.get("speakers", ""))
        record = TranscriptRecord(
            event_id=str(merged["event_id"]),
            event_type=str(merged.get("event_type", "unknown")),
            event_date=event_date,
            speakers=speakers,
            pdf_path=pdf_path,
            source_url=merged.get("source_url"),
            provenance=merged.get("provenance"),
            extra={
                key: value
                for key, value in merged.items()
                if key
                not in {
                    "event_id",
                    "event_type",
                    "event_date",
                    "speakers",
                    "pdf_file",
                    "source_url",
                    "provenance",
                }
            },
        )
        return record
