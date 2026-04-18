"""Dataset publishing utilities for segmented transcripts."""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Pattern, Sequence, Set

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
from .speakers import SpeakerFilter

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
    speaker_filter: Optional[SpeakerFilter] = None

    def __post_init__(self) -> None:
        # Normalize the target words to a list to avoid exhausting generators.
        self.target_words = [word for word in self.target_words]
        mapping = {}
        seen_columns = set()
        patterns = {}
        prefilters: Dict[str, Set[str]] = {}
        for word in self.target_words:
            column = self._target_column_name(word)
            if column in seen_columns:
                raise ValueError(
                    f"Duplicate target column {column!r} generated from target words"
                )
            mapping[word] = column
            seen_columns.add(column)
            patterns[word] = _compile_target_pattern(word)
            # Build a set of first-token substrings (one per slash-alternative).
            # The regex always includes the base word as a variant, so if no first
            # token appears in the lowercased text the full regex cannot match.
            first_tokens: Set[str] = set()
            for alt in _split_alternatives(word):
                tokens = _tokenize_phrase(alt)
                if tokens:
                    first_tokens.add(tokens[0].lower())
            prefilters[word] = first_tokens
        self._target_mapping = mapping
        self._target_patterns = patterns
        self._target_prefilters = prefilters

    def _segment_rows(
        self, record: TranscriptRecord, segments: Iterable[Segment]
    ) -> List[Dict[str, object]]:
        rows: List[Dict[str, object]] = []
        target_pairs = [(word, word.lower()) for word in self.target_words]
        for segment in segments:
            if self.speaker_filter and not self.speaker_filter.allows(segment.speaker):
                continue
            text_lower = segment.text.lower()
            target_hits = {}
            for original, lower in target_pairs:
                prefilter = self._target_prefilters.get(original, set())
                if prefilter and not any(ft in text_lower for ft in prefilter):
                    target_hits[self._target_mapping[original]] = False
                    continue
                pattern = self._target_patterns.get(original)
                if pattern is not None:
                    matched = bool(pattern.search(segment.text))
                else:
                    matched = lower in text_lower
                target_hits[self._target_mapping[original]] = matched
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
APOSTROPHES = ("'", "’")


def _compile_target_pattern(term: str) -> Optional[Pattern[str]]:
    """Compile a regular expression that matches the target term.

    The generated pattern adheres to the contract rules for payout criteria:

    * Includes plural and possessive forms (e.g., "immigrant" -> "immigrants", "immigrant's").
    * Excludes tense inflections by only modelling plural/possessive endings.
    * Allows hyphenated or compound usages while preventing closed compounds that subsume the
      target (e.g., "fire station" counts but "firetruck" does not).
    * Handles ordinals for numeric tokens (e.g., "January 6" -> "January 6th").
    * Supports slash-delimited alternatives (e.g., "Elon / Musk" -> matches either "Elon" or
      "Musk").
    """

    alternatives = _split_alternatives(term)
    patterns: List[str] = []
    for alternative in alternatives:
        tokens = _tokenize_phrase(alternative)
        if not tokens:
            continue
        token_patterns = [_token_pattern(token) for token in tokens]
        if not all(token_patterns):
            continue
        phrase_pattern = token_patterns[0]
        for token_pattern in token_patterns[1:]:
            phrase_pattern += _separator_pattern()
            phrase_pattern += token_pattern
        patterns.append(f"(?<!\\w){phrase_pattern}(?!\\w)")

    if not patterns:
        stripped = term.strip()
        if not stripped:
            return None
        # Fallback to a simple substring search as a last resort.
        escaped = re.escape(stripped)
        return re.compile(escaped, re.IGNORECASE)

    combined = "|".join(patterns)
    return re.compile(combined, re.IGNORECASE)


def _split_alternatives(term: str) -> Sequence[str]:
    if not term:
        return []
    parts = [part.strip() for part in re.split(r"\s*/\s*", term) if part.strip()]
    return parts or [term.strip()]


def _tokenize_phrase(phrase: str) -> Sequence[str]:
    return [token for token in phrase.split() if token]


def _separator_pattern() -> str:
    # Allow whitespace, hyphenated, or dash-separated compounds between tokens.
    return r"(?:\s+|\s*[-‐‑–—]\s*)"


def _token_pattern(token: str) -> str:
    token = token.strip()
    if not token:
        return ""
    if re.fullmatch(r"\d+", token):
        return rf"{re.escape(token)}(?:st|nd|rd|th)?"
    if re.fullmatch(r"[A-Za-z]+", token):
        variants = _word_variants(token)
        escaped = sorted({re.escape(variant) for variant in variants}, key=len, reverse=True)
        return "(?:" + "|".join(escaped) + ")"
    return re.escape(token)


def _word_variants(word: str) -> Set[str]:
    lowercase = word.lower()
    singular_forms = {lowercase}
    plural_forms = _plural_forms(lowercase)
    forms = set(singular_forms)
    forms.update(plural_forms)

    for apostrophe in APOSTROPHES:
        for singular in singular_forms:
            forms.add(singular + apostrophe + "s")
            if singular.endswith("s"):
                forms.add(singular + apostrophe)
        for plural in plural_forms:
            if plural.endswith("s"):
                forms.add(plural + apostrophe)

    return forms


def _plural_forms(word: str) -> Set[str]:
    forms: Set[str] = set()
    if not word:
        return forms

    forms.add(word + "s")

    if re.search(r"(s|x|z|ch|sh)$", word):
        forms.add(word + "es")
    if re.search(r"[^aeiou]y$", word):
        forms.add(word[:-1] + "ies")
    if word.endswith("f"):
        forms.add(word[:-1] + "ves")
        forms.add(word + "s")
    if word.endswith("fe"):
        forms.add(word[:-2] + "ves")
    if word.endswith("us"):
        forms.add(word[:-2] + "i")
    if word.endswith("is"):
        forms.add(word[:-2] + "es")
    if word.endswith("o"):
        forms.add(word + "es")

    return forms
