"""Speaker targeting utilities for transcript processing."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List, Optional, Pattern

_SEPARATOR_PATTERN = r"(?:\s+|[-‐‑–—]\s*)"


def _compile_speaker_pattern(term: str) -> Pattern[str]:
    """Compile a case-insensitive pattern used for speaker matching."""

    if term is None:
        raise ValueError("Speaker filter terms must be strings")
    stripped = term.strip()
    if not stripped:
        raise ValueError("Speaker filter terms cannot be empty")

    if stripped.lower().startswith("regex:"):
        pattern_body = stripped[6:].strip()
        if not pattern_body:
            raise ValueError("Regex-based speaker filter terms cannot be empty")
        return re.compile(pattern_body, re.IGNORECASE)

    tokens = [re.escape(token) for token in re.split(r"\s+", stripped) if token]
    if not tokens:
        raise ValueError("Speaker filter terms must contain alphanumeric characters")
    phrase_pattern = _SEPARATOR_PATTERN.join(tokens)
    pattern = rf"\b{phrase_pattern}\b"
    return re.compile(pattern, re.IGNORECASE)


@dataclass
class SpeakerFilter:
    """Filter that determines whether a segment speaker should be retained."""

    include: List[Pattern[str]]
    exclude: List[Pattern[str]]

    @classmethod
    def from_terms(
        cls,
        include: Optional[Iterable[str]] = None,
        exclude: Optional[Iterable[str]] = None,
    ) -> "SpeakerFilter":
        include_patterns = [
            _compile_speaker_pattern(term)
            for term in (include or [])
            if term and term.strip()
        ]
        exclude_patterns = [
            _compile_speaker_pattern(term)
            for term in (exclude or [])
            if term and term.strip()
        ]
        return cls(include_patterns, exclude_patterns)

    def allows(self, speaker: Optional[str]) -> bool:
        """Return ``True`` if the speaker should be included."""

        speaker_text = (speaker or "").strip()
        if not speaker_text:
            # When includes are specified we require an explicit speaker label.
            return not self.include

        if any(pattern.search(speaker_text) for pattern in self.exclude):
            return False
        if not self.include:
            return True
        return any(pattern.search(speaker_text) for pattern in self.include)
