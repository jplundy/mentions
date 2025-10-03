"""Text preprocessing utilities for transcript data."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Iterable, List

HEADER_PATTERNS = [
    re.compile(r"^\s*page\s+\d+", re.IGNORECASE),
    re.compile(r"^\s*federal reserve", re.IGNORECASE),
    re.compile(r"^\s*press\s+conference", re.IGNORECASE),
]

FOOTER_PATTERNS = [
    re.compile(r"^\s*\d+\s*/\s*\d+\s*$"),
    re.compile(r"^\s*\*+\s*$"),
]

SPEAKER_LABEL_PATTERN = re.compile(r"^(?P<label>[A-Z][A-Z\.\s\-']{1,40}):")


@dataclass
class TranscriptPreprocessor:
    """Apply cleaning heuristics to raw transcript text."""

    header_patterns: Iterable[re.Pattern[str]] = tuple(HEADER_PATTERNS)
    footer_patterns: Iterable[re.Pattern[str]] = tuple(FOOTER_PATTERNS)

    def clean_pages(self, pages: Iterable[str]) -> List[str]:
        cleaned_pages = []
        for page_text in pages:
            lines = self._split_lines(page_text)
            lines = self._remove_headers(lines)
            lines = self._remove_footers(lines)
            normalized_lines = [self._normalize_line(line) for line in lines if line.strip()]
            cleaned_pages.append("\n".join(normalized_lines))
        return cleaned_pages

    def _split_lines(self, text: str) -> List[str]:
        text = text.replace("\u2013", "-").replace("\u2014", "-")
        text = text.replace("\u2019", "'").replace("\u2018", "'")
        text = text.replace("\u201c", '"').replace("\u201d", '"')
        return text.splitlines()

    def _remove_headers(self, lines: List[str]) -> List[str]:
        while lines and any(pattern.search(lines[0]) for pattern in self.header_patterns):
            lines = lines[1:]
        return lines

    def _remove_footers(self, lines: List[str]) -> List[str]:
        while lines and any(pattern.search(lines[-1]) for pattern in self.footer_patterns):
            lines = lines[:-1]
        return lines

    def _normalize_line(self, line: str) -> str:
        line = re.sub(r"\s+", " ", line.strip())
        match = SPEAKER_LABEL_PATTERN.match(line)
        if match:
            label = match.group("label").upper().replace(".", "").replace("  ", " ")
            remainder = line[match.end():].lstrip()
            return f"{label}: {remainder}" if remainder else f"{label}:"
        return line

    def merge_pages(self, pages: Iterable[str]) -> str:
        return "\n\n".join(page for page in pages if page.strip())

    def preprocess(self, pages: Iterable[str]) -> str:
        cleaned_pages = self.clean_pages(pages)
        merged = self.merge_pages(cleaned_pages)
        return merged
