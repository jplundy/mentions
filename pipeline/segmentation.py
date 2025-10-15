"""Segmentation logic for transcript text."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from typing import Iterable, Iterator, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:  # pragma: no cover - for type checking only
    from .config import SegmentationConfig


@dataclass
class Segment:
    """A contiguous chunk of transcript text."""

    event_id: str
    segment_id: str
    text: str
    speaker: Optional[str]
    start_char: int
    end_char: int


class Segmenter:
    """Create transcript segments according to configurable rules."""

    def __init__(self, config: "SegmentationConfig"):
        self.config = config

    def segment(self, event_id: str, text: str) -> List[Segment]:
        if self.config.mode == "speaker_turn":
            return list(self._segment_by_speaker(event_id, text))
        if self.config.mode == "fixed_window":
            return list(self._segment_by_window(event_id, text))
        raise ValueError(f"Unsupported segmentation mode: {self.config.mode}")

    def _segment_by_speaker(self, event_id: str, text: str) -> Iterator[Segment]:
        speaker_regex = re.compile(r"^(?P<speaker>[A-Z][A-Z\s\-']{1,40}):", re.MULTILINE)
        segments: List[Segment] = []
        last_end = 0
        segment_counter = itertools.count(1)
        matches = list(speaker_regex.finditer(text))
        if not matches:
            yield from self._segment_by_window(event_id, text)
            return
        for idx, match in enumerate(matches):
            speaker = match.group("speaker").strip()
            start = match.end()
            end = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
            raw_segment = text[start:end]
            content = raw_segment.strip()
            if not content:
                continue
            leading_ws = len(raw_segment) - len(raw_segment.lstrip())
            trailing_ws = len(raw_segment) - len(raw_segment.rstrip())
            content_start = start + leading_ws
            content_end = end - trailing_ws
            segment_id = f"{event_id}_S{next(segment_counter):04d}"
            segments.append(
                Segment(
                    event_id=event_id,
                    segment_id=segment_id,
                    text=content,
                    speaker=speaker,
                    start_char=content_start,
                    end_char=content_end,
                )
            )
            last_end = end
        if last_end < len(text):
            raw_tail = text[last_end:]
            tail = raw_tail.strip()
            if tail:
                leading_ws = len(raw_tail) - len(raw_tail.lstrip())
                trailing_ws = len(raw_tail) - len(raw_tail.rstrip())
                segment_id = f"{event_id}_S{next(segment_counter):04d}"
                segments.append(
                    Segment(
                        event_id=event_id,
                        segment_id=segment_id,
                        text=tail,
                        speaker=None,
                        start_char=last_end + leading_ws,
                        end_char=len(text) - trailing_ws,
                    )
                )
        yield from segments

    def _segment_by_window(self, event_id: str, text: str) -> Iterator[Segment]:
        window = self.config.window_size or 200
        stride = self.config.stride or window
        tokens = text.split()
        if not tokens:
            return
        segment_counter = itertools.count(1)
        for start_idx in range(0, len(tokens), stride):
            end_idx = min(start_idx + window, len(tokens))
            chunk_tokens = tokens[start_idx:end_idx]
            content = " ".join(chunk_tokens).strip()
            if not content:
                continue
            segment_id = f"{event_id}_W{next(segment_counter):04d}"
            # Convert token indices back to character indices approximately
            char_start = len(" ".join(tokens[:start_idx]))
            char_end = char_start + len(content)
            yield Segment(
                event_id=event_id,
                segment_id=segment_id,
                text=content,
                speaker=None,
                start_char=char_start,
                end_char=char_end,
            )
