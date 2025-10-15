"""Data pipeline package for transcript ingestion and preprocessing."""

from __future__ import annotations

import importlib
from typing import Any, Dict, Tuple

_EXPORTS: Dict[str, Tuple[str, str]] = {
    "PipelineConfig": ("config", "PipelineConfig"),
    "SpeakerFilterConfig": ("config", "SpeakerFilterConfig"),
    "DatasetPublisher": ("dataset", "DatasetPublisher"),
    "PDFIngestor": ("ingestion", "PDFIngestor"),
    "TranscriptInventory": ("inventory", "TranscriptInventory"),
    "TranscriptPreprocessor": ("preprocessing", "TranscriptPreprocessor"),
    "Segment": ("segmentation", "Segment"),
    "Segmenter": ("segmentation", "Segmenter"),
    "SpeakerFilter": ("speakers", "SpeakerFilter"),
}

__all__ = list(_EXPORTS)


def __getattr__(name: str) -> Any:
    if name not in _EXPORTS:
        raise AttributeError(f"module 'pipeline' has no attribute {name!r}")
    module_name, attribute = _EXPORTS[name]
    module = importlib.import_module(f".{module_name}", __name__)
    value = getattr(module, attribute)
    globals()[name] = value
    return value
