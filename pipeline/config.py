"""Configuration utilities for the transcript data pipeline."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import yaml


@dataclass
class SegmentationConfig:
    """Configuration for transcript segmentation."""

    mode: str = "speaker_turn"
    window_size: Optional[int] = None
    stride: Optional[int] = None

    def validate(self) -> None:
        supported_modes = {"speaker_turn", "fixed_window"}
        if self.mode not in supported_modes:
            raise ValueError(f"Unsupported segmentation mode: {self.mode}")
        if self.mode == "fixed_window":
            if not self.window_size or self.window_size <= 0:
                raise ValueError("window_size must be a positive integer for fixed_window segmentation")
            if self.stride is not None and self.stride <= 0:
                raise ValueError("stride must be positive when provided")


@dataclass
class PipelineConfig:
    """Top-level configuration for building the historical transcript dataset."""

    base_dir: Path
    inventory_path: Path
    pdf_directory: Path
    output_directory: Path
    target_words: List[str] = field(default_factory=list)
    segmentation: SegmentationConfig = field(default_factory=SegmentationConfig)
    metadata_overrides: Dict[str, Dict[str, str]] = field(default_factory=dict)

    @classmethod
    def from_file(cls, path: Path | str) -> "PipelineConfig":
        path = Path(path)
        with path.open("r", encoding="utf-8") as handle:
            if path.suffix in {".yml", ".yaml"}:
                raw = yaml.safe_load(handle)
            elif path.suffix == ".json":
                raw = json.load(handle)
            else:
                raise ValueError(f"Unsupported config format: {path.suffix}")

        base_dir = path.parent.resolve()
        segmentation_cfg = SegmentationConfig(**raw.get("segmentation", {}))
        segmentation_cfg.validate()
        config = cls(
            base_dir=base_dir,
            inventory_path=(base_dir / raw["inventory_path"]).expanduser().resolve(),
            pdf_directory=(base_dir / raw["pdf_directory"]).expanduser().resolve(),
            output_directory=(base_dir / raw["output_directory"]).expanduser().resolve(),
            target_words=list(raw.get("target_words", [])),
            segmentation=segmentation_cfg,
            metadata_overrides=raw.get("metadata_overrides", {}),
        )
        return config

    def ensure_directories(self) -> None:
        self.pdf_directory.mkdir(parents=True, exist_ok=True)
        self.output_directory.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: Path | str) -> Path:
        return (self.base_dir / Path(path)).expanduser().resolve()

    def iter_target_words(self) -> Iterable[str]:
        return (word.lower().strip() for word in self.target_words if word)
