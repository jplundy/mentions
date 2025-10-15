"""Configuration helpers for news ingestion."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import yaml


@dataclass
class NewsSourceConfig:
    """Configuration describing a single upstream news source."""

    name: str
    loader: str
    schedule: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)
    enabled: bool = True

    def is_enabled(self) -> bool:
        return bool(self.enabled)


@dataclass
class NewsIngestConfig:
    """Top-level configuration for the news ingestion pipeline."""

    base_dir: Path
    raw_directory: Path
    manifest_directory: Path
    sources: List[NewsSourceConfig] = field(default_factory=list)

    @classmethod
    def from_file(cls, path: Path | str) -> "NewsIngestConfig":
        config_path = Path(path).resolve()
        with config_path.open("r", encoding="utf-8") as handle:
            if config_path.suffix in {".yml", ".yaml"}:
                data = yaml.safe_load(handle)
            elif config_path.suffix == ".json":
                data = json.load(handle)
            else:
                raise ValueError(f"Unsupported config format: {config_path.suffix}")

        base_dir = config_path.parent
        raw_dir = cls._resolve_path(base_dir, data.get("raw_directory"))
        manifest_dir = cls._resolve_path(base_dir, data.get("manifest_directory"))
        sources = [
            NewsSourceConfig(
                name=item["name"],
                loader=item["loader"],
                schedule=item.get("schedule"),
                params=item.get("params", {}),
                enabled=item.get("enabled", True),
            )
            for item in data.get("sources", [])
        ]
        return cls(
            base_dir=base_dir,
            raw_directory=raw_dir,
            manifest_directory=manifest_dir,
            sources=sources,
        )

    @staticmethod
    def _resolve_path(base: Path, value: Optional[str]) -> Path:
        if not value:
            raise ValueError("Both raw_directory and manifest_directory must be configured")
        return (base / Path(value)).expanduser().resolve()

    def ensure_directories(self) -> None:
        self.raw_directory.mkdir(parents=True, exist_ok=True)
        self.manifest_directory.mkdir(parents=True, exist_ok=True)

    def iter_sources(self, *, include_disabled: bool = False) -> Iterable[NewsSourceConfig]:
        for source in self.sources:
            if include_disabled or source.is_enabled():
                yield source
