"""CLI entrypoint for baseline modeling experiments."""
from __future__ import annotations

import argparse
import dataclasses
import json
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import pandas as pd
import yaml

from model import (
    ExperimentConfig,
    ExperimentPaths,
    ModelConfig,
    ModelResult,
    TabularFeatureConfig,
    TextFeatureConfig,
    EmbeddingFeatureConfig,
    CalibrationConfig,
    ValidationConfig,
    run_experiment,
    save_model_artifacts,
)


def _instantiate_dataclass(cls, data: Optional[Dict[str, Any]]):
    if data is None:
        return cls()
    return cls(**data)


def load_experiment_config(path: Path) -> ExperimentConfig:
    with path.open("r") as handle:
        raw = yaml.safe_load(handle)

    if "target_column" not in raw:
        raise ValueError("Config must specify 'target_column'.")
    if "text" not in raw:
        raise ValueError("Config must include a 'text' section.")
    if "models" not in raw:
        raise ValueError("Config must include at least one model definition.")

    text_cfg = TextFeatureConfig(**raw["text"])
    tabular_cfg = _instantiate_dataclass(TabularFeatureConfig, raw.get("tabular"))
    embedding_cfg = _instantiate_dataclass(EmbeddingFeatureConfig, raw.get("embedding"))
    calibration_cfg = (
        CalibrationConfig(**raw["calibration"]) if raw.get("calibration") else None
    )
    validation_cfg = _instantiate_dataclass(ValidationConfig, raw.get("validation"))

    output_data = raw.get("output") or {}
    if "output_dir" in output_data:
        output_data = dict(output_data)
        output_data["output_dir"] = Path(output_data["output_dir"])
    output_cfg = ExperimentPaths(**output_data)

    models_cfg = [ModelConfig(**model_dict) for model_dict in raw["models"]]

    return ExperimentConfig(
        target_column=raw["target_column"],
        text=text_cfg,
        tabular=tabular_cfg,
        embedding=embedding_cfg,
        calibration=calibration_cfg,
        validation=validation_cfg,
        models=models_cfg,
        dataset_path=raw.get("dataset_path"),
        output=output_cfg,
    )


class ExperimentTracker:
    """Minimal tracker that organises run metadata and outputs."""

    def __init__(self, base_dir: Path, run_name: Optional[str] = None):
        timestamp = datetime.utcnow().strftime("%Y%m%d-%H%M%S")
        sanitized = run_name or "baseline"
        sanitized = re.sub(r"[^A-Za-z0-9_-]+", "_", sanitized)
        self.run_id = f"{timestamp}_{sanitized}"
        self.base_dir = base_dir
        self.run_dir = base_dir / self.run_id
        self.run_dir.mkdir(parents=True, exist_ok=True)

    def log_config(self, config: ExperimentConfig, filename: str = "config.json") -> None:
        payload = dataclasses.asdict(config)
        (self.run_dir / filename).write_text(json.dumps(payload, indent=2, default=str))

    def log_dataset_profile(self, df: pd.DataFrame, filename: str = "dataset_profile.json") -> None:
        profile = {
            "num_rows": int(df.shape[0]),
            "num_columns": int(df.shape[1]),
            "columns": {
                column: {
                    "dtype": str(df[column].dtype),
                    "num_missing": int(df[column].isna().sum()),
                }
                for column in df.columns
            },
        }
        (self.run_dir / filename).write_text(json.dumps(profile, indent=2))

    def log_summary(self, result: ModelResult, filename: Optional[str] = None) -> None:
        payload = {
            "model": result.model_name,
            "aggregate_metrics": result.aggregate,
            "holdout_metrics": result.holdout_metrics,
        }
        target = filename or f"{result.model_name}_summary.json"
        (self.run_dir / target).write_text(json.dumps(payload, indent=2))


def load_dataset(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found at {path}")
    suffix = path.suffix.lower()
    if suffix == ".csv":
        return pd.read_csv(path)
    if suffix == ".parquet":
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def filter_models(models: List[ModelConfig], include: Optional[Iterable[str]]) -> List[ModelConfig]:
    if not include:
        return models
    include_set = {name.lower() for name in include}
    selected = [model for model in models if model.name.lower() in include_set]
    if not selected:
        raise ValueError(f"No models matched requested filters: {sorted(include_set)}")
    return selected


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run baseline modeling experiments.")
    parser.add_argument("--config", required=True, help="Path to experiment YAML file.")
    parser.add_argument("--dataset", help="Override dataset path defined in the config.")
    parser.add_argument("--output-dir", help="Directory for storing artifacts (defaults to config value).")
    parser.add_argument("--run-name", help="Optional run name override for artifact directory.")
    parser.add_argument("--model", action="append", help="Name(s) of specific models to run.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate inputs without training models.",
    )
    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    config_path = Path(args.config)
    config = load_experiment_config(config_path)

    if args.dataset:
        config.dataset_path = args.dataset
    if not config.dataset_path:
        raise ValueError("Dataset path must be provided via config or --dataset.")

    dataset_path = Path(config.dataset_path)
    df = load_dataset(dataset_path)

    if args.output_dir:
        config.output.output_dir = Path(args.output_dir)
    config.output.output_dir.mkdir(parents=True, exist_ok=True)

    tracker = ExperimentTracker(config.output.output_dir, args.run_name)
    tracker.log_config(config)
    tracker.log_dataset_profile(df)

    if args.dry_run:
        print("Dry run completed. Artifacts staged at", tracker.run_dir)
        return

    selected_models = filter_models(config.models, args.model)

    for model_cfg in selected_models:
        print(f"Training model: {model_cfg.name}")
        result = run_experiment(df, config, model_cfg)
        model_dir = tracker.run_dir / model_cfg.name
        save_model_artifacts(result, config, model_dir)
        tracker.log_summary(result, filename=f"{model_cfg.name}_summary.json")
        print(f"Completed model: {model_cfg.name}. Artifacts saved to {model_dir}")

    print("Run complete. Artifacts at", tracker.run_dir)


if __name__ == "__main__":
    main()
