"""Experiment tracking utilities."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Optional

import json

try:
    import mlflow  # type: ignore
except ModuleNotFoundError:  # pragma: no cover - optional dependency
    mlflow = None  # type: ignore

from .config import ExperimentConfig
from .evaluation import ExperimentResult, FoldResult


class ExperimentTracker:
    """Persist configurations, metrics, and artifacts."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.run_id: Optional[str] = None
        self.output_dir = config.tracking.output_directory
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._init_mlflow()
        self._persist_config()

    def _init_mlflow(self) -> None:
        if not self.config.tracking.use_mlflow:
            return
        if mlflow is None:
            raise ImportError("MLflow requested but not installed")
        tracking_uri = self.config.tracking.mlflow_tracking_uri
        if tracking_uri:
            mlflow.set_tracking_uri(tracking_uri)
        mlflow.set_experiment(self.config.tracking.experiment_name)
        run = mlflow.start_run()
        self.run_id = run.info.run_id
        mlflow.log_dict(self.config.to_dict(), "config.yaml")

    def _persist_config(self) -> None:
        config_path = self.output_dir / "config.yaml"
        self.config.dump(config_path)

    def log_fold(self, fold: FoldResult) -> None:
        if self.run_id and mlflow is not None:
            mlflow.log_metrics({f"{fold.fold}/{k}": v for k, v in fold.metrics.items()})
        fold_path = self.output_dir / f"fold_{fold.fold}_metrics.json"
        fold_path.write_text(json.dumps(fold.metrics, indent=2), encoding="utf-8")

    def log_result(self, result: ExperimentResult) -> None:
        metrics_path = self.output_dir / "aggregate_metrics.json"
        metrics_path.write_text(json.dumps(result.metrics, indent=2), encoding="utf-8")
        calibration_path = self.output_dir / "calibration.json"
        calibration_path.write_text(json.dumps(result.calibration, indent=2), encoding="utf-8")
        feature_path = self.output_dir / "feature_importances.json"
        feature_path.write_text(
            json.dumps(result.feature_importances, indent=2), encoding="utf-8"
        )
        if self.run_id and mlflow is not None:
            mlflow.log_metrics(result.metrics)
            mlflow.log_dict(result.calibration, "calibration.json")
            mlflow.log_dict(result.feature_importances, "feature_importances.json")
            mlflow.end_run()

    def log_artifact(self, name: str, path: Path) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        artifact_path = self.output_dir / f"{timestamp}_{name}{path.suffix}"
        artifact_path.write_bytes(path.read_bytes())
        if self.run_id and mlflow is not None:
            mlflow.log_artifact(str(path))
