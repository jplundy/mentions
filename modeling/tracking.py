"""Experiment tracking utilities."""

from __future__ import annotations

import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional

import json

LOGGER = logging.getLogger(__name__)

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
        self._warn_on_degenerate_metrics(result.metrics)

    def _warn_on_degenerate_metrics(self, agg_metrics: Dict[str, float]) -> None:
        """Emit actionable warnings when aggregate metrics signal a broken model."""

        exp_name = self.output_dir.name
        f1 = agg_metrics.get("f1")
        roc_auc = agg_metrics.get("roc_auc")

        if f1 is not None and f1 == 0.0:
            LOGGER.warning(
                "[%s] DEGENERATE MODEL: aggregate F1=0.0. The model predicts only the "
                "majority class. Likely causes: severe class imbalance with a hard 0.5 "
                "threshold, or all validation folds contain only one class. "
                "Fix: tune the decision threshold, use class_weight='balanced', or "
                "ensure each validation fold contains positive examples.",
                exp_name,
            )

        if roc_auc is not None and math.isnan(roc_auc):
            LOGGER.warning(
                "[%s] ROC AUC is NaN: one or more validation folds contained only a "
                "single class. Consider increasing minimum_train_events or switching "
                "to a validation strategy that guarantees both classes per fold.",
                exp_name,
            )

    def log_artifact(self, name: str, path: Path) -> None:
        timestamp = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")
        artifact_path = self.output_dir / f"{timestamp}_{name}{path.suffix}"
        artifact_path.write_bytes(path.read_bytes())
        if self.run_id and mlflow is not None:
            mlflow.log_artifact(str(path))

    def log_news_metadata(
        self,
        *,
        snapshot_hash: Optional[str],
        provenance: Dict[str, list],
    ) -> None:
        if snapshot_hash:
            hash_path = self.output_dir / "news_snapshot_hash.txt"
            hash_path.write_text(snapshot_hash, encoding="utf-8")
            if self.run_id and mlflow is not None:
                mlflow.log_param("news_snapshot_hash", snapshot_hash)
        if provenance:
            provenance_path = self.output_dir / "news_provenance.json"
            provenance_path.write_text(json.dumps(provenance, indent=2), encoding="utf-8")
            if self.run_id and mlflow is not None:
                mlflow.log_dict(provenance, "news_provenance.json")
