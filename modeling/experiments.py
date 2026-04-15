"""Baseline experiment orchestration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import joblib
import pandas as pd
from sklearn.pipeline import Pipeline

from .calibration import CalibrationResult, calibrate_model
from .config import ExperimentConfig
from .dataset import DatasetBundle, DatasetLoader
from .evaluation import (
    ExperimentResult,
    FoldResult,
    aggregate_optimal_threshold,
    compute_classification_metrics,
    extract_feature_importances,
    optimize_threshold,
    summarize_metrics,
)
from .models import build_estimator
from .pipeline import build_training_pipeline
from .tracking import ExperimentTracker
from .validation import SplitIndices, build_validator


@dataclass
class FoldArtifacts:
    """Artifacts produced for each validation fold."""

    pipeline: Pipeline
    metrics: FoldResult


class BaselineExperiment:
    """Coordinate feature construction, model training, and evaluation."""

    def __init__(self, config: ExperimentConfig) -> None:
        self.config = config
        self.tracker = ExperimentTracker(config)
        self.fold_artifacts: List[FoldArtifacts] = []
        self.calibration_result: Optional[CalibrationResult] = None
        self.final_pipeline: Optional[Pipeline] = None
        self.dataset_bundle: Optional[DatasetBundle] = None

    def load_dataset(self) -> pd.DataFrame:
        """Load the dataset referenced by the configuration."""

        loader = DatasetLoader(self.config)
        bundle = loader.load()
        self.dataset_bundle = bundle
        if bundle.news:
            self.tracker.log_news_metadata(
                snapshot_hash=bundle.news.snapshot_hash,
                provenance=bundle.news.provenance_by_event,
            )
        return bundle.frame

    def run(self) -> ExperimentResult:
        """Execute the configured experiment."""

        df = self.load_dataset()
        X, y = self._prepare_features_and_target(df)
        validator = build_validator(self.config.validation)

        fold_results: List[FoldResult] = []
        for fold_idx, split in enumerate(validator.split(df)):
            metrics = self._run_fold(fold_idx, split, X, y)
            fold_results.append(metrics)

        aggregate_metrics = summarize_metrics(fold_results)
        optimal_threshold = aggregate_optimal_threshold(fold_results)
        calibration_summary = {}
        feature_importances: Dict[str, float] = {}

        self._train_final_pipeline(X, y, optimal_threshold=optimal_threshold)

        if self.final_pipeline is not None and self.calibration_result is not None:
            calibration_summary = {
                "method": self.calibration_result.method,
                "brier": self.calibration_result.brier,
                "log_loss": self.calibration_result.log_loss,
            }
            feature_importances = self._extract_feature_importances(self.final_pipeline)

        result = ExperimentResult(
            metrics=aggregate_metrics,
            folds=fold_results,
            calibration=calibration_summary,
            feature_importances=feature_importances,
            optimal_threshold=optimal_threshold,
        )
        self.tracker.log_result(result)
        return result

    def _run_fold(
        self,
        fold_idx: int,
        split: SplitIndices,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> FoldResult:
        """Train and evaluate a single validation fold."""

        estimator = build_estimator(self.config.model)
        pipeline = build_training_pipeline(self.config.feature, estimator)
        X_train = X.iloc[split.train]
        y_train = y.iloc[split.train]
        X_valid = X.iloc[split.validate]
        y_valid = y.iloc[split.validate]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_valid)[:, 1]

        has_positives = int(y_valid.sum()) > 0
        optimal_threshold, _ = optimize_threshold(y_valid, y_prob)
        fold_metrics = compute_classification_metrics(
            y_valid, y_prob, threshold=optimal_threshold
        )
        fold_result = FoldResult(
            fold=fold_idx,
            metrics=fold_metrics,
            optimal_threshold=optimal_threshold,
            has_positives=has_positives,
        )
        self.fold_artifacts.append(FoldArtifacts(pipeline=pipeline, metrics=fold_result))

        self.tracker.log_fold(fold_result)
        return fold_result

    def _prepare_features_and_target(
        self, df: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Filter invalid targets and return aligned features and labels."""

        target = df[self.config.target_column]
        missing_mask = target.isna()
        if missing_mask.any():
            df = df.loc[~missing_mask].copy()
            target = df[self.config.target_column]
        if df.empty:
            raise ValueError(
                "No rows with a valid target value remain after filtering missing targets."
            )
        return df, target.astype(int)

    def _train_final_pipeline(
        self, X: pd.DataFrame, y: pd.Series, optimal_threshold: float = 0.5
    ) -> None:
        """Train the final calibrated pipeline on the full dataset."""

        estimator = build_estimator(self.config.model)
        pipeline = build_training_pipeline(self.config.feature, estimator)
        pipeline.fit(X, y)
        features = pipeline.named_steps["features"].transform(X)
        calibrated_model, calibration_result = calibrate_model(
            pipeline.named_steps["model"], features, y, self.config.calibration
        )

        pipeline.named_steps["model"] = calibrated_model
        self.final_pipeline = pipeline
        self.calibration_result = calibration_result
        self._persist_artifacts(optimal_threshold=optimal_threshold)

    def _persist_artifacts(self, optimal_threshold: float = 0.5) -> None:
        """Persist trained pipeline, threshold, and vectorizers for downstream teams."""

        if self.final_pipeline is None:
            return

        output_dir = self.config.tracking.output_directory
        output_dir.mkdir(parents=True, exist_ok=True)
        model_path = output_dir / "baseline_pipeline.joblib"
        joblib.dump(self.final_pipeline, model_path)
        self.tracker.log_artifact("pipeline", model_path)
        self.tracker.log_threshold(optimal_threshold)

    def _extract_feature_importances(self, pipeline: Pipeline) -> Dict[str, float]:
        """Extract feature names from the trained pipeline and compute importances."""

        feature_transformer = pipeline.named_steps.get("features")
        if feature_transformer is None:
            return {}

        try:
            feature_names = list(feature_transformer.get_feature_names_out())
        except AttributeError:
            feature_names = self._fallback_feature_names(feature_transformer)
        return extract_feature_importances(pipeline, feature_names)

    def _fallback_feature_names(self, transformer) -> List[str]:
        names: List[str] = []
        transformers = getattr(transformer, "transformers_", [])
        for alias, sub_transformer, columns in transformers:
            if sub_transformer == "drop":
                continue
            try:
                sub_names = list(sub_transformer.get_feature_names_out())
            except AttributeError:
                if isinstance(columns, (list, tuple)):
                    sub_names = [f"{alias}__{col}" for col in columns]
                else:
                    sub_names = [f"{alias}__{columns}"]
            names.extend(sub_names)
        return names
