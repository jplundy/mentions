"""Baseline modeling utilities for the mentions project."""
from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from joblib import dump
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    brier_score_loss,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

try:  # pragma: no cover - optional dependency
    from sentence_transformers import SentenceTransformer
except Exception:  # pragma: no cover - optional dependency
    SentenceTransformer = None  # type: ignore[misc]


@dataclass
class TextFeatureConfig:
    column: str
    max_features: int = 20000
    ngram_range: Tuple[int, int] = (1, 2)
    min_df: int | float = 2
    max_df: int | float = 0.95
    stop_words: Optional[str] = "english"
    lowercase: bool = True


@dataclass
class TabularFeatureConfig:
    categorical: List[str] = field(default_factory=list)
    numeric: List[str] = field(default_factory=list)
    categorical_impute: str = "most_frequent"
    numeric_impute: str = "median"


@dataclass
class EmbeddingFeatureConfig:
    enabled: bool = False
    column: Optional[str] = None
    model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    batch_size: int = 32
    device: Optional[str] = None


@dataclass
class CalibrationConfig:
    method: str = "isotonic"  # or "sigmoid"
    cv: int = 3


@dataclass
class ValidationConfig:
    strategy: str = "rolling"  # or "leave_one_event_out"
    time_column: str = "timestamp"
    min_train_size: int = 100
    test_window_size: int = 50
    step_size: int = 50
    group_column: Optional[str] = None
    holdout_ratio: Optional[float] = None


@dataclass
class ModelConfig:
    name: str
    type: str  # logistic_regression, gradient_boosting
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentPaths:
    output_dir: Path = Path("artifacts")
    model_filename: str = "model.joblib"
    vectorizer_filename: str = "vectorizer.joblib"
    metrics_filename: str = "metrics.json"
    calibration_filename: str = "calibration.csv"
    importances_filename: str = "feature_importances.csv"


@dataclass
class ExperimentConfig:
    target_column: str
    text: TextFeatureConfig
    tabular: TabularFeatureConfig = field(default_factory=TabularFeatureConfig)
    embedding: EmbeddingFeatureConfig = field(default_factory=EmbeddingFeatureConfig)
    calibration: Optional[CalibrationConfig] = None
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    models: List[ModelConfig] = field(default_factory=list)
    dataset_path: Optional[str] = None
    output: ExperimentPaths = field(default_factory=ExperimentPaths)


class SentenceTransformerEncoder(BaseEstimator, TransformerMixin):
    """Wraps a sentence-transformers model for sklearn pipelines."""

    def __init__(self, model_name: str, batch_size: int = 32, device: Optional[str] = None):
        self.model_name = model_name
        self.batch_size = batch_size
        self.device = device
        self._model: Optional[SentenceTransformer] = None

    def fit(self, X: Sequence[str], y: Optional[Sequence[Any]] = None):  # pragma: no cover - optional dependency
        if SentenceTransformer is None:
            raise ImportError(
                "sentence-transformers is not installed. Install it to enable semantic embeddings."
            )
        self._model = SentenceTransformer(self.model_name, device=self.device)
        return self

    def transform(self, X: Sequence[str]):  # pragma: no cover - optional dependency
        if self._model is None:
            raise RuntimeError("SentenceTransformerEncoder must be fitted before calling transform().")
        return self._model.encode(list(X), batch_size=self.batch_size, convert_to_numpy=True)

    def get_feature_names_out(self, input_features: Optional[Sequence[str]] = None) -> np.ndarray:
        if self._model is None:
            raise RuntimeError("Encoder not fitted.")
        dim = self._model.get_sentence_embedding_dimension()
        return np.array([f"embedding_{i}" for i in range(dim)])


def _selector(column: str) -> FunctionTransformer:
    return FunctionTransformer(lambda df: df[column].fillna("").astype(str), validate=False)


def _tabular_selector(columns: List[str]) -> FunctionTransformer:
    return FunctionTransformer(lambda df: df[columns], validate=False)


def build_feature_pipeline(config: ExperimentConfig) -> ColumnTransformer:
    transformers: List[Tuple[str, Any, List[str]]] = []

    text_pipeline = Pipeline(
        steps=[
            ("select", _selector(config.text.column)),
            (
                "tfidf",
                TfidfVectorizer(
                    max_features=config.text.max_features,
                    ngram_range=config.text.ngram_range,
                    min_df=config.text.min_df,
                    max_df=config.text.max_df,
                    stop_words=config.text.stop_words,
                    lowercase=config.text.lowercase,
                ),
            ),
        ]
    )
    transformers.append(("text_tfidf", text_pipeline, [config.text.column]))

    if config.embedding.enabled:
        column = config.embedding.column or config.text.column
        embedding_pipeline = Pipeline(
            steps=[
                ("select", _selector(column)),
                (
                    "embed",
                    SentenceTransformerEncoder(
                        model_name=config.embedding.model_name,
                        batch_size=config.embedding.batch_size,
                        device=config.embedding.device,
                    ),
                ),
            ]
        )
        transformers.append(("semantic", embedding_pipeline, [column]))

    if config.tabular.categorical:
        categorical_pipeline = Pipeline(
            steps=[
                ("select", _tabular_selector(config.tabular.categorical)),
                ("impute", SimpleImputer(strategy=config.tabular.categorical_impute)),
                ("encode", OneHotEncoder(handle_unknown="ignore")),
            ]
        )
        transformers.append(("categorical", categorical_pipeline, config.tabular.categorical))

    if config.tabular.numeric:
        numeric_pipeline = Pipeline(
            steps=[
                ("select", _tabular_selector(config.tabular.numeric)),
                ("impute", SimpleImputer(strategy=config.tabular.numeric_impute)),
                ("scale", StandardScaler()),
            ]
        )
        transformers.append(("numeric", numeric_pipeline, config.tabular.numeric))

    return ColumnTransformer(transformers=transformers, sparse_threshold=0.3)


def _build_estimator(model: ModelConfig) -> BaseEstimator:
    kind = model.type.lower()
    params = dict(model.params)
    if kind == "logistic_regression":
        from sklearn.linear_model import LogisticRegression

        return LogisticRegression(max_iter=200, solver="lbfgs", **params)
    if kind == "gradient_boosting":
        from sklearn.ensemble import HistGradientBoostingClassifier

        defaults = {"loss": "log_loss", "max_iter": 200}
        defaults.update(params)
        return HistGradientBoostingClassifier(**defaults)
    raise ValueError(f"Unsupported model type: {model.type}")


def build_pipeline(config: ExperimentConfig, model: ModelConfig) -> Pipeline:
    features = build_feature_pipeline(config)
    estimator = _build_estimator(model)
    if config.calibration:
        estimator = CalibratedClassifierCV(
            base_estimator=estimator,
            method=config.calibration.method,
            cv=config.calibration.cv,
        )
    return Pipeline([("features", features), ("classifier", estimator)])


def rolling_time_series_splits(
    df: pd.DataFrame,
    time_column: str,
    min_train_size: int,
    test_window_size: int,
    step_size: int,
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    ordered = df.sort_values(time_column)
    index = ordered.index.to_numpy()
    n = len(index)
    cursor = min_train_size
    while cursor < n:
        train_end = cursor
        test_end = min(train_end + test_window_size, n)
        if test_end <= train_end:
            break
        yield index[:train_end], index[train_end:test_end]
        if test_end == n:
            break
        cursor += step_size


def leave_one_event_out_splits(df: pd.DataFrame, group_column: str) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    if group_column not in df.columns:
        raise ValueError(f"Group column '{group_column}' not present in dataframe")
    for value, part in df.groupby(group_column):
        test_idx = part.index.to_numpy()
        train_idx = df.index.difference(part.index).to_numpy()
        if train_idx.size and test_idx.size:
            yield train_idx, test_idx


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, y_prob: np.ndarray) -> Dict[str, float]:
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "brier": brier_score_loss(y_true, y_prob),
    }
    try:
        metrics["roc_auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["roc_auc"] = float("nan")
    try:
        metrics["log_loss"] = log_loss(y_true, np.stack([1 - y_prob, y_prob], axis=1))
    except ValueError:
        metrics["log_loss"] = float("nan")
    return metrics


def aggregate_metrics(records: List[Dict[str, float]]) -> Dict[str, Dict[str, float]]:
    summary: Dict[str, Dict[str, float]] = {}
    if not records:
        return summary
    keys = records[0].keys()
    for key in keys:
        values = np.array([r[key] for r in records if not np.isnan(r[key])])
        if values.size:
            summary[key] = {
                "mean": float(values.mean()),
                "std": float(values.std(ddof=1) if values.size > 1 else 0.0),
            }
        else:
            summary[key] = {"mean": float("nan"), "std": float("nan")}
    return summary


def compute_calibration_data(y_true: np.ndarray, y_prob: np.ndarray, bins: int = 10) -> pd.DataFrame:
    prob_true, prob_pred = calibration_curve(y_true, y_prob, n_bins=bins, strategy="uniform")
    return pd.DataFrame({"prob_pred": prob_pred, "prob_true": prob_true})


def get_feature_names(pipeline: Pipeline) -> List[str]:
    features = pipeline.named_steps.get("features")
    if features and hasattr(features, "get_feature_names_out"):
        return list(features.get_feature_names_out())
    return []


def compute_feature_importances(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    from sklearn.inspection import permutation_importance

    result = permutation_importance(pipeline, X, y, n_repeats=5, random_state=42, scoring="neg_log_loss")
    feature_names = get_feature_names(pipeline)
    if not feature_names:
        feature_names = [f"feature_{i}" for i in range(result.importances_mean.size)]
    df = pd.DataFrame({
        "feature": feature_names,
        "importance": result.importances_mean,
    })
    return df.sort_values("importance", ascending=False)


@dataclass
class FoldResult:
    fold_id: str
    metrics: Dict[str, float]
    calibration: pd.DataFrame
    y_true: np.ndarray
    y_prob: np.ndarray


@dataclass
class ModelResult:
    model_name: str
    fold_results: List[FoldResult]
    aggregate: Dict[str, Dict[str, float]]
    final_pipeline: Pipeline
    feature_importances: pd.DataFrame
    holdout_metrics: Optional[Dict[str, float]] = None
    holdout_calibration: Optional[pd.DataFrame] = None


def split_holdout(df: pd.DataFrame, validation: ValidationConfig) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    if validation.holdout_ratio is None:
        return df, None
    train_df, holdout_df = train_test_split(df, test_size=validation.holdout_ratio, shuffle=False)
    return train_df, holdout_df


def run_experiment(df: pd.DataFrame, config: ExperimentConfig, model_cfg: ModelConfig) -> ModelResult:
    df = df.copy()
    train_df, holdout_df = split_holdout(df, config.validation)

    if config.validation.strategy == "rolling":
        splitter = rolling_time_series_splits(
            train_df,
            time_column=config.validation.time_column,
            min_train_size=config.validation.min_train_size,
            test_window_size=config.validation.test_window_size,
            step_size=config.validation.step_size,
        )
    elif config.validation.strategy == "leave_one_event_out":
        if not config.validation.group_column:
            raise ValueError("group_column is required for leave_one_event_out validation")
        splitter = leave_one_event_out_splits(train_df, config.validation.group_column)
    else:
        raise ValueError(f"Unknown validation strategy: {config.validation.strategy}")

    fold_results: List[FoldResult] = []
    fold_metrics: List[Dict[str, float]] = []

    for fold_index, (train_idx, test_idx) in enumerate(splitter):
        pipeline = build_pipeline(config, model_cfg)
        X_train = train_df.loc[train_idx]
        y_train = train_df.loc[train_idx, config.target_column]
        X_test = train_df.loc[test_idx]
        y_test = train_df.loc[test_idx, config.target_column]

        pipeline.fit(X_train, y_train)
        y_prob = pipeline.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= 0.5).astype(int)
        metrics = compute_metrics(y_test.to_numpy(), y_pred, y_prob)
        calibration = compute_calibration_data(y_test.to_numpy(), y_prob)

        fold_id = f"fold_{fold_index}"
        fold_metrics.append(metrics)
        fold_results.append(
            FoldResult(
                fold_id=fold_id,
                metrics=metrics,
                calibration=calibration,
                y_true=y_test.to_numpy(),
                y_prob=y_prob,
            )
        )

    aggregate = aggregate_metrics(fold_metrics)

    final_pipeline = build_pipeline(config, model_cfg)
    final_pipeline.fit(train_df, train_df[config.target_column])
    feature_importances = compute_feature_importances(final_pipeline, train_df, train_df[config.target_column])

    holdout_metrics = None
    holdout_calibration = None
    if holdout_df is not None and not holdout_df.empty:
        y_prob_holdout = final_pipeline.predict_proba(holdout_df)[:, 1]
        y_pred_holdout = (y_prob_holdout >= 0.5).astype(int)
        holdout_metrics = compute_metrics(
            holdout_df[config.target_column].to_numpy(), y_pred_holdout, y_prob_holdout
        )
        holdout_calibration = compute_calibration_data(
            holdout_df[config.target_column].to_numpy(), y_prob_holdout
        )

    return ModelResult(
        model_name=model_cfg.name,
        fold_results=fold_results,
        aggregate=aggregate,
        final_pipeline=final_pipeline,
        feature_importances=feature_importances,
        holdout_metrics=holdout_metrics,
        holdout_calibration=holdout_calibration,
    )


def save_model_artifacts(result: ModelResult, config: ExperimentConfig, run_dir: Path) -> None:
    run_dir.mkdir(parents=True, exist_ok=True)

    dump(result.final_pipeline, run_dir / config.output.model_filename)
    dump(result.final_pipeline.named_steps["features"], run_dir / config.output.vectorizer_filename)

    metrics_payload = {
        "fold_metrics": [fold.metrics for fold in result.fold_results],
        "aggregate_metrics": result.aggregate,
        "holdout_metrics": result.holdout_metrics,
    }
    (run_dir / config.output.metrics_filename).write_text(json.dumps(metrics_payload, indent=2))

    calibration_frames = []
    for fold in result.fold_results:
        df_fold = fold.calibration.copy()
        df_fold["fold_id"] = fold.fold_id
        calibration_frames.append(df_fold)
    if result.holdout_calibration is not None:
        df_holdout = result.holdout_calibration.copy()
        df_holdout["fold_id"] = "holdout"
        calibration_frames.append(df_holdout)
    if calibration_frames:
        pd.concat(calibration_frames, ignore_index=True).to_csv(
            run_dir / config.output.calibration_filename, index=False
        )

    result.feature_importances.to_csv(run_dir / config.output.importances_filename, index=False)


__all__ = [
    "TextFeatureConfig",
    "TabularFeatureConfig",
    "EmbeddingFeatureConfig",
    "CalibrationConfig",
    "ValidationConfig",
    "ModelConfig",
    "ExperimentPaths",
    "ExperimentConfig",
    "ModelResult",
    "run_experiment",
    "save_model_artifacts",
]
