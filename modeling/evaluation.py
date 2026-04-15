"""Evaluation utilities for baseline experiments."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
from sklearn import metrics


@dataclass
class FoldResult:
    """Metrics computed for a validation fold."""

    fold: int
    metrics: Dict[str, Optional[float]]
    optimal_threshold: float = 0.5
    has_positives: bool = True


@dataclass
class ExperimentResult:
    """Aggregate results across validation folds."""

    metrics: Dict[str, float]
    folds: List[FoldResult]
    calibration: Dict[str, float]
    feature_importances: Dict[str, float]
    optimal_threshold: float = 0.5


def compute_classification_metrics(
    y_true, y_prob, threshold: float = 0.5
) -> Dict[str, Optional[float]]:
    """Compute classification metrics for probabilistic models.

    Ranking metrics (roc_auc, average_precision) are set to None when the
    validation set contains only one class, which would otherwise produce NaN
    or raise exceptions and corrupt aggregate summaries.
    """
    y_pred = (y_prob >= threshold).astype(int)
    unique_classes = np.unique(y_true)
    single_class = len(unique_classes) < 2

    result: Dict[str, Optional[float]] = {
        "log_loss": float(metrics.log_loss(y_true, y_prob, labels=[0, 1])),
        "brier": float(metrics.brier_score_loss(y_true, y_prob)),
        "roc_auc": None,
        "average_precision": None,
        "f1": float(metrics.f1_score(y_true, y_pred, zero_division=0)),
        "precision": float(metrics.precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(metrics.recall_score(y_true, y_pred, zero_division=0)),
        "accuracy": float(metrics.accuracy_score(y_true, y_pred)),
    }

    if not single_class:
        result["roc_auc"] = float(metrics.roc_auc_score(y_true, y_prob))
        result["average_precision"] = float(
            metrics.average_precision_score(y_true, y_prob)
        )

    return result


def optimize_threshold(
    y_true, y_prob, metric: str = "f1"
) -> Tuple[float, float]:
    """Find the probability threshold that maximises the given metric.

    Sweeps 99 candidate thresholds between 0.01 and 0.99.  When the
    validation set contains only one class no sweep is possible; (0.5, 0.0)
    is returned as a safe default.

    Parameters
    ----------
    y_true:
        Ground-truth binary labels.
    y_prob:
        Predicted positive-class probabilities.
    metric:
        One of ``"f1"``, ``"precision"``, or ``"recall"``.

    Returns
    -------
    Tuple[float, float]
        ``(optimal_threshold, best_metric_value)``
    """
    if len(np.unique(y_true)) < 2:
        return 0.5, 0.0

    thresholds = np.linspace(0.01, 0.99, 99)
    best_threshold = 0.5
    best_value = -1.0

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        if metric == "f1":
            value = float(metrics.f1_score(y_true, y_pred, zero_division=0))
        elif metric == "precision":
            value = float(metrics.precision_score(y_true, y_pred, zero_division=0))
        elif metric == "recall":
            value = float(metrics.recall_score(y_true, y_pred, zero_division=0))
        else:
            raise ValueError(f"Unsupported threshold metric: {metric!r}")
        if value > best_value:
            best_value = value
            best_threshold = float(thr)

    return best_threshold, max(best_value, 0.0)


def aggregate_optimal_threshold(fold_results: List[FoldResult]) -> float:
    """Return the median optimal threshold across folds that had positive examples."""

    thresholds = [
        f.optimal_threshold
        for f in fold_results
        if f.has_positives and f.optimal_threshold is not None
    ]
    if not thresholds:
        return 0.5
    return float(np.median(thresholds))


def summarize_metrics(fold_metrics: List[FoldResult]) -> Dict[str, float]:
    """Aggregate fold metrics, excluding None and NaN values per metric.

    Folds where a metric is None (e.g. roc_auc on a single-class validation
    set) do not contribute to that metric's average.  Each returned value is
    the mean over the non-missing folds.
    """
    aggregated: Dict[str, List[float]] = {}
    for fold in fold_metrics:
        for key, value in fold.metrics.items():
            if value is None:
                continue
            try:
                f = float(value)
            except (TypeError, ValueError):
                continue
            if math.isnan(f):
                continue
            aggregated.setdefault(key, []).append(f)
    return {key: float(np.mean(values)) for key, values in aggregated.items()}


def extract_feature_importances(pipeline, feature_names: List[str]) -> Dict[str, float]:
    """Extract feature importances or coefficients from the trained model."""

    model = pipeline.named_steps.get("model")
    if model is None:
        return {}

    if hasattr(model, "coef_"):
        coefs = np.ravel(model.coef_)
        return {
            name: float(coeff)
            for name, coeff in zip(feature_names, coefs[: len(feature_names)])
        }

    if hasattr(model, "feature_importances_"):
        importances = np.ravel(model.feature_importances_)
        return {
            name: float(importance)
            for name, importance in zip(feature_names, importances[: len(feature_names)])
        }

    return {}
