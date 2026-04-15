"""Evaluation utilities for baseline experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import numpy as np
from sklearn import metrics


@dataclass
class FoldResult:
    """Metrics computed for a validation fold."""

    fold: int
    metrics: Dict[str, float]


@dataclass
class ExperimentResult:
    """Aggregate results across validation folds."""

    metrics: Dict[str, float]
    folds: List[FoldResult]
    calibration: Dict[str, float]
    feature_importances: Dict[str, float]


def compute_classification_metrics(y_true, y_prob) -> Dict[str, float]:
    """Compute classification metrics for probabilistic models."""

    y_pred = (y_prob >= 0.5).astype(int)
    try:
        roc_auc = metrics.roc_auc_score(y_true, y_prob)
    except ValueError:
        # Single-class validation fold — metric is undefined.
        roc_auc = float("nan")
    return {
        "log_loss": metrics.log_loss(y_true, y_prob, labels=[0, 1]),
        "brier": metrics.brier_score_loss(y_true, y_prob),
        "roc_auc": roc_auc,
        "average_precision": metrics.average_precision_score(y_true, y_prob),
        "f1": metrics.f1_score(y_true, y_pred),
        "precision": metrics.precision_score(y_true, y_pred, zero_division=0),
        "recall": metrics.recall_score(y_true, y_pred),
        "accuracy": metrics.accuracy_score(y_true, y_pred),
    }


def summarize_metrics(fold_metrics: List[FoldResult]) -> Dict[str, float]:
    """Aggregate fold metrics by averaging."""

    aggregated: Dict[str, List[float]] = {}
    for fold in fold_metrics:
        for key, value in fold.metrics.items():
            aggregated.setdefault(key, []).append(value)
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
