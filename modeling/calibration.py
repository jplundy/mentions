"""Model calibration utilities."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable

import numpy as np
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.metrics import brier_score_loss, log_loss

from .config import CalibrationConfig


@dataclass
class CalibrationResult:
    """Container for calibration diagnostics."""

    method: str
    brier: float
    log_loss: float
    curve: Dict[str, Iterable[float]]


def calibrate_model(estimator, X, y, config: CalibrationConfig):
    """Calibrate estimator outputs if requested."""

    if config.method == "none":
        estimator.fit(X, y)
        probs = estimator.predict_proba(X)[:, 1]
        return estimator, build_calibration_result(probs, y, config.method)

    calibrated = CalibratedClassifierCV(
        estimator=estimator,
        method="sigmoid" if config.method == "sigmoid" else "isotonic",
        cv=config.cv,
    )
    calibrated.fit(X, y)
    probs = calibrated.predict_proba(X)[:, 1]
    return calibrated, build_calibration_result(probs, y, config.method)


def build_calibration_result(probs: np.ndarray, y_true, method: str) -> CalibrationResult:
    """Compute calibration metrics and curves."""

    brier = brier_score_loss(y_true, probs)
    loss = log_loss(y_true, probs, labels=[0, 1])
    fraction_of_positives, mean_predicted_value = calibration_curve(y_true, probs, n_bins=10)
    return CalibrationResult(
        method=method,
        brier=brier,
        log_loss=loss,
        curve={
            "fraction_of_positives": fraction_of_positives.tolist(),
            "mean_predicted_value": mean_predicted_value.tolist(),
        },
    )
