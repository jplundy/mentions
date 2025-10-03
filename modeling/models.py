"""Estimator factories for baseline models."""

from __future__ import annotations

from typing import Any, Dict

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression

from .config import ModelConfig


def build_estimator(config: ModelConfig):
    """Construct an estimator according to the configuration."""

    params: Dict[str, Any] = {**config.params}

    if config.type == "logistic_regression":
        default_params = {
            "penalty": "l2",
            "solver": "lbfgs",
            "max_iter": 1000,
            "class_weight": "balanced",
        }
        default_params.update(params)
        return LogisticRegression(**default_params)

    if config.type == "gradient_boosting":
        default_params = {
            "learning_rate": 0.05,
            "n_estimators": 200,
            "subsample": 0.8,
            "max_depth": 3,
            "random_state": 42,
        }
        default_params.update(params)
        return GradientBoostingClassifier(**default_params)

    raise ValueError(f"Unsupported model type: {config.type}")
