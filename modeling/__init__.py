"""Modeling package for baseline experiments."""

from importlib import import_module

from .config import (
    ExperimentConfig,
    FeatureConfig,
    ModelConfig,
    ValidationConfig,
    CalibrationConfig,
    TrackingConfig,
)

__all__ = [
    "ExperimentConfig",
    "FeatureConfig",
    "ModelConfig",
    "ValidationConfig",
    "CalibrationConfig",
    "TrackingConfig",
    "BaselineExperiment",
    "KalshiClient",
    "KalshiAPIError",
    "MarketComparison",
    "compare_model_to_market_odds",
]


_LAZY_IMPORTS = {
    "BaselineExperiment": (".experiments", "BaselineExperiment"),
    "KalshiClient": (".markets", "KalshiClient"),
    "KalshiAPIError": (".markets", "KalshiAPIError"),
    "MarketComparison": (".markets", "MarketComparison"),
    "compare_model_to_market_odds": (".markets", "compare_model_to_market_odds"),
}


def __getattr__(name):
    if name in _LAZY_IMPORTS:
        module_name, attribute = _LAZY_IMPORTS[name]
        module = import_module(module_name, __name__)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(set(__all__))
