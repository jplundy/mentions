"""Configuration dataclasses and helpers for modeling experiments."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional

import yaml


@dataclass
class TextFeatureConfig:
    """Configuration for text feature extraction."""

    type: Literal["tfidf", "sentence_transformer"] = "tfidf"
    column: str = "text"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CategoricalFeatureConfig:
    """Configuration for categorical metadata encoding."""

    columns: List[str] = field(default_factory=list)
    drop: Optional[Literal["first", "if_binary"]] = None
    sparse: bool = True
    handle_unknown: Literal["ignore", "error"] = "ignore"


@dataclass
class FeatureConfig:
    """Feature configuration wrapper."""

    text: TextFeatureConfig = field(default_factory=TextFeatureConfig)
    categorical: CategoricalFeatureConfig = field(
        default_factory=CategoricalFeatureConfig
    )
    include_embeddings: bool = False
    embedding_params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ModelConfig:
    """Estimator configuration for the baseline models."""

    type: Literal["logistic_regression", "gradient_boosting"] = "logistic_regression"
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ValidationConfig:
    """Configuration for time-aware validation."""

    strategy: Literal["rolling", "leave_one_event_out"] = "rolling"
    n_splits: int = 3
    minimum_train_events: int = 5
    date_column: str = "event_date"
    group_column: str = "event_id"


@dataclass
class CalibrationConfig:
    """Configuration for model calibration."""

    method: Literal["isotonic", "sigmoid", "none"] = "isotonic"
    cv: int = 3


@dataclass
class TrackingConfig:
    """Experiment tracking configuration."""

    output_directory: Path = Path("experiments")
    use_mlflow: bool = False
    mlflow_tracking_uri: Optional[str] = None
    experiment_name: str = "baseline"


@dataclass
class ExperimentConfig:
    """Composite configuration for an experiment run."""

    dataset_path: Path
    target_column: str
    news_features_path: Optional[Path] = None
    feature: FeatureConfig = field(default_factory=FeatureConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)
    calibration: CalibrationConfig = field(default_factory=CalibrationConfig)
    tracking: TrackingConfig = field(default_factory=TrackingConfig)

    @classmethod
    def from_file(cls, path: Path | str) -> "ExperimentConfig":
        """Load an experiment configuration from a YAML file."""

        with Path(path).open("r", encoding="utf-8") as stream:
            data = yaml.safe_load(stream)
        return cls.from_dict(data)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ExperimentConfig":
        """Instantiate from a nested dictionary."""

        dataset_path = Path(data["dataset_path"])
        target_column = data["target_column"]
        news_features_path_raw = data.get("news_features_path")
        news_features_path = (
            Path(news_features_path_raw) if news_features_path_raw else None
        )
        feature = FeatureConfig(
            text=TextFeatureConfig(**data.get("feature", {}).get("text", {})),
            categorical=CategoricalFeatureConfig(
                **data.get("feature", {}).get("categorical", {})
            ),
            include_embeddings=data.get("feature", {}).get("include_embeddings", False),
            embedding_params=data.get("feature", {}).get("embedding_params", {}),
        )
        model = ModelConfig(**data.get("model", {}))
        validation = ValidationConfig(**data.get("validation", {}))
        calibration = CalibrationConfig(**data.get("calibration", {}))
        tracking_dict = data.get("tracking", {})
        tracking = TrackingConfig(
            output_directory=Path(tracking_dict.get("output_directory", "experiments")),
            use_mlflow=tracking_dict.get("use_mlflow", False),
            mlflow_tracking_uri=tracking_dict.get("mlflow_tracking_uri"),
            experiment_name=tracking_dict.get("experiment_name", "baseline"),
        )
        return cls(
            dataset_path=dataset_path,
            target_column=target_column,
            feature=feature,
            model=model,
            validation=validation,
            calibration=calibration,
            tracking=tracking,
            news_features_path=news_features_path,
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize configuration to a dictionary."""

        return {
            "dataset_path": str(self.dataset_path),
            "target_column": self.target_column,
            "news_features_path": str(self.news_features_path)
            if self.news_features_path
            else None,
            "feature": {
                "text": vars(self.feature.text),
                "categorical": vars(self.feature.categorical),
                "include_embeddings": self.feature.include_embeddings,
                "embedding_params": self.feature.embedding_params,
            },
            "model": vars(self.model),
            "validation": vars(self.validation),
            "calibration": vars(self.calibration),
            "tracking": {
                "output_directory": str(self.tracking.output_directory),
                "use_mlflow": self.tracking.use_mlflow,
                "mlflow_tracking_uri": self.tracking.mlflow_tracking_uri,
                "experiment_name": self.tracking.experiment_name,
            },
        }

    def dump(self, path: Path | str) -> None:
        """Write the configuration to disk."""

        with Path(path).open("w", encoding="utf-8") as stream:
            yaml.safe_dump(self.to_dict(), stream, sort_keys=False)
