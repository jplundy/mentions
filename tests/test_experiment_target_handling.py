import pytest

pytest.importorskip("joblib")
pytest.importorskip("pandas")

from modeling import BaselineExperiment, ExperimentConfig

import pandas as pd


def _build_config(tmp_path):
    dataset_path = tmp_path / "dummy.csv"
    dataset_path.write_text("", encoding="utf-8")
    return ExperimentConfig(dataset_path=dataset_path, target_column="label")


def test_prepare_features_and_target_drops_missing_rows(tmp_path):
    config = _build_config(tmp_path)
    experiment = BaselineExperiment(config)
    df = pd.DataFrame(
        {
            "label": [1, None, 0],
            "text": ["first", "second", "third"],
        }
    )

    features, target = experiment._prepare_features_and_target(df)

    assert len(features) == 2
    assert list(target) == [1, 0]
    assert features.index.tolist() == target.index.tolist()


def test_prepare_features_and_target_raises_when_all_missing(tmp_path):
    config = _build_config(tmp_path)
    experiment = BaselineExperiment(config)
    df = pd.DataFrame({"label": [None, None]})

    with pytest.raises(ValueError, match="No rows with a valid target value"):
        experiment._prepare_features_and_target(df)
