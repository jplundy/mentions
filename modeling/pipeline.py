"""Utilities to assemble modeling pipelines."""

from __future__ import annotations

from typing import List

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from .config import FeatureConfig
from .features.metadata import CategoricalMetadataBlock
from .features.text import SentenceTransformerBlock, TfidfTextBlock


def build_feature_blocks(config: FeatureConfig) -> List[tuple[str, object, str | list[str]]]:
    """Construct feature transformer tuples for a column transformer."""

    blocks: List[tuple[str, object, str | list[str]]] = []

    text_block = TfidfTextBlock(
        column=config.text.column,
        max_features=config.text.params.get("max_features", None),
        ngram_range=tuple(config.text.params.get("ngram_range", (1, 2))),
        min_df=config.text.params.get("min_df", 2),
        max_df=config.text.params.get("max_df", 0.9),
        sublinear_tf=config.text.params.get("sublinear_tf", True),
        lowercase=config.text.params.get("lowercase", True),
        analyzer=config.text.params.get("analyzer", "word"),
    )
    blocks.append((text_block.name, text_block.build(), text_block.column))

    if config.categorical.columns:
        categorical_block = CategoricalMetadataBlock(
            columns=config.categorical.columns,
            drop=config.categorical.drop,
            sparse=config.categorical.sparse,
            handle_unknown=config.categorical.handle_unknown,
        )
        blocks.append(
            (categorical_block.name, categorical_block.build(), categorical_block.columns)
        )

    if config.include_embeddings:
        embeddings_block = SentenceTransformerBlock(
            model_name=config.embedding_params.get(
                "model_name", "sentence-transformers/all-MiniLM-L6-v2"
            ),
            column=config.text.column,
            batch_size=config.embedding_params.get("batch_size", 32),
            normalize_embeddings=config.embedding_params.get(
                "normalize_embeddings", True
            ),
        )
        blocks.append(
            (embeddings_block.name, embeddings_block.build(), embeddings_block.column)
        )

    return blocks


def build_feature_pipeline(config: FeatureConfig) -> ColumnTransformer:
    """Assemble a column transformer from the configured blocks."""

    blocks = build_feature_blocks(config)
    return ColumnTransformer(blocks, remainder="drop")


def append_standardizer(pipeline: Pipeline) -> Pipeline:
    """Attach a standard scaler to dense feature outputs."""

    pipeline.steps.append(("scaler", StandardScaler(with_mean=False)))
    return pipeline


def build_training_pipeline(config: FeatureConfig, estimator) -> Pipeline:
    """Create a full pipeline consisting of features and estimator."""

    feature_transformer = build_feature_pipeline(config)
    steps: List[tuple[str, object]] = [("features", feature_transformer), ("model", estimator)]
    return Pipeline(steps)
