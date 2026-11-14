"""Data loading and preprocessing transformers for the DIA prediction project.

Nothing here runs on import: every transformer is fit inside a CV training fold by the
scripts that import this module. All settings come from config.py.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import VarianceThreshold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.utils.validation import (
    _check_feature_names_in,
    check_is_fitted,
    validate_data,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config


# --------------------------------------------------------------------------
# Logging
# --------------------------------------------------------------------------


def configure_logging(log_path: Path) -> None:
    """Send a report to both the log file and the console, overwriting any previous run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------


def load_split(csv_path: Path) -> pd.DataFrame:
    """Read one split of the DIA dataset from disk."""
    dataframe = pd.read_csv(csv_path)
    return dataframe


def get_descriptor_columns(dataframe: pd.DataFrame) -> list[str]:
    """Return the descriptor column names, excluding the label and the SMILES string."""
    descriptor_columns = [
        column
        for column in dataframe.columns
        if column not in (config.LABEL_COLUMN, config.SMILES_COLUMN)
    ]
    return descriptor_columns


def split_features_labels(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split one dataframe into descriptors, labels, and SMILES strings."""
    descriptor_columns = get_descriptor_columns(dataframe)
    features = dataframe[descriptor_columns]
    labels = dataframe[config.LABEL_COLUMN]
    smiles = dataframe[config.SMILES_COLUMN]
    return features, labels, smiles


# --------------------------------------------------------------------------
# Preprocessing steps (fit inside each CV fold, never on the full dataset)
# --------------------------------------------------------------------------


class LogTransformColumns(BaseEstimator, TransformerMixin):
    """Apply log1p to the named descriptors and leave every other column untouched.

    This is row-independent, so it learns nothing and cannot leak between folds.
    Expects a DataFrame, and so belongs first in the pipeline.
    """

    def __init__(self, columns: tuple[str, ...] = config.HEAVY_TAILED_COLUMNS):
        self.columns = columns

    def fit(self, X, y=None):
        self.feature_names_in_ = np.asarray(X.columns)
        self.n_features_in_ = X.shape[1]
        return self

    def transform(self, X):
        check_is_fitted(self)
        transformed = X.copy()
        for column in self.columns:
            if column in transformed.columns:
                transformed[column] = np.log1p(transformed[column])
        return transformed

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        return np.asarray(self.feature_names_in_)


class CorrelationFilter(BaseEstimator, TransformerMixin):
    """Drop descriptors whose absolute correlation with an earlier kept descriptor exceeds a threshold.

    Of each correlated pair the earlier column is kept, so the result depends on column order.
    The correlation matrix is computed on the training fold only.
    """

    def __init__(self, threshold: float = 0.95):
        self.threshold = threshold  # use grid search to find optimal threshold (default 0.95)

    def fit(self, X, y=None):
        validated_features = validate_data(self, X, dtype=np.float64, reset=True)
        n_features = validated_features.shape[1]
        keep_mask = np.ones(n_features, dtype=bool)
        redundant_pairs: list[tuple[int, int, float]] = []

        if n_features > 1:
            # 1.Build the correlation matrix using absolute value
            correlation_matrix = np.corrcoef(validated_features, rowvar=False)
            absolute_correlation = np.abs(np.nan_to_num(correlation_matrix, nan=0.0))

            # 2. Greedy left-to-right sweep to remove redundant features (keep 1, remove others above threshold).
            for kept_index in range(n_features):
                if not keep_mask[kept_index]:
                    continue
                for later_index in range(kept_index + 1, n_features):
                    is_redundant = (
                        absolute_correlation[kept_index, later_index] > self.threshold
                    )
                    if keep_mask[later_index] and is_redundant:
                        keep_mask[later_index] = False
                        redundant_pairs.append(
                            (
                                later_index,
                                kept_index,
                                float(absolute_correlation[kept_index, later_index]),
                            )
                        )

        self.support_mask_ = keep_mask
        self.n_features_out_ = int(keep_mask.sum())
        # Each entry is (dropped column, column that displaced it, their absolute correlation).
        self.redundant_pairs_ = redundant_pairs
        return self

    def transform(self, X):
        check_is_fitted(self)
        validated_features = validate_data(self, X, dtype=np.float64, reset=False)
        return validated_features[:, self.support_mask_]

    def get_feature_names_out(self, input_features=None):
        check_is_fitted(self)
        feature_names = _check_feature_names_in(self, input_features)
        return np.asarray(feature_names)[self.support_mask_]


def build_preprocessor(
    scale: bool,
    correlation_threshold: float | None = config.CORRELATION_THRESHOLD,
) -> Pipeline:
    """Assemble the preprocessing pipeline: drop constant columns, optionally drop correlated ones, optionally standardize.

    VarianceThreshold learns the constant columns from the training fold rather than a hard-coded list.
    Callers normally reach this through build_model_pipeline, which reads scale from config.yaml.
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("log_heavy_tailed", LogTransformColumns()),
        ("drop_constant", VarianceThreshold(threshold=0.0)),
    ]

    if correlation_threshold is not None:
        steps.append(
            ("drop_correlated", CorrelationFilter(threshold=correlation_threshold))
        )

    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def needs_scaling(model_name: str) -> bool:
    """Look up in config.yaml whether an algorithm needs standardized descriptors.

    Raises on an unknown name so a typo fails loudly instead of silently skipping the scaler.
    """
    if model_name not in config.MODELS:
        raise KeyError(
            f"unknown model {model_name!r}; config.yaml lists {list(config.MODEL_NAMES)}"
        )
    return bool(config.MODELS[model_name]["scale"])


def build_model_pipeline(
    estimator: BaseEstimator,
    model_name: str,
    correlation_threshold: float | None = config.CORRELATION_THRESHOLD,
) -> Pipeline:
    """Put a classifier behind the preprocessing pipeline so every transform is fit on the training fold only.

    Scaling is decided by model_name via config.yaml, not by the caller.
    """
    preprocessor = build_preprocessor(
        scale=needs_scaling(model_name),
        correlation_threshold=correlation_threshold,
    )
    model_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    return model_pipeline
