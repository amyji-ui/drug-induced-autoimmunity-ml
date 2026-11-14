"""Shared paths, data loading, and preprocessing transformers for the DIA prediction project.

Nothing here runs on import: every transformer is fit inside a CV training fold by the
scripts that import this module.
"""

from __future__ import annotations

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

PROJECT_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_DIR / "data"
LOG_DIR = PROJECT_DIR / "logs"

TRAIN_FILE = DATA_DIR / "DIA_trainingset_RDKit_descriptors.csv"
TEST_FILE = DATA_DIR / "DIA_testset_RDKit_descriptors.csv"

LABEL_COLUMN = "Label"
SMILES_COLUMN = "SMILES"
POSITIVE_LABEL = 1

# Models that need standardized descriptors; trees and boosting do not.
SCALED_MODELS = ("logistic_regression", "rbf_svm", "knn")


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
        if column not in (LABEL_COLUMN, SMILES_COLUMN)
    ]
    return descriptor_columns


def split_features_labels(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """Split one dataframe into descriptors, labels, and SMILES strings."""
    descriptor_columns = get_descriptor_columns(dataframe)
    features = dataframe[descriptor_columns]
    labels = dataframe[LABEL_COLUMN]
    smiles = dataframe[SMILES_COLUMN]
    return features, labels, smiles


# --------------------------------------------------------------------------
# Preprocessing steps (fit inside each CV fold, never on the full dataset)
# --------------------------------------------------------------------------


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
    scale: bool, correlation_threshold: float | None = None
) -> Pipeline:
    """Assemble the preprocessing pipeline: drop constant columns, optionally drop correlated ones, optionally standardize.

    VarianceThreshold learns the constant columns from the training fold rather than a hard-coded list.
    Pass scale=True for logistic regression, RBF-SVM, and kNN.
    """
    steps: list[tuple[str, BaseEstimator]] = [
        ("drop_constant", VarianceThreshold(threshold=0.0))
    ]

    if correlation_threshold is not None:
        steps.append(
            ("drop_correlated", CorrelationFilter(threshold=correlation_threshold))
        )

    if scale:
        steps.append(("scale", StandardScaler()))

    return Pipeline(steps)


def build_model_pipeline(
    estimator: BaseEstimator,
    scale: bool,
    correlation_threshold: float | None = None,
) -> Pipeline:
    """Put a classifier behind the preprocessing pipeline so every transform is fit on the training fold only."""
    preprocessor = build_preprocessor(
        scale=scale, correlation_threshold=correlation_threshold
    )
    model_pipeline = Pipeline(
        [
            ("preprocess", preprocessor),
            ("model", estimator),
        ]
    )
    return model_pipeline
