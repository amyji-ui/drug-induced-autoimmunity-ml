"""Check dependencies and run a training-only kNN smoke check on the shared folds."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

preprocessing = importlib.import_module("00_preprocessing")
grouping = importlib.import_module("01_scaffold_grouping")


def main() -> None:
    """Validate imports, dictionary loading, scaffold isolation, and pipeline predictions."""
    print(f"Python: {sys.version.split()[0]}")
    for module_name in (
        "numpy", "pandas", "scipy", "sklearn", "xgboost", "rdkit",
        "openpyxl", "yaml", "joblib", "matplotlib", "seaborn",
    ):
        module = importlib.import_module(module_name)
        print(f"{module_name}: {module.__version__}")
    for module_name in ("02_tuning", "data_report", "grouping_report", "tuning_report"):
        importlib.import_module(module_name)

    dictionary = pd.read_excel(config.DESCRIPTOR_DICTIONARY_FILE)
    print(f"Descriptor dictionary: {len(dictionary)} rows")
    dataframe = preprocessing.load_split(config.TRAIN_FILE)
    features, labels, smiles = preprocessing.split_features_labels(dataframe)
    if not np.isfinite(features.to_numpy(dtype=float)).all():
        raise ValueError("Training descriptors contain non-finite values")
    groups = grouping.build_scaffold_groups(smiles)
    folds = grouping.build_cv_folds(features, labels, groups)
    print(f"Training: {len(features)} rows, {features.shape[1]} descriptors, {len(set(groups))} groups")

    validation_indices = []
    for fold_index, (train_index, validation_index) in enumerate(folds):
        overlap = set(groups[train_index]) & set(groups[validation_index])
        if overlap:
            raise ValueError(f"Fold {fold_index}: scaffold leakage")
        pipeline = preprocessing.build_model_pipeline(KNeighborsClassifier(), model_name="knn")
        pipeline.fit(features.iloc[train_index], labels.iloc[train_index])
        probabilities = pipeline.predict_proba(features.iloc[validation_index])
        if not np.isfinite(probabilities).all() or not np.allclose(probabilities.sum(axis=1), 1):
            raise ValueError(f"Fold {fold_index}: invalid probabilities")
        validation_indices.extend(validation_index)
        print(f"Fold {fold_index}: fit/predict passed; validation rows={len(validation_index)}, shared groups=0")

    if sorted(validation_indices) != list(range(len(features))):
        raise ValueError("Validation folds do not cover every training row exactly once")
    print("PASS: dependencies, dictionary, scaffold folds, and kNN pipelines; test split not loaded.")


if __name__ == "__main__":
    main()
