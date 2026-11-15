"""Tune kNN with the shared preprocessing, scaffold folds, and reporting."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from sklearn.neighbors import KNeighborsClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")


def main() -> None:
    """Run the configured kNN grid and report its validation results."""
    estimator = KNeighborsClassifier()
    param_grid = config.MODELS["knn"]["param_grid"]
    outcome = tuning.run_grid_search(
        model_name="knn",
        estimator=estimator,
        param_grid=param_grid,
    )
    tuning_report.report_grid_search(outcome)


if __name__ == "__main__":
    main()
