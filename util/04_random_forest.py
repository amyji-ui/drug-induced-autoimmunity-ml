"""Tune Random Forest with the shared preprocessing, scaffold folds, and reporting.

The grid lives in config.yaml under models.random_forest.param_grid, including both
class_weight settings the proposal asks to compare (section 4.3).
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")

MODEL_NAME = "random_forest"


def main() -> None:
    """Run the configured Random Forest grid and report its validation results."""
    estimator = RandomForestClassifier(
        n_jobs=1,
        random_state=config.RANDOM_SEED,
    )
    param_grid = config.MODELS[MODEL_NAME]["param_grid"]
    outcome = tuning.run_grid_search(
        model_name=MODEL_NAME,
        estimator=estimator,
        param_grid=param_grid,
    )
    tuning_report.report_grid_search(outcome)


if __name__ == "__main__":
    main()
