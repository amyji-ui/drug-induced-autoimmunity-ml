"""Tune XGBoost with the shared preprocessing, scaffold folds, and reporting.

The grid lives in config.yaml under models.xgboost.param_grid, including the
"per_fold" sentinel that makes 02_tuning.py compute scale_pos_weight per training fold.
"""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

from xgboost import XGBClassifier

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")

MODEL_NAME = "xgboost"


def main() -> None:
    """Run the configured XGBoost grid and report its validation results."""
    estimator = XGBClassifier(
        eval_metric="logloss",
        n_jobs=1,
        random_state=config.RANDOM_SEED,
        verbosity=0,
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
