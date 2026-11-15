"""Grid search for XGBoost.

scale_pos_weight is not searched: the proposal asks for it to be computed from each
training fold, which the PER_FOLD sentinel does inside 02_tuning.py.
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

# A first pass with max_depth [2,3,4] and min_child_weight [1,5] put every top configuration
# on the grid boundary, so both ranges were shifted; max_depth 4 now wins from inside [3,4,5].
# A separate ablation at the selected configuration confirmed min_child_weight 1 is a true
# optimum (0 -> 0.760, 0.5 -> 0.776, 1 -> 0.777, 2 -> 0.769, 5 -> 0.748), so [1,2] stands.
# Held at 48 combinations to keep the search budget comparable across the five algorithms.
PARAM_GRID = {
    "model__max_depth": [3, 4, 5],
    "model__learning_rate": [0.05, 0.1],
    "model__n_estimators": [200, 500],
    "model__colsample_bytree": [0.5, 0.8],
    "model__min_child_weight": [1, 2],
    "model__scale_pos_weight": [tuning.PER_FOLD],
}


def main() -> None:
    """Tune XGBoost on the shared scaffold-grouped folds."""
    estimator = XGBClassifier(
        eval_metric="logloss",
        n_jobs=1,
        random_state=config.RANDOM_SEED,
        verbosity=0,
    )
    outcome = tuning.run_grid_search(
        model_name=MODEL_NAME,
        estimator=estimator,
        param_grid=PARAM_GRID,
    )
    tuning_report.report_grid_search(outcome)


if __name__ == "__main__":
    main()
