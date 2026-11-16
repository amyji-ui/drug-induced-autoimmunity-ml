from __future__ import annotations

import importlib
import sys
from pathlib import Path

from sklearn.svm import SVC

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")

def main() -> None:
    """Run the configured RBF-SVM grid and report its validation results."""
    estimator = SVC(kernel="rbf", probability=True, random_state=config.RANDOM_SEED)
    param_grid = config.MODELS["rbf_svm"]["param_grid"]
    outcome = tuning.run_grid_search(
        model_name="rbf_svm",
        estimator=estimator,
        param_grid=param_grid,
    )
    tuning_report.report_grid_search(outcome)


if __name__ == "__main__":
    main()