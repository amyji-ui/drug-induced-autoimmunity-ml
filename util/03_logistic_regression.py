"""Tune L2 Logistic Regression on the shared scaffold folds using config.yaml."""

from __future__ import annotations

import importlib
import logging
import math
import sys
import warnings
from pathlib import Path

from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

tuning = importlib.import_module("02_tuning")
tuning_report = importlib.import_module("tuning_report")


def main() -> None:
    """Run the agreed grid and reject any fit that reports a convergence failure."""
    model_name = "logistic_regression"
    settings = config.MODELS[model_name]
    estimator = LogisticRegression(
        random_state=config.RANDOM_SEED,
        **settings["estimator"],
    )
    param_grid = settings["param_grid"]
    n_combinations = math.prod(len(values) for values in param_grid.values())
    print(
        f"Starting {model_name}: {n_combinations} combinations x {config.N_SPLITS} folds; "
        f"seed={config.RANDOM_SEED}; convergence warnings are fatal.",
        flush=True,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", ConvergenceWarning)
        outcome = tuning.run_grid_search(model_name, estimator, param_grid)

    tuning_report.report_grid_search(outcome)
    logging.info("  estimator settings: %s", settings["estimator"])
    logging.info("  convergence: all %d fits completed without ConvergenceWarning", len(outcome.results))
    logging.info("  selected-model iterations per fold: %s", [
        pipeline.named_steps["model"].n_iter_.tolist()
        for pipeline in outcome.best_pipelines
    ])


if __name__ == "__main__":
    main()
