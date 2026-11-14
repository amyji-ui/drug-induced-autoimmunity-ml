"""Report on one grid search, written by 02_tuning.py's caller.

Each per-model script runs the search, then passes the outcome here to write
logs/tuning_<model_name>.log. Keeps 02_tuning.py free of any logging.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

preprocessing = importlib.import_module("00_preprocessing")
tuning = importlib.import_module("02_tuning")

N_TOP_CONFIGURATIONS = 5


def report_header(outcome) -> None:
    """Log what was searched and how the pipeline was configured."""
    logging.info("Grid search: %s", outcome.model_name)
    logging.info("=" * 60)
    logging.info("  scaling: %s (from config.yaml)", outcome.scaled)
    logging.info(
        "  %d combinations x %d folds = %d fits",
        outcome.n_combinations,
        outcome.n_folds,
        outcome.n_combinations * outcome.n_folds,
    )
    logging.info("")


def report_ranking(outcome) -> None:
    """Log the best configurations by the mean selection metric."""
    logging.info("  top configurations by mean %s:", config.SELECTION_METRIC)
    for parameters_json, row in outcome.summary.head(N_TOP_CONFIGURATIONS).iterrows():
        logging.info(
            "    %.3f +/- %.3f  %s",
            row[(config.SELECTION_METRIC, "mean")],
            row[(config.SELECTION_METRIC, "std")],
            json.loads(parameters_json),
        )
    logging.info("")


def report_selection(outcome) -> None:
    """Log the chosen configuration and its fold-level scores across every metric."""
    best_mean = outcome.summary.loc[
        outcome.best_params_json, (config.SELECTION_METRIC, "mean")
    ]
    best_std = outcome.summary.loc[
        outcome.best_params_json, (config.SELECTION_METRIC, "std")
    ]

    logging.info("  selected: %s", json.loads(outcome.best_params_json))
    logging.info(
        "  mean %s: %.3f +/- %.3f", config.SELECTION_METRIC, best_mean, best_std
    )
    logging.info("")

    winner_rows = outcome.results[
        outcome.results["params_json"] == outcome.best_params_json
    ]
    metric_columns = tuning.get_metric_columns(outcome.results)

    logging.info("  fold-level scores for the selected configuration:")
    for _, row in winner_rows.iterrows():
        scores = "  ".join(f"{name}={row[name]:.3f}" for name in metric_columns)
        logging.info("    fold %d: %s", row["fold_index"], scores)
    logging.info(
        "    mean:   %s",
        "  ".join(f"{name}={winner_rows[name].mean():.3f}" for name in metric_columns),
    )
    logging.info(
        "    std:    %s",
        "  ".join(f"{name}={winner_rows[name].std():.3f}" for name in metric_columns),
    )
    logging.info("")


def report_artifacts(outcome) -> None:
    """Log the runtime and where the scores and fitted pipelines were written."""
    logging.info(
        "  %d fits in %.1fs", len(outcome.results), outcome.elapsed_seconds
    )
    logging.info("  scores    -> %s", outcome.results_path)
    logging.info("  pipelines -> %s", outcome.pipeline_path)


def report_grid_search(outcome) -> Path:
    """Write the full report for one grid search and return the log path."""
    log_path = config.LOG_DIR / f"tuning_{outcome.model_name}.log"
    preprocessing.configure_logging(log_path)

    report_header(outcome)
    report_ranking(outcome)
    report_selection(outcome)
    report_artifacts(outcome)

    return log_path
