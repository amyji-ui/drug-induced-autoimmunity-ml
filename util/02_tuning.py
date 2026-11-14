"""Shared grid-search harness, used by every per-model script.

Freeze this file once the team agrees on it: it fixes the folds, the metrics, the selection
rule, and the output format, so each person only supplies an estimator and a parameter grid.
"""

from __future__ import annotations

import importlib
import json
import logging
import sys
import time
from itertools import product
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, clone
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

preprocessing = importlib.import_module("00_preprocessing")
grouping = importlib.import_module("01_scaffold_grouping")


# --------------------------------------------------------------------------
# Scoring
# --------------------------------------------------------------------------


def score_predictions(labels: pd.Series, probabilities: np.ndarray) -> dict[str, float]:
    """Compute the seven metrics the proposal lists for one validation fold.

    ROC AUC and average precision use the continuous scores; the rest use the predefined
    decision threshold from config.yaml, which is never tuned.
    """
    predictions = (probabilities >= config.DECISION_THRESHOLD).astype(int)
    return {
        "roc_auc": roc_auc_score(labels, probabilities),
        "average_precision": average_precision_score(labels, probabilities),
        "precision": precision_score(labels, predictions, zero_division=0),
        "recall": recall_score(labels, predictions, zero_division=0),
        "balanced_accuracy": balanced_accuracy_score(labels, predictions),
        "mcc": matthews_corrcoef(labels, predictions),
        "f1": f1_score(labels, predictions, zero_division=0),
    }


# --------------------------------------------------------------------------
# Grid search
# --------------------------------------------------------------------------


def expand_param_grid(param_grid: dict[str, list]) -> list[dict]:
    """Turn a grid of lists into the explicit list of parameter combinations."""
    parameter_names = list(param_grid)
    combinations = [
        dict(zip(parameter_names, chosen_values))
        for chosen_values in product(*param_grid.values())
    ]
    return combinations


def load_training_data() -> tuple[pd.DataFrame, pd.Series, np.ndarray]:
    """Load the training split and its scaffold group keys."""
    train_df = preprocessing.load_split(config.TRAIN_FILE)
    features, labels, smiles = preprocessing.split_features_labels(train_df)
    groups = grouping.build_scaffold_groups(smiles)
    return features, labels, groups


def evaluate_combination(
    model_name: str,
    estimator: BaseEstimator,
    combination: dict,
    features: pd.DataFrame,
    labels: pd.Series,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict], list]:
    """Fit one parameter combination on every fold and score its held-out predictions.

    A fresh clone is fitted per fold, so the five pipelines share hyperparameters but not training data.
    """
    parameters_json = json.dumps(combination, sort_keys=True)
    fold_rows = []
    fitted_pipelines = []

    for fold_index, (train_index, validation_index) in enumerate(folds):
        pipeline = preprocessing.build_model_pipeline(
            clone(estimator), model_name=model_name
        )
        pipeline.set_params(**combination)
        pipeline.fit(features.iloc[train_index], labels.iloc[train_index])

        probabilities = pipeline.predict_proba(features.iloc[validation_index])[:, 1]
        fold_scores = score_predictions(labels.iloc[validation_index], probabilities)

        fold_rows.append(
            {
                "model_name": model_name,
                "params_json": parameters_json,
                "fold_index": fold_index,
                **fold_scores,
            }
        )
        fitted_pipelines.append(pipeline)

    return fold_rows, fitted_pipelines


def get_metric_columns(results: pd.DataFrame) -> list[str]:
    """Return the score columns, excluding the identifying ones."""
    identifier_columns = ("model_name", "params_json", "fold_index")
    return [column for column in results.columns if column not in identifier_columns]


def summarize_results(results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate fold-level scores per parameter combination, ranked by the selection metric."""
    metric_columns = get_metric_columns(results)
    summary = results.groupby("params_json")[metric_columns].agg(["mean", "std"])
    summary = summary.sort_values((config.SELECTION_METRIC, "mean"), ascending=False)
    return summary


# --------------------------------------------------------------------------
# Persistence
# --------------------------------------------------------------------------


def save_results(model_name: str, results: pd.DataFrame) -> Path:
    """Write the fold-level scores so the five models' tables can be concatenated later."""
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    results_path = config.RESULTS_DIR / f"{model_name}.csv"
    results.to_csv(results_path, index=False)
    return results_path


def save_pipelines(model_name: str, pipelines: list) -> Path:
    """Persist the winning configuration's five fitted pipelines, which become the final ensemble."""
    config.PIPELINE_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path = config.PIPELINE_DIR / f"{model_name}.joblib"
    joblib.dump(pipelines, pipeline_path)
    return pipeline_path


# --------------------------------------------------------------------------
# Entry point for the per-model scripts
# --------------------------------------------------------------------------


def run_grid_search(
    model_name: str,
    estimator: BaseEstimator,
    param_grid: dict[str, list],
) -> pd.DataFrame:
    """Tune one algorithm on the shared folds and persist its scores and winning pipelines.

    Everything except the estimator and the grid is fixed here, so all five models stay comparable.
    """
    preprocessing.configure_logging(config.LOG_DIR / f"tuning_{model_name}.log")

    features, labels, groups = load_training_data()
    folds = grouping.build_cv_folds(features, labels, groups)
    combinations = expand_param_grid(param_grid)

    logging.info("Grid search: %s", model_name)
    logging.info("=" * 60)
    logging.info(
        "  scaling: %s (from config.yaml)", preprocessing.needs_scaling(model_name)
    )
    logging.info(
        "  %d combinations x %d folds = %d fits",
        len(combinations),
        len(folds),
        len(combinations) * len(folds),
    )
    logging.info("")

    all_rows = []
    pipelines_by_combination = {}
    started_at = time.perf_counter()

    for combination in combinations:
        fold_rows, fitted_pipelines = evaluate_combination(
            model_name, estimator, combination, features, labels, folds
        )
        all_rows.extend(fold_rows)
        pipelines_by_combination[fold_rows[0]["params_json"]] = fitted_pipelines

    elapsed_seconds = time.perf_counter() - started_at
    results = pd.DataFrame(all_rows)
    summary = summarize_results(results)

    best_parameters_json = summary.index[0]
    best_mean = summary.loc[best_parameters_json, (config.SELECTION_METRIC, "mean")]
    best_std = summary.loc[best_parameters_json, (config.SELECTION_METRIC, "std")]

    results_path = save_results(model_name, results)
    pipeline_path = save_pipelines(
        model_name, pipelines_by_combination[best_parameters_json]
    )

    logging.info("  top configurations by mean %s:", config.SELECTION_METRIC)
    for parameters_json, row in summary.head(5).iterrows():
        logging.info(
            "    %.3f +/- %.3f  %s",
            row[(config.SELECTION_METRIC, "mean")],
            row[(config.SELECTION_METRIC, "std")],
            json.loads(parameters_json),
        )
    logging.info("")
    logging.info("  selected: %s", json.loads(best_parameters_json))
    logging.info("  mean %s: %.3f +/- %.3f", config.SELECTION_METRIC, best_mean, best_std)
    logging.info("")

    winner_rows = results[results["params_json"] == best_parameters_json]
    metric_columns = get_metric_columns(results)
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
    logging.info("  %d fits in %.1fs", len(results), elapsed_seconds)
    logging.info("  scores    -> %s", results_path)
    logging.info("  pipelines -> %s", pipeline_path)

    return results
