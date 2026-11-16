"""Loader for config.yaml, which holds every setting in the project.

Scripts do `import config` and read the names below; edit config.yaml, never this file.
Paths from the YAML are resolved against the repo root and returned as Path objects.
"""

from __future__ import annotations

from pathlib import Path

import yaml

PROJECT_DIR = Path(__file__).resolve().parents[1]
CONFIG_FILE = PROJECT_DIR / "config.yaml"

with CONFIG_FILE.open() as config_stream:
    _settings = yaml.safe_load(config_stream)


def _resolve(relative_path: str) -> Path:
    """Turn a repo-relative path from the YAML into an absolute Path."""
    return PROJECT_DIR / relative_path


_paths = _settings["paths"]
DATA_DIR = _resolve(_paths["data_dir"])
LOG_DIR = _resolve(_paths["log_dir"])
TRAIN_FILE = _resolve(_paths["train_file"])
TEST_FILE = _resolve(_paths["test_file"])
DESCRIPTOR_DICTIONARY_FILE = _resolve(_paths["descriptor_dictionary_file"])
RESULTS_DIR = _resolve(_paths["results_dir"])
PIPELINE_DIR = _resolve(_paths["pipeline_dir"])
CV_FOLDS_FILE = _resolve(_paths["cv_folds_file"])

_columns = _settings["columns"]
LABEL_COLUMN = _columns["label"]
SMILES_COLUMN = _columns["smiles"]
POSITIVE_LABEL = _columns["positive_label"]

RANDOM_SEED = _settings["random_seed"]

MODELS = _settings["models"]
MODEL_NAMES = tuple(MODELS)
SCALED_MODELS = tuple(name for name, spec in MODELS.items() if spec["scale"])

_preprocessing = _settings["preprocessing"]
HEAVY_TAILED_COLUMNS = tuple(_preprocessing["heavy_tailed_columns"])
CORRELATION_THRESHOLD = _preprocessing["correlation_threshold"]

_cross_validation = _settings["cross_validation"]
N_SPLITS = _cross_validation["n_splits"]
ACYCLIC_PREFIX = _cross_validation["acyclic_prefix"]

_evaluation = _settings["evaluation"]
DECISION_THRESHOLD = _evaluation["decision_threshold"]
SELECTION_METRIC = _evaluation["selection_metric"]

_reports = _settings["reports"]
DATA_REPORT_LOG = _resolve(_reports["data_report_log"])
GROUPING_REPORT_LOG = _resolve(_reports["grouping_report_log"])
REDUNDANCY_THRESHOLDS = tuple(_reports["redundancy_thresholds"])
EXAMPLE_THRESHOLD = _reports["example_threshold"]
N_REDUNDANCY_EXAMPLES = _reports["n_redundancy_examples"]
N_OVERLAP_EXAMPLES = _reports["n_overlap_examples"]
