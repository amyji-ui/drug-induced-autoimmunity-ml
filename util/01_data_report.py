"""Data exploration report for the DIA prediction project.

Running this file writes class balance, shape, and quality checks to logs/01_data_report.log.
Paths and data loading come from 00_preprocessing.py.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

sys.path.insert(0, str(Path(__file__).resolve().parent))
preprocessing = importlib.import_module("00_preprocessing")

LOG_FILE = preprocessing.LOG_DIR / "01_data_report.log"

# Thresholds to compare, and the one whose dropped pairs are listed in full.
REDUNDANCY_THRESHOLDS = (0.99, 0.95, 0.90)
EXAMPLE_THRESHOLD = 0.95
N_EXAMPLES = 8


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------


def summarize_shape(dataframe: pd.DataFrame) -> dict[str, int]:
    """Count rows, total columns, and descriptor columns for one split."""
    descriptor_columns = preprocessing.get_descriptor_columns(dataframe)
    return {
        "n_rows": int(dataframe.shape[0]),
        "n_columns": int(dataframe.shape[1]),
        "n_descriptors": len(descriptor_columns),
    }


def summarize_class_balance(labels: pd.Series) -> dict[str, float]:
    """Count the two classes and the positive rate for one split."""
    label_counts = labels.value_counts()
    n_negative = int(label_counts.get(0, 0))
    n_positive = int(label_counts.get(preprocessing.POSITIVE_LABEL, 0))
    n_total = n_negative + n_positive
    positive_rate = n_positive / n_total if n_total else float("nan")
    return {
        "n_negative": n_negative,
        "n_positive": n_positive,
        "positive_rate": positive_rate,
    }


def find_constant_columns(features: pd.DataFrame) -> list[str]:
    """List descriptors that take a single value across the split."""
    unique_counts = features.nunique(dropna=False)
    constant_columns = unique_counts[unique_counts <= 1].index.tolist()
    return constant_columns


def count_non_finite(features: pd.DataFrame) -> int:
    """Count NaN and infinite descriptor cells."""
    numeric_values = features.to_numpy(dtype=float)
    non_finite_mask = ~np.isfinite(numeric_values)
    return int(non_finite_mask.sum())


def find_duplicate_smiles(smiles: pd.Series) -> list[str]:
    """List SMILES strings that appear more than once within one split."""
    duplicate_mask = smiles.duplicated(keep=False)
    return sorted(smiles[duplicate_mask].unique().tolist())


def find_invalid_smiles(smiles: pd.Series) -> list[str] | None:
    """Return SMILES that RDKit cannot parse, or None when RDKit is unavailable."""
    try:
        from rdkit import Chem
        from rdkit import RDLogger
    except ImportError:
        return None

    RDLogger.DisableLog("rdApp.*")
    invalid_smiles = [
        smiles_string
        for smiles_string in smiles
        if Chem.MolFromSmiles(smiles_string) is None
    ]
    return invalid_smiles


# --------------------------------------------------------------------------
# Report
# --------------------------------------------------------------------------


def configure_logging(log_path: Path) -> None:
    """Send the report to both the log file and the console, overwriting any previous run."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
        handlers=[
            logging.FileHandler(log_path, mode="w"),
            logging.StreamHandler(),
        ],
        force=True,
    )


def report_split(name: str, dataframe: pd.DataFrame) -> list[str]:
    """Log the shape, class balance, and quality checks for one split, returning its constant columns."""
    features, labels, smiles = preprocessing.split_features_labels(dataframe)
    shape_summary = summarize_shape(dataframe)
    balance_summary = summarize_class_balance(labels)
    constant_columns = find_constant_columns(features)
    n_non_finite = count_non_finite(features)
    duplicate_smiles = find_duplicate_smiles(smiles)
    invalid_smiles = find_invalid_smiles(smiles)

    logging.info("[%s]", name)
    logging.info(
        "  rows: %d | columns: %d | descriptors: %d",
        shape_summary["n_rows"],
        shape_summary["n_columns"],
        shape_summary["n_descriptors"],
    )
    logging.info(
        "  class balance: %d negative (0), %d positive (1), positive rate %.1f%%",
        balance_summary["n_negative"],
        balance_summary["n_positive"],
        100 * balance_summary["positive_rate"],
    )
    logging.info("  non-finite descriptor cells: %d", n_non_finite)
    logging.info("  duplicate SMILES: %d", len(duplicate_smiles))

    if invalid_smiles is None:
        logging.info("  invalid SMILES: not checked (RDKit not installed)")
    else:
        logging.info("  invalid SMILES: %d", len(invalid_smiles))
        for smiles_string in invalid_smiles:
            logging.info("    %s", smiles_string)

    logging.info("  constant descriptors: %d", len(constant_columns))
    for column in constant_columns:
        logging.info("    %s", column)
    logging.info("")

    return constant_columns


def report_overlap(train_df: pd.DataFrame, test_df: pd.DataFrame) -> None:
    """Log whether the two splits share column names or molecules."""
    columns_match = list(train_df.columns) == list(test_df.columns)
    train_smiles = set(train_df[preprocessing.SMILES_COLUMN])
    test_smiles = set(test_df[preprocessing.SMILES_COLUMN])
    shared_smiles = train_smiles & test_smiles

    logging.info("[train vs test]")
    logging.info("  identical column names and order: %s", columns_match)
    logging.info("  SMILES shared between splits: %d", len(shared_smiles))
    for smiles_string in sorted(shared_smiles):
        logging.info("    %s", smiles_string)
    logging.info("")


def report_constant_agreement(
    train_constant: list[str], test_constant: list[str]
) -> None:
    """Log how the constant descriptors differ between splits, since only the training fold may decide what is dropped."""
    train_only = sorted(set(train_constant) - set(test_constant))
    test_only = sorted(set(test_constant) - set(train_constant))

    logging.info("[constant descriptor agreement]")
    logging.info("  constant in train only: %d", len(train_only))
    for column in train_only:
        logging.info("    %s", column)
    logging.info("  constant in test only: %d", len(test_only))
    for column in test_only:
        logging.info("    %s", column)
    logging.info(
        "  note: this comparison is diagnostic only. The test set never decides which columns"
    )
    logging.info(
        "        are removed: VarianceThreshold learns the mask on each CV training fold and"
    )
    logging.info(
        "        replays it unchanged on validation and test data. A descriptor that is constant"
    )
    logging.info(
        "        in test only is therefore kept, but it takes one value across every test"
    )
    logging.info(
        "        molecule, so it cannot separate them -- keep that in mind when reading"
    )
    logging.info("        the permutation importance ranking.")
    logging.info("")


def summarize_redundancy(features: pd.DataFrame, threshold: float) -> dict:
    """Drop the constant descriptors, then fit the correlation filter and record what it removes.

    Standardization is skipped because Pearson correlation does not depend on scale.
    """
    variance_filter = VarianceThreshold(threshold=0.0).fit(features)
    varying_names = np.asarray(variance_filter.get_feature_names_out())
    varying_features = variance_filter.transform(features)

    correlation_filter = preprocessing.CorrelationFilter(threshold=threshold)
    correlation_filter.fit(varying_features)
    keep_mask = correlation_filter.support_mask_

    named_pairs = [
        (varying_names[dropped_index], varying_names[kept_index], correlation)
        for dropped_index, kept_index, correlation in correlation_filter.redundant_pairs_
    ]
    return {
        "n_varying": len(varying_names),
        "n_kept": int(keep_mask.sum()),
        "n_dropped": int((~keep_mask).sum()),
        "pairs": named_pairs,
    }


def report_redundancy(train_df: pd.DataFrame) -> None:
    """Log how many descriptors survive the correlation filter at several thresholds, using the training split only."""
    features, _, _ = preprocessing.split_features_labels(train_df)
    summaries = {
        threshold: summarize_redundancy(features, threshold)
        for threshold in REDUNDANCY_THRESHOLDS
    }

    logging.info("[redundancy among varying descriptors (train only)]")
    for threshold, summary in summaries.items():
        logging.info(
            "  |r| > %.2f : %3d kept, %3d dropped  (of %d varying descriptors)",
            threshold,
            summary["n_kept"],
            summary["n_dropped"],
            summary["n_varying"],
        )

    example_pairs = summaries[EXAMPLE_THRESHOLD]["pairs"]
    logging.info("  first %d drops at |r| > %.2f:", N_EXAMPLES, EXAMPLE_THRESHOLD)
    for dropped_name, kept_name, correlation in example_pairs[:N_EXAMPLES]:
        logging.info(
            "    %-22s displaced by %-22s |r| = %.4f",
            dropped_name,
            kept_name,
            correlation,
        )

    absorber_counts = Counter(kept_name for _, kept_name, _ in example_pairs)
    logging.info("  descriptors that displaced the most others:")
    for kept_name, count in absorber_counts.most_common(5):
        logging.info("    %-22s displaced %d", kept_name, count)

    logging.info(
        "  note: diagnostic only, computed on the whole training split. The filter is refit"
    )
    logging.info(
        "        inside every CV fold, so the exact survivors differ from fold to fold, and it"
    )
    logging.info(
        "        is off unless build_preprocessor is given a correlation_threshold."
    )
    logging.info("")


def main() -> None:
    """Write the dataset exploration report."""
    configure_logging(LOG_FILE)

    train_df = preprocessing.load_split(preprocessing.TRAIN_FILE)
    test_df = preprocessing.load_split(preprocessing.TEST_FILE)

    logging.info("DIA dataset exploration report")
    logging.info("=" * 60)
    logging.info(
        "constant = one value only, zero variance, carries no information at all;"
    )
    logging.info("           dropped by VarianceThreshold.")
    logging.info(
        "redundant = varies normally but duplicates another descriptor (high |r|);"
    )
    logging.info(
        "           dropped separately by CorrelationFilter, and not reported here."
    )
    logging.info("")

    train_constant = report_split("train", train_df)
    test_constant = report_split("test", test_df)
    report_overlap(train_df, test_df)
    report_constant_agreement(train_constant, test_constant)
    report_redundancy(train_df)

    logging.info("report written to %s", LOG_FILE)


if __name__ == "__main__":
    main()
