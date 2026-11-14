"""Report on Murcko scaffold groups and the composition of the cross-validation folds.

Running this file writes logs/grouping_report.log.
Scaffolds, groups, and folds all come from 01_scaffold_grouping.py.
"""

from __future__ import annotations

import importlib
import logging
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

preprocessing = importlib.import_module("00_preprocessing")
grouping = importlib.import_module("01_scaffold_grouping")


def report_groups(name: str, smiles: pd.Series, groups: np.ndarray) -> None:
    """Log how many scaffolds one split has and how large the biggest groups are."""
    scaffolds = [grouping.compute_scaffold(smiles_string) for smiles_string in smiles]
    n_acyclic = sum(1 for scaffold in scaffolds if not scaffold)
    group_sizes = Counter(groups)

    logging.info("[%s]", name)
    logging.info("  molecules: %d", len(smiles))
    logging.info("  distinct Murcko scaffolds: %d", len(set(scaffolds)))
    logging.info("  acyclic molecules (empty scaffold): %d", n_acyclic)
    logging.info("  CV groups after the acyclic split: %d", len(group_sizes))
    logging.info("  largest groups:")
    for group_key, size in group_sizes.most_common(5):
        label = (
            group_key
            if not group_key.startswith(config.ACYCLIC_PREFIX)
            else "<acyclic>"
        )
        logging.info("    %3d molecules  %s", size, label)
    logging.info("")


def report_folds(
    labels: pd.Series,
    groups: np.ndarray,
    folds: list[tuple[np.ndarray, np.ndarray]],
) -> None:
    """Log the size, class balance, and group isolation of each validation fold."""
    logging.info("[cross-validation folds]")
    logging.info(
        "  StratifiedGroupKFold(n_splits=%d, shuffle=True, random_state=%d)",
        config.N_SPLITS,
        config.RANDOM_SEED,
    )

    for fold_index, (train_index, validation_index) in enumerate(folds):
        validation_labels = labels.iloc[validation_index]
        positive_rate = validation_labels.mean()
        leaked_groups = set(groups[train_index]) & set(groups[validation_index])
        largest_group = Counter(groups[validation_index]).most_common(1)[0][1]

        logging.info(
            "  fold %d: n=%3d  positive=%5.1f%%  groups leaked to train=%d  largest group in fold=%d",
            fold_index,
            len(validation_index),
            100 * positive_rate,
            len(leaked_groups),
            largest_group,
        )

    logging.info("  overall positive rate: %.1f%%", 100 * labels.mean())
    logging.info("")


def report_train_test_overlap(
    train_groups: np.ndarray, test_groups: np.ndarray
) -> None:
    """Log scaffolds shared by the provided training and test splits, which the proposal asks for."""
    shared_scaffolds = grouping.find_scaffold_overlap(train_groups, test_groups)

    logging.info("[scaffold overlap: provided train vs provided test]")
    logging.info("  shared scaffolds: %d", len(shared_scaffolds))
    for scaffold in shared_scaffolds[:config.N_OVERLAP_EXAMPLES]:
        logging.info("    %s", scaffold)
    if len(shared_scaffolds) > config.N_OVERLAP_EXAMPLES:
        logging.info("    ... and %d more", len(shared_scaffolds) - config.N_OVERLAP_EXAMPLES)
    logging.info(
        "  note: overlap means some test molecules share a core with training molecules,"
    )
    logging.info(
        "        so the held-out test estimate is easier than a fully novel-scaffold screen."
    )
    logging.info("")


def main() -> None:
    """Write the scaffold grouping and fold composition report."""
    preprocessing.configure_logging(config.GROUPING_REPORT_LOG)

    train_df = preprocessing.load_split(config.TRAIN_FILE)
    test_df = preprocessing.load_split(config.TEST_FILE)
    train_features, train_labels, train_smiles = preprocessing.split_features_labels(
        train_df
    )
    _, _, test_smiles = preprocessing.split_features_labels(test_df)

    train_groups = grouping.build_scaffold_groups(train_smiles)
    test_groups = grouping.build_scaffold_groups(test_smiles)
    folds = grouping.build_cv_folds(train_features, train_labels, train_groups)

    logging.info("Scaffold grouping and cross-validation folds")
    logging.info("=" * 60)
    logging.info("")

    report_groups("train", train_smiles, train_groups)
    report_groups("test", test_smiles, test_groups)
    report_folds(train_labels, train_groups, folds)
    report_train_test_overlap(train_groups, test_groups)

    logging.info("report written to %s", config.GROUPING_REPORT_LOG)


if __name__ == "__main__":
    main()
