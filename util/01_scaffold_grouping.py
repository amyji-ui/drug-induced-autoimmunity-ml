"""Murcko scaffold grouping and the shared cross-validation folds.

Nothing here runs on import. Every later script imports build_scaffold_groups and
build_cv_folds from this module so that all five models are tuned on identical folds.
"""

from __future__ import annotations

import importlib
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config

RDLogger.DisableLog("rdApp.*")
_fold_cache = {}


def compute_scaffold(smiles_string: str) -> str:
    """Return the Murcko scaffold SMILES, or an empty string for an acyclic or unparseable molecule."""
    molecule = Chem.MolFromSmiles(smiles_string)
    if molecule is None:
        return ""
    return MurckoScaffold.MurckoScaffoldSmiles(mol=molecule)


def build_scaffold_groups(smiles: pd.Series) -> np.ndarray:
    """Map each molecule to its CV group key, giving every acyclic drug a group of its own.

    Acyclic drugs share only the absence of rings, so they are not structural analogs
    and binding them into one group would create a block worth 45% of a fold.
    """
    group_keys = []
    for smiles_string in smiles:
        scaffold = compute_scaffold(smiles_string)
        if scaffold:
            group_keys.append(scaffold)
        else:
            group_keys.append(f"{config.ACYCLIC_PREFIX}{smiles_string}")
    return np.asarray(group_keys)


def build_cv_folds(
    features: pd.DataFrame, labels: pd.Series, groups: np.ndarray
) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build once and reuse persisted folds, rejecting changed data or split settings."""
    digest = hashlib.sha256()
    for values in (features, labels, pd.Series(groups)):
        digest.update(pd.util.hash_pandas_object(values, index=True).to_numpy().tobytes())
    digest.update(json.dumps(list(features.columns)).encode())
    metadata = {
        "data_sha256": digest.hexdigest(),
        "n_rows": len(features),
        "n_splits": config.N_SPLITS,
        "random_seed": config.RANDOM_SEED,
    }
    cache_key = (str(config.CV_FOLDS_FILE), json.dumps(metadata, sort_keys=True))
    if cache_key in _fold_cache:
        return _fold_cache[cache_key]

    if config.CV_FOLDS_FILE.exists():
        saved = json.loads(config.CV_FOLDS_FILE.read_text())
        if saved["metadata"] != metadata:
            raise ValueError(
                "Saved CV folds do not match the data/order or split settings; "
                "use a new paths.cv_folds_file for an intentional new split."
            )
        validation_folds = saved["validation_indices"]
    else:
        splitter = StratifiedGroupKFold(
            n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED
        )
        validation_folds = [
            validation_index.tolist()
            for _, validation_index in splitter.split(features, labels, groups)
        ]

    flat_indices = [index for indices in validation_folds for index in indices]
    if (
        len(validation_folds) != config.N_SPLITS
        or any(type(index) is not int for index in flat_indices)
        or sorted(flat_indices) != list(range(len(features)))
    ):
        raise ValueError("CV folds must cover each row exactly once")

    row_indices = np.arange(len(features))
    folds = []
    for indices in validation_folds:
        validation_index = np.asarray(indices, dtype=int)
        train_index = np.setdiff1d(row_indices, validation_index)
        if set(groups[train_index]) & set(groups[validation_index]):
            raise ValueError("CV folds contain scaffold leakage")
        if labels.iloc[train_index].nunique() != 2 or labels.iloc[validation_index].nunique() != 2:
            raise ValueError("Each training and validation fold must contain both classes")
        folds.append((train_index, validation_index))

    if not config.CV_FOLDS_FILE.exists():
        config.CV_FOLDS_FILE.parent.mkdir(parents=True, exist_ok=True)
        saved = {"metadata": metadata, "validation_indices": validation_folds}
        config.CV_FOLDS_FILE.write_text(json.dumps(saved, indent=2) + "\n")
    _fold_cache[cache_key] = folds
    return folds


def find_scaffold_overlap(
    train_groups: np.ndarray, test_groups: np.ndarray
) -> list[str]:
    """List scaffolds that appear in both the provided training and test splits."""
    shared = set(train_groups) & set(test_groups)
    return sorted(
        group for group in shared if not group.startswith(config.ACYCLIC_PREFIX)
    )
