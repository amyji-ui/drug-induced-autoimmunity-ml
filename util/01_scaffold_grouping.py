"""Murcko scaffold grouping and the shared cross-validation folds.

Nothing here runs on import. Every later script imports build_scaffold_groups and
build_cv_folds from this module so that all five models are tuned on identical folds.
"""

from __future__ import annotations

import importlib
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
    """Build the 5 scaffold-grouped, label-stratified folds once, for every model to reuse.

    Materializing the folds means a later script cannot accidentally build a different splitter.
    """
    splitter = StratifiedGroupKFold(
        n_splits=config.N_SPLITS, shuffle=True, random_state=config.RANDOM_SEED
    )
    return list(splitter.split(features, labels, groups))


def find_scaffold_overlap(
    train_groups: np.ndarray, test_groups: np.ndarray
) -> list[str]:
    """List scaffolds that appear in both the provided training and test splits."""
    shared = set(train_groups) & set(test_groups)
    return sorted(
        group for group in shared if not group.startswith(config.ACYCLIC_PREFIX)
    )
