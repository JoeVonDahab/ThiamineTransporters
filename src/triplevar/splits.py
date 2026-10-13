"""Leakage-safe data splits.

The published split was stratified by *row*: alleles of the same rsID and the
paired wild-type ('normal') row could land on different sides. Here every rsID
is one group, so all of its rows stay in the same fold.
"""

from __future__ import annotations

from typing import Iterator, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from .labels import NORMAL, VUS


def group_strata(snp_ids: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Per-row stratum = the most severe variant label of the row's rsID."""
    df = pd.DataFrame({"g": snp_ids, "y": y})
    variant = df[~df["y"].isin([NORMAL, VUS])]
    worst = variant.groupby("g")["y"].min()  # -2 is most severe
    return df["g"].map(worst).fillna(NORMAL).to_numpy()


def grouped_folds(
    snp_ids: np.ndarray, y: np.ndarray, n_splits: int = 5, seed: int = 42
) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
    """StratifiedGroupKFold over rsIDs; yields (train_idx, test_idx) row indices."""
    strata = group_strata(snp_ids, y)
    skf = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    yield from skf.split(np.zeros(len(y)), strata, groups=snp_ids)


def grouped_train_val_test(
    snp_ids: np.ndarray, y: np.ndarray, seed: int = 42, n_splits: int = 7
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """One grouped ~70/15/15 split: one fold is test, one of the rest is validation."""
    folds = list(grouped_folds(snp_ids, y, n_splits=n_splits, seed=seed))
    test = folds[0][1]
    val = folds[1][1]
    train = np.setdiff1d(np.arange(len(y)), np.concatenate([test, val]))
    return train, val, test
