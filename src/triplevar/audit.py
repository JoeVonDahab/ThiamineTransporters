"""Training-label audit: which labeled rows carry a label of their own allele.

dbSNP exports report clinical significance per rsID, so the published data
copied one label to every allele of a multi-allelic rsID (e.g. a nonsense
allele inheriting 'likely benign' from a synonymous sibling).
``validation/external/training_label_audit.csv`` records, per labeled row,
whether its own allele had a ClinVar submission on/before the training cutoff
(2025-06-27) and whether that agrees with the training label.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple, Union

import numpy as np
import pandas as pd

from .dataset import FeatureSet
from .labels import NORMAL

KEY = ["snp_id", "pos", "original", "mutant"]


def _keys(df: pd.DataFrame):
    return list(zip(*(df[c].astype(str).str.replace(r"\.0$", "", regex=True) for c in KEY)))


def label_source(labeled: FeatureSet, audit_csv: Union[str, Path]) -> np.ndarray:
    """Per labeled row: 'own_allele', 'copied_from_sibling_allele', 'disagrees_with_own_allele',
    'own_allele_classification_unknown_at_cutoff', 'normal' (synthetic wild-type rows) or 'unaudited'."""
    audit = pd.read_csv(audit_csv)
    lookup = dict(zip(_keys(audit), audit["label_source"]))
    src = np.array([lookup.get(k, "unaudited") for k in _keys(labeled.meta)], dtype=object)
    src[labeled.y == NORMAL] = "normal"
    return src


def masks(labeled: FeatureSet, audit_csv: Union[str, Path]) -> Tuple[np.ndarray, np.ndarray]:
    """(drop_from_training, trusted_for_evaluation) boolean masks.

    Copied and disagreeing labels are dropped from training; evaluation on the
    trusted subset additionally leaves out rows whose own-allele classification
    at the cutoff cannot be reconstructed."""
    src = label_source(labeled, audit_csv)
    untrusted = np.isin(src, ["copied_from_sibling_allele", "disagrees_with_own_allele"])
    trusted = np.isin(src, ["own_allele", "normal"])
    return untrusted, trusted
