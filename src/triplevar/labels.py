"""Clinical-significance labels: normalisation, numeric codes and groupings.

Numeric codes (as in the published model):
    -2 pathogenic, -1 likely pathogenic, 0 VUS (never trained on),
     1 likely benign, 2 benign, 3 "normal" (wild-type reference row).
"""

from __future__ import annotations

from typing import Dict, Iterable, List, Optional

import numpy as np
import pandas as pd

PATHOGENIC, LIKELY_PATHOGENIC, VUS, LIKELY_BENIGN, BENIGN, NORMAL = -2, -1, 0, 1, 2, 3

CODE_BY_NAME: Dict[str, int] = {
    "pathogenic": PATHOGENIC,
    "likely-pathogenic": LIKELY_PATHOGENIC,
    "uncertain-significance": VUS,
    "likely-benign": LIKELY_BENIGN,
    "benign": BENIGN,
    "normal": NORMAL,
}

DISPLAY_NAME: Dict[int, str] = {
    PATHOGENIC: "Pathogenic",
    LIKELY_PATHOGENIC: "Likely Pathogenic",
    VUS: "Uncertain Significance",
    LIKELY_BENIGN: "Likely Benign",
    BENIGN: "Benign",
    NORMAL: "Normal",
}

# Classes the classifier is trained on, in LabelEncoder (sorted) order.
TRAINING_CLASSES: List[int] = [PATHOGENIC, LIKELY_PATHOGENIC, LIKELY_BENIGN, BENIGN, NORMAL]

# dbSNP exports concatenate submitter labels; these collapse them to one label.
# Copied verbatim from the training notebook that produced the published model.
COMPOSITE_LABELS: Dict[str, str] = {
    "likely-benign;benign-likely-benign": "likely-benign",
    "likely-pathogenic;pathogenic": "likely-pathogenic",
    "uncertain-significance;likely-pathogenic": "likely-pathogenic",
    "likely-pathogenic;uncertain-significance": "likely-pathogenic",
    "likely-benign;uncertain-significance": "likely-benign",
    "likely-benign;conflicting-interpretations-of-pathogenicity": "likely-benign",
    "pathogenic;pathogenic-likely-pathogenic": "pathogenic",
    "conflicting-interpretations-of-pathogenicity": "uncertain-significance",
    "uncertain-significance;likely-benign": "likely-benign",
    "conflicting-interpretations-of-pathogenicity;uncertain-significance": "uncertain-significance",
    "conflicting-interpretations-of-pathogenicity;benign-likely-benign": "likely-benign",
    "uncertain-significance;likely-benign;conflicting-interpretations-of-pathogenicity;benign": "likely-benign",
    "conflicting-interpretations-of-pathogenicity;likely-benign": "likely-benign",
    "benign-likely-benign;conflicting-interpretations-of-pathogenicity;uncertain-significance": "likely-benign",
    "benign-likely-benign;benign": "likely-benign",
    "benign;likely-benign": "likely-benign",
    "benign;likely-benign;conflicting-interpretations-of-pathogenicity": "likely-benign",
    "benign;likely-benign;conflicting-interpretations-of-pathogenicity;uncertain-significance": "likely-benign",
    "benign;likely-benign;conflicting-interpretations-of-pathogenicity;benign-likely-benign": "likely-benign",
    "benign;likely-benign;conflicting-interpretations-of-pathogenicity;benign": "likely-benign",
    "uncertain-significance;pathogenic": "pathogenic",
    "likely-pathogenic;pathogenic-likely-pathogenic": "likely-pathogenic",
    "likely-pathogenic;benign": "uncertain-significance",
    "uncertain-significance;not-provided": "uncertain-significance",
    "uncertain-significance;benign-likely-benign": "likely-benign",
    "likely-benign;uncertain-significance;benign": "likely-benign",
    "benign-likely-benign;uncertain-significance;benign": "likely-benign",
    "pathogenic-likely-pathogenic;likely-pathogenic": "likely-pathogenic",
    "benign;uncertain-significance;likely-benign": "likely-benign",
    "benign;likely-benign;uncertain-significance": "likely-benign",
    "conflicting-interpretations-of-pathogenicity;uncertain-significance;likely-benign": "likely-benign",
    "pathogenic;likely-pathogenic": "likely-pathogenic",
    "likely-benign;benign": "likely-benign",
    "pathogenic;uncertain-significance": "likely-pathogenic",
    "benign;uncertain-significance": "likely-benign",
    "uncertain-significance;likely-benign;benign": "likely-benign",
    "pathogenic-likely-pathogenic": "likely-pathogenic",
    # The notebook also had a " pathogenic-..." key (leading space) mapping to
    # "pathogenic"; labels are stripped first, so only this entry ever applied.
    "pathogenic-likely-pathogenic;pathogenic": "likely-pathogenic",
    "likely-benign;pathogenic": "uncertain-significance",
    "uncertain-significance;conflicting-interpretations-of-pathogenicity": "uncertain-significance",
    "likely-benign;conflicting-interpretations-of-pathogenicity;uncertain-significance": "likely-benign",
    "benign-likely-benign;uncertain-significance": "likely-benign",
    "pathogenic-likely-pathogenic;likely-benign;pathogenic": "likely-pathogenic",
    "benign;benign-likely-benign": "likely-benign",
    "benign-likely-benign;likely-benign": "likely-benign",
    "pathogenic;likely-benign": "uncertain-significance",
    "likely-pathogenic;benign-likely-benign;uncertain-significance": "uncertain-significance",
    "benign;conflicting-interpretations-of-pathogenicity;likely-benign": "likely-benign",
    "conflicting-interpretations-of-pathogenicity;benign-likely-benign;likely-benign": "likely-benign",
    "conflicting-interpretations-of-pathogenicity;likely-benign;uncertain-significance": "uncertain-significance",
}


def normalise_label(raw: Optional[str]) -> Optional[str]:
    """Collapse a raw dbSNP clinical-significance string to a single label (None if unlabeled)."""
    if raw is None or (isinstance(raw, float) and np.isnan(raw)):
        return None
    label = str(raw).strip()
    if label in ("", "nan"):
        return None
    return COMPOSITE_LABELS.get(label, label)


def to_code(labels: Iterable[Optional[str]]) -> np.ndarray:
    """Map normalised labels to numeric codes; unknown/unlabeled become NaN."""
    return np.array([CODE_BY_NAME.get(l, np.nan) if l is not None else np.nan for l in labels], dtype=float)


def is_pathogenic(codes) -> np.ndarray:
    return np.isin(np.asarray(codes), [PATHOGENIC, LIKELY_PATHOGENIC])


def is_benign(codes) -> np.ndarray:
    return np.isin(np.asarray(codes), [LIKELY_BENIGN, BENIGN])


def group3(codes) -> np.ndarray:
    """0 = benign/likely benign, 1 = pathogenic/likely pathogenic, 2 = normal."""
    codes = np.asarray(codes)
    return np.where(is_benign(codes), 0, np.where(is_pathogenic(codes), 1, 2))


def display(codes) -> List[str]:
    return [DISPLAY_NAME.get(int(c), "Unknown") for c in np.asarray(codes)]


def label_series(raw: pd.Series) -> pd.Series:
    return raw.map(normalise_label)
