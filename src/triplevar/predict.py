"""Turn model probabilities into a per-variant prediction table."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .labels import display
from .metrics import Scores


def confidence_band(p: np.ndarray) -> np.ndarray:
    return np.where(p >= 0.8, "High", np.where(p >= 0.6, "Medium", "Low"))


def prediction_table(bundle, X: np.ndarray, meta: pd.DataFrame, device: str = "cpu") -> pd.DataFrame:
    """One row per variant: 5-class call, class probabilities and a pathogenicity score.

    ``p_pathogenic`` is P(pathogenic or likely pathogenic) renormalised over the
    four variant classes, i.e. ignoring the synthetic 'normal' class.
    """
    probs = bundle.predict_proba(X, device=device)
    s = Scores(probs, bundle.classes)
    out = meta.reset_index(drop=True).copy()
    out["predicted_code"] = s.pred
    out["predicted_class"] = display(s.pred)
    out["confidence"] = probs.max(1)
    out["confidence_band"] = confidence_band(out["confidence"].to_numpy())
    for code, col in zip(bundle.classes, ["p_P", "p_LP", "p_LB", "p_B", "p_normal"]):
        out[col] = probs[:, bundle.classes.index(code)]
    out["p_pathogenic"] = s.pathogenic_given_variant
    out["pathogenic_call"] = s.call_grouped()
    return out
