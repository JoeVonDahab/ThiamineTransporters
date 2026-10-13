"""Evaluation metrics, with every definition used in the abstract made explicit.

``probs`` is always an (n, 5) softmax matrix in ``classes`` order
([-2, -1, 1, 2, 3]); ``y_true`` holds numeric label codes.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Dict, List, Optional, Sequence

import numpy as np
from sklearn.metrics import accuracy_score, average_precision_score, roc_auc_score

from .labels import (
    BENIGN,
    LIKELY_BENIGN,
    LIKELY_PATHOGENIC,
    NORMAL,
    PATHOGENIC,
    TRAINING_CLASSES,
    group3,
)


@dataclass
class BinaryReport:
    n: int
    tp: int
    fn: int
    fp: int
    tn: int
    accuracy: float
    precision: float
    recall: float
    specificity: float
    f1: float
    auc: Optional[float]

    def as_dict(self) -> Dict[str, float]:
        return asdict(self)

    def short(self) -> str:
        auc = "n/a" if self.auc is None else f"{self.auc:.3f}"
        return (
            f"n={self.n} TP={self.tp} FN={self.fn} FP={self.fp} TN={self.tn} | acc={self.accuracy:.3f} "
            f"prec={self.precision:.3f} rec={self.recall:.3f} spec={self.specificity:.3f} "
            f"F1={self.f1:.3f} AUC={auc}"
        )


def binary_report(y_pathogenic: np.ndarray, pred_pathogenic: np.ndarray, score: Optional[np.ndarray] = None) -> BinaryReport:
    y = np.asarray(y_pathogenic, dtype=bool)
    p = np.asarray(pred_pathogenic, dtype=bool)
    tp, fn = int((y & p).sum()), int((y & ~p).sum())
    fp, tn = int((~y & p).sum()), int((~y & ~p).sum())

    def div(a: float, b: float) -> float:
        return float(a / b) if b else 0.0

    prec, rec = div(tp, tp + fp), div(tp, tp + fn)
    auc = None
    if score is not None and 0 < y.sum() < len(y):
        auc = float(roc_auc_score(y, score))
    return BinaryReport(
        n=len(y), tp=tp, fn=fn, fp=fp, tn=tn,
        accuracy=div(tp + tn, len(y)), precision=prec, recall=rec,
        specificity=div(tn, tn + fp), f1=div(2 * prec * rec, prec + rec), auc=auc,
    )


class Scores:
    """Derived probabilities and calls from a 5-class softmax matrix."""

    def __init__(self, probs: np.ndarray, classes: Sequence[int] = TRAINING_CLASSES):
        self.probs = np.asarray(probs, dtype=np.float64)
        self.classes = list(classes)
        idx = {c: self.classes.index(c) for c in self.classes}
        self.p_path = self.probs[:, idx[PATHOGENIC]]
        self.p_lpath = self.probs[:, idx[LIKELY_PATHOGENIC]]
        self.p_lbenign = self.probs[:, idx[LIKELY_BENIGN]]
        self.p_benign = self.probs[:, idx[BENIGN]]
        self.p_normal = self.probs[:, idx[NORMAL]]
        self.pred = np.asarray(self.classes)[self.probs.argmax(1)]

    @property
    def pathogenic_sum(self) -> np.ndarray:
        """P(pathogenic) + P(likely pathogenic)."""
        return self.p_path + self.p_lpath

    @property
    def pathogenic_given_variant(self) -> np.ndarray:
        """P(P or LP) renormalised over the four variant classes (excludes 'normal')."""
        variant_mass = self.pathogenic_sum + self.p_lbenign + self.p_benign
        return self.pathogenic_sum / np.clip(variant_mass, 1e-12, None)

    @property
    def strict_pathogenic(self) -> np.ndarray:
        """P(-2) / (P(-2) + P(2)): the 'binary probability' written by the notebooks."""
        return self.p_path / np.clip(self.p_path + self.p_benign, 1e-12, None)

    def call_grouped(self) -> np.ndarray:
        """Pathogenic call from the 5-class argmax; a 'normal' argmax falls back to
        comparing pathogenic vs benign probability mass (notebook cell 61)."""
        benign_mass = self.p_lbenign + self.p_benign
        return np.where(
            np.isin(self.pred, [PATHOGENIC, LIKELY_PATHOGENIC]),
            True,
            np.where(np.isin(self.pred, [LIKELY_BENIGN, BENIGN]), False, self.pathogenic_sum > benign_mass),
        )


def evaluate(probs: np.ndarray, y_true: np.ndarray, classes: Sequence[int] = TRAINING_CLASSES) -> Dict[str, object]:
    """All metrics reported for a ClinVar-style labelled set."""
    s = Scores(probs, classes)
    y = np.asarray(y_true)
    out: Dict[str, object] = {"n": int(len(y))}
    out["accuracy_5class"] = float(accuracy_score(y, s.pred))
    out["accuracy_grouped3"] = float(accuracy_score(group3(y), group3(s.pred)))

    strict = np.isin(y, [PATHOGENIC, BENIGN])
    if strict.any():
        out["strict_P_vs_B"] = binary_report(
            y[strict] == PATHOGENIC, s.pred[strict] == PATHOGENIC, s.p_path[strict]
        ).as_dict()

    variants = np.isin(y, [PATHOGENIC, LIKELY_PATHOGENIC, LIKELY_BENIGN, BENIGN])
    if variants.any():
        y_path = np.isin(y[variants], [PATHOGENIC, LIKELY_PATHOGENIC])
        rep = binary_report(y_path, s.call_grouped()[variants], s.pathogenic_sum[variants])
        out["PLP_vs_BLB"] = rep.as_dict()
        if 0 < y_path.sum() < len(y_path):
            out["PLP_vs_BLB"]["auc_renormalised"] = float(roc_auc_score(y_path, s.pathogenic_given_variant[variants]))
            out["PLP_vs_BLB"]["pr_auc"] = float(average_precision_score(y_path, s.pathogenic_sum[variants]))
    return out


def evaluate_external(
    probs: np.ndarray, y_pathogenic: np.ndarray, classes: Sequence[int] = TRAINING_CLASSES
) -> Dict[str, Dict[str, float]]:
    """Binary external validation under each pathogenic-call rule."""
    s = Scores(probs, classes)
    y = np.asarray(y_pathogenic, dtype=bool)
    return {
        "grouped_argmax": binary_report(y, s.call_grouped(), s.pathogenic_given_variant).as_dict(),
        "strict_ratio_gt_0.5": binary_report(y, s.strict_pathogenic > 0.5, s.strict_pathogenic).as_dict(),
        "renormalised_gt_0.5": binary_report(y, s.pathogenic_given_variant > 0.5, s.pathogenic_given_variant).as_dict(),
    }


def bootstrap_ci(
    y: np.ndarray, score: np.ndarray, metric=roc_auc_score, n_boot: int = 2000, seed: int = 0, alpha: float = 0.05
) -> List[float]:
    """Percentile bootstrap CI for a score-based metric (resamples rows)."""
    rng = np.random.default_rng(seed)
    y, score = np.asarray(y), np.asarray(score)
    vals = []
    for _ in range(n_boot):
        i = rng.integers(0, len(y), len(y))
        if 0 < y[i].sum() < len(i):
            vals.append(metric(y[i], score[i]))
    if not vals:
        return [float("nan"), float("nan")]
    return [float(np.quantile(vals, alpha / 2)), float(np.quantile(vals, 1 - alpha / 2))]
