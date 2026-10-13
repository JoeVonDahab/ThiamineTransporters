import numpy as np
import pytest

from triplevar.labels import (
    BENIGN, LIKELY_BENIGN, LIKELY_PATHOGENIC, NORMAL, PATHOGENIC, group3, normalise_label, to_code,
)
from triplevar.metrics import Scores, binary_report, evaluate


def test_normalise_composite_labels():
    assert normalise_label(" pathogenic-likely-pathogenic;pathogenic ") == "likely-pathogenic"
    assert normalise_label("benign;likely-benign") == "likely-benign"
    assert normalise_label("conflicting-interpretations-of-pathogenicity") == "uncertain-significance"
    assert normalise_label(np.nan) is None
    assert normalise_label("pathogenic") == "pathogenic"


def test_codes_and_groups():
    codes = to_code(["pathogenic", "likely-benign", "normal", None])
    assert codes[:3].tolist() == [PATHOGENIC, LIKELY_BENIGN, NORMAL]
    assert np.isnan(codes[3])
    assert group3([PATHOGENIC, LIKELY_PATHOGENIC, LIKELY_BENIGN, BENIGN, NORMAL]).tolist() == [1, 1, 0, 0, 2]


def test_binary_report_counts():
    y = np.array([1, 1, 1, 0, 0], bool)
    p = np.array([1, 1, 0, 1, 0], bool)
    r = binary_report(y, p, score=np.array([0.9, 0.8, 0.3, 0.6, 0.1]))
    assert (r.tp, r.fn, r.fp, r.tn) == (2, 1, 1, 1)
    assert r.accuracy == pytest.approx(0.6)
    assert r.precision == pytest.approx(2 / 3)
    assert r.specificity == pytest.approx(0.5)
    assert r.auc == pytest.approx(5 / 6)


def test_scores_renormalise_ignores_normal_class():
    # classes: [-2, -1, 1, 2, 3]
    probs = np.array([[0.2, 0.2, 0.05, 0.05, 0.5]])
    s = Scores(probs)
    assert s.pathogenic_given_variant[0] == pytest.approx(0.8)
    assert s.strict_pathogenic[0] == pytest.approx(0.8)
    assert bool(s.call_grouped()[0])  # argmax is 'normal', pathogenic mass > benign mass


def test_evaluate_strict_and_grouped():
    y = np.array([PATHOGENIC, BENIGN, LIKELY_BENIGN, NORMAL])
    probs = np.eye(5)[[0, 3, 2, 4]]
    out = evaluate(probs, y)
    assert out["accuracy_5class"] == 1.0
    assert out["strict_P_vs_B"]["accuracy"] == 1.0
    assert out["PLP_vs_BLB"]["n"] == 3
