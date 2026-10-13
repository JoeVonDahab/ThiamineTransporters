"""Integration tests against the published artifacts (skipped when not linked)."""

import numpy as np
import pytest

from conftest import LEGACY_ROOT, REFERENCE_DIR, requires_legacy, requires_reference

pytestmark = pytest.mark.integration


@requires_legacy
@requires_reference
def test_sequence_pipeline_matches_published_rows():
    from triplevar.legacy import LegacyArtifacts
    from triplevar.reference import load_reference
    from triplevar.sequences import Variant

    df = LegacyArtifacts(LEGACY_ROOT).combined()
    df = df[df["clinical_significance"] != "normal"].sample(500, random_state=1)
    refs = {g: load_reference(g, REFERENCE_DIR) for g in ("SLC19A2", "SLC19A3")}
    for rec in df.to_dict("records"):
        ref = refs["SLC19A2" if "SLC19A2" in str(rec["gene"]) else "SLC19A3"]
        dna, mrna, protein = ref.sequences(Variant(int(rec["pos"]), str(rec["original"]), str(rec["mutant"])))
        assert dna == rec["generated_dna"]
        assert mrna == rec["generated_mRNA"]
        assert protein == rec["protein_sequence"]


@requires_legacy
def test_abstract_numbers(tmp_path):
    from triplevar.reproduce import reproduce_abstract

    root = LEGACY_ROOT.parents[1]
    report = reproduce_abstract(LEGACY_ROOT, tmp_path, root / "validation/external/abstract_literature_labels.csv")
    strict = report["clinvar_test"]["strict_P_vs_B"]
    assert (strict["tp"], strict["fn"], strict["fp"], strict["tn"]) == (11, 2, 0, 17)
    assert report["clinvar_test"]["PLP_vs_BLB"]["auc_renormalised"] == pytest.approx(0.913, abs=5e-4)
    assert report["prioritised"]["pathogenic_high_confidence"] == 111
    assert report["prioritised"]["likely_pathogenic_lower_confidence"] == 303
    assert report["split"] == {"train": 604, "val": 130, "test": 130, "synthetic_normal": 962}
