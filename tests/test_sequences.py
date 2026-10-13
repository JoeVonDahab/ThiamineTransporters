import pytest

from triplevar.external import normalise_alleles
from triplevar.genes import SLC19A2, SLC19A3
from triplevar.sequences import Variant, parse_protein_change, protein_change_to_variant, reverse_complement

from conftest import requires_reference


def test_reverse_complement():
    assert reverse_complement("AACGTN") == "NACGTT"


def test_gene_windows():
    assert SLC19A2.window_length == 26062
    assert SLC19A3.window_length == 37266
    assert SLC19A3.window_start == 227682763


def test_parse_protein_change():
    assert parse_protein_change("E320Q") == ("E", 320, "Q")
    assert parse_protein_change("p.W94*") == ("W", 94, "*")
    with pytest.raises(ValueError):
        parse_protein_change("Glu320Gln")


def test_normalise_vcf_alleles():
    assert normalise_alleles(100, "A", "G") == (100, "A", "G")
    assert normalise_alleles(100, "AT", "A") == (101, "T", "-")  # deletion
    assert normalise_alleles(100, "A", "AT") == (101, "-", "T")  # insertion


@requires_reference
def test_reference_proteins(slc19a2_reference, slc19a3_reference):
    assert len(slc19a2_reference.wt_protein) == 497
    assert len(slc19a3_reference.wt_protein) == 496
    assert slc19a3_reference.wt_protein.startswith("MDCYRTSLSSSWIYPTVILCLFGFFSMMRPSEPF")
    assert slc19a2_reference.wt_mrna[SLC19A2.cds_start : SLC19A2.cds_start + 3] == "AUG"


@requires_reference
def test_missense_roundtrip(slc19a3_reference):
    ref = slc19a3_reference
    for change in ("E320Q", "W94R", "N297A", "T422A"):
        variant = protein_change_to_variant(ref, change)
        _, _, protein = ref.sequences(variant)
        wt, pos, alt = parse_protein_change(change)
        assert protein[pos - 1] == alt
        assert protein[: pos - 1] == ref.wt_protein[: pos - 1]
        assert protein[pos:] == ref.wt_protein[pos:]


@requires_reference
def test_reference_mismatch_returns_none(slc19a3_reference):
    ref = slc19a3_reference
    i = 5000
    wrong = "A" if ref.window[i] != "A" else "C"
    assert ref.mutate_window(Variant(ref.gene.window_start + i, wrong, "G")) is None


@requires_reference
def test_intronic_variant_keeps_wt_mrna(slc19a3_reference):
    ref = slc19a3_reference
    pos = ref.gene.exons[0][1] + 50  # inside intron 1 (+ strand numbering)
    base = ref.window[pos - ref.gene.window_start]
    alt = "A" if base != "A" else "C"
    dna, mrna, protein = ref.sequences(Variant(pos, base, alt))
    assert dna is not None and mrna == ref.wt_mrna and protein == ref.wt_protein
