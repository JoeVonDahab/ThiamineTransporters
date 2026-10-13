import pandas as pd
import pytest

from triplevar.prepare import expand_alleles, frequency_features, validation_flags


def test_frequency_features_only_count_the_alt_allele():
    f = frequency_features("A:0.25:10:TOPMED|G:0.75:30:TOPMED|A:0.1:2:GnomAD_genomes|A:0.3:1:NotAStudy", "A")
    assert f["TOPMED_freq"] == 0.25 and f["TOPMED_is_mentioned"] == 1
    assert f["GnomAD_genomes_freq"] == 0.1
    assert f["Total_Studies"] == 2
    assert f["Max_Frequency"] == 0.25 and f["Min_Frequency"] == 0.1
    assert f["Avg_Frequency"] == pytest.approx(0.175)


def test_frequency_features_empty():
    f = frequency_features(None, "A")
    assert f["Total_Studies"] == 0 and f["Avg_Frequency"] == 0.0


def test_validation_flags():
    assert validation_flags("by-frequency;by-alfa") == {"by-frequency": 1, "by-alfa": 1, "by-cluster": 0}
    assert validation_flags(None) == {"by-frequency": 0, "by-alfa": 0, "by-cluster": 0}


def test_expand_alleles():
    df = pd.DataFrame({"variation": ["G>A,C", "T>-"], "snp_id": ["1", "2"]})
    out = expand_alleles(df)
    assert out[["original", "mutant"]].values.tolist() == [["G", "A"], ["G", "C"], ["T", "-"]]
    assert (out["mutant_dbsnp"] == out["mutant"]).all()
