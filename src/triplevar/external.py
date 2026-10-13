"""Featurise an external variant list (literature, ClinVar time-split, ...).

Input CSV columns:
    variant_id, gene, label_binary (1 pathogenic / 0 benign / blank unknown) and either
    * pos, ref, alt   -- GRCh38 + strand; VCF-style anchored indels are accepted, or
    * protein_change  -- one-letter missense change such as E320Q.
Optional allele-frequency columns (see features.TABULAR_COLUMNS) are used when
present; otherwise the tabular block is all zeros ("no population data").
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional, Tuple, Union

import pandas as pd

from .dataset import FeatureSet
from .embeddings import embed_sequences
from .features import tabular_matrix
from .reference import load_reference
from .sequences import Variant, protein_change_to_variant

log = logging.getLogger(__name__)


def normalise_alleles(pos: int, ref: str, alt: str) -> Tuple[int, str, str]:
    """VCF-style -> dbSNP-style: drop the shared leading base(s), '-' for an empty side."""
    ref, alt = str(ref).upper(), str(alt).upper()
    if ref in ("-", "") or alt in ("-", ""):
        return int(pos), ref or "-", alt or "-"
    k = 0
    while k < min(len(ref), len(alt)) and ref[k] == alt[k]:
        k += 1
    if k == 0:
        return int(pos), ref, alt
    return int(pos) + k, ref[k:] or "-", alt[k:] or "-"


def variant_sequences(table: pd.DataFrame, reference_dir: Union[str, Path]) -> pd.DataFrame:
    refs = {}
    rows = []
    for rec in table.to_dict("records"):
        gene = rec["gene"]
        if gene not in refs:
            refs[gene] = load_reference(gene, reference_dir)
        ref = refs[gene]
        if isinstance(rec.get("protein_change"), str) and rec["protein_change"]:
            variant = protein_change_to_variant(ref, rec["protein_change"])
        else:
            variant = Variant(*normalise_alleles(rec["pos"], rec["ref"], rec["alt"]))
        dna, mrna, protein = ref.sequences(variant)
        rows.append({
            **rec, "pos_dbsnp": variant.pos, "ref_dbsnp": variant.ref, "alt_dbsnp": variant.alt,
            "transcript_dna": dna, "protein_sequence": protein,
            "protein_changed": protein != ref.wt_protein, "status": "ok" if dna and protein else "reference_mismatch",
        })
    return pd.DataFrame(rows)


def build_external_set(
    input_csv: Union[str, Path], out_prefix: Union[str, Path], reference_dir: Union[str, Path] = "data/reference",
    device: Optional[str] = None,
) -> FeatureSet:
    table = variant_sequences(pd.read_csv(input_csv), reference_dir)
    bad = table["status"] != "ok"
    if bad.any():
        log.warning("%d variants could not be applied to the reference and are skipped", int(bad.sum()))
    table = table[~bad].reset_index(drop=True)
    X_dna, X_prot = embed_sequences(table["transcript_dna"], table["protein_sequence"], device)
    fs = FeatureSet(meta=table.drop(columns=["transcript_dna"]), dna=X_dna, protein=X_prot, tabular=tabular_matrix(table))
    fs.save(out_prefix)
    log.info("external set: %d variants -> %s", len(fs), out_prefix)
    return fs
