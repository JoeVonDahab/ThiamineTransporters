"""Build the per-gene variant table from a raw dbSNP tabular export.

Port of the preparation notebooks. Steps, per gene:
 1. read the export (the web export repeats its header once per page),
 2. split multi-allelic rows into one row per alternative allele,
 3. check every reference allele against the genome; mismatches are resolved
    from dbSNP's SPDI notation when an ``spdi`` resolver is supplied,
 4. build transcript DNA, mRNA and protein for every allele,
 5. population-frequency features (fixed study order) and validation flags,
 6. one synthetic 'normal' (wild-type) row per rsID whose frequency features
    are drawn from the benign rows (seed 42), as in the published dataset.

Known limitations inherited from the published data are flagged, not silently
changed: dbSNP reports clinical significance per rsID, so every allele of a
multi-allelic rsID receives the same label (see ``triplevar.audit``).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .features import FREQUENCY_STUDIES, VALIDATION_COLUMNS
from .labels import normalise_label
from .sequences import GeneReference, Variant

log = logging.getLogger(__name__)

EXPORT_COLUMNS = [
    "#chr", "pos", "variation", "variant_type", "snp_id", "clinical_significance",
    "validation_status", "function_class", "gene", "frequency",
]

# (snp_id, pos) -> (ref, alt) or None
SpdiResolver = Callable[[str, int], Optional[Tuple[str, str]]]


def ncbi_spdi_resolver(chromosome_accession: str, delay: float = 0.34) -> SpdiResolver:
    """Resolve alleles from dbSNP's SPDI notation via NCBI E-utilities.

    Set NCBI_API_KEY / NCBI_EMAIL in the environment to raise the rate limit.
    Like the published pipeline it takes the first SPDI on the record.
    """
    import os
    import re
    import time
    import urllib.parse
    import urllib.request

    pattern = re.compile(rf"{re.escape(chromosome_accession)}:(\d+):([ACGT]*):([ACGT]*)")

    def resolve(snp_id: str, pos: int) -> Optional[Tuple[str, str]]:
        params = {"db": "snp", "id": str(snp_id).replace("rs", ""), "rettype": "xml", "retmode": "xml", "tool": "triplevar"}
        for env, key in (("NCBI_API_KEY", "api_key"), ("NCBI_EMAIL", "email")):
            if os.environ.get(env):
                params[key] = os.environ[env]
        time.sleep(delay)
        url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi?" + urllib.parse.urlencode(params)
        with urllib.request.urlopen(url, timeout=60) as r:
            m = pattern.search(r.read().decode())
        if not m or not (m.group(2) or m.group(3)):
            return None
        return m.group(2), m.group(3)

    return resolve


def read_dbsnp_export(path: Union[str, Path], drop_duplicates: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path, sep="\t", dtype=str)
    df = df[df["#chr"] != "#chr"]  # repeated page headers
    n = len(df)
    if drop_duplicates:
        df = df.drop_duplicates()
        if len(df) < n:
            log.info("%s: dropped %d exact duplicate lines", Path(path).name, n - len(df))
    df["pos"] = pd.to_numeric(df["pos"], errors="coerce")
    return df.dropna(subset=["pos"]).astype({"pos": int}).reset_index(drop=True)


def expand_alleles(df: pd.DataFrame) -> pd.DataFrame:
    """'G>A,C' -> two rows with original='G' and mutant 'A' / 'C'."""
    parts = df["variation"].str.split(">", n=1, expand=True)
    df = df.assign(original=parts[0], mutant=parts[1].str.split(","))
    df = df.explode("mutant").reset_index(drop=True)
    # the allele as spelled in dbSNP's frequency strings (kept if SPDI later rewrites it)
    return df.assign(mutant_dbsnp=df["mutant"])


def check_alleles(df: pd.DataFrame, ref: GeneReference, spdi: Optional[SpdiResolver] = None) -> pd.DataFrame:
    """Flag (and optionally fix via SPDI) reference alleles that do not match the genome."""
    start = ref.gene.window_start
    status = []
    for i, rec in df.iterrows():
        if rec["original"] == "-":
            status.append("insertion_unchecked")
            continue
        k = rec["pos"] - start
        if ref.window[k : k + len(rec["original"])] == rec["original"]:
            status.append("ok")
            continue
        fixed = spdi(rec["snp_id"], rec["pos"]) if spdi else None
        if fixed and ref.window[k : k + len(fixed[0])] == fixed[0]:
            df.at[i, "original"], df.at[i, "mutant"] = fixed
            status.append("spdi_corrected")
        else:
            status.append("reference_mismatch")
    return df.assign(allele_status=status)


def add_sequences(df: pd.DataFrame, ref: GeneReference) -> pd.DataFrame:
    dna, mrna, protein = [], [], []
    for rec in df.itertuples():
        variant = None if rec.clinical_significance == "normal" else Variant(rec.pos, rec.original, rec.mutant)
        d, m, p = ref.sequences(variant)
        dna.append(d), mrna.append(m), protein.append(p)
    return df.assign(generated_dna=dna, generated_mRNA=mrna, protein_sequence=protein)


def frequency_features(frequency: Optional[str], mutant: str) -> Dict[str, float]:
    """dbSNP 'allele:freq:count:STUDY|...' -> per-study frequency/mention + summaries.

    Only entries reported for the variant's alternative allele are used.
    """
    out: Dict[str, float] = {}
    for s in FREQUENCY_STUDIES:
        out[f"{s}_freq"] = 0.0
        out[f"{s}_is_mentioned"] = 0
    found: List[float] = []
    if isinstance(frequency, str) and frequency.strip():
        for entry in frequency.split("|"):
            parts = entry.split(":")
            if len(parts) == 4 and parts[0] == mutant and parts[3] in FREQUENCY_STUDIES:
                value = float(parts[1])
                out[f"{parts[3]}_freq"] = value
                out[f"{parts[3]}_is_mentioned"] = 1
                found.append(value)
    out["Avg_Frequency"] = float(np.mean(found)) if found else 0.0
    out["Max_Frequency"] = float(np.max(found)) if found else 0.0
    out["Min_Frequency"] = float(np.min(found)) if found else 0.0
    out["Total_Studies"] = len(found)
    return out


def validation_flags(status: Optional[str]) -> Dict[str, int]:
    tokens = set(status.split(";")) if isinstance(status, str) else set()
    return {flag: int(flag in tokens) for flag in VALIDATION_COLUMNS}


def add_normal_rows(df: pd.DataFrame) -> pd.DataFrame:
    """One wild-type ('normal') row per rsID, copied from the rsID's first row."""
    normals = df.drop_duplicates("snp_id").copy()
    normals["clinical_significance"] = "normal"
    normals["mutant"] = normals["original"]
    return pd.concat([df, normals], ignore_index=True).sort_values("snp_id", kind="stable").reset_index(drop=True)


def impute_normal_frequencies(df: pd.DataFrame, seed: int = 42, p_zero: float = 0.5) -> pd.DataFrame:
    """Draw 'normal' rows' frequencies from N(mean, sd) of the benign rows, zero half of them."""
    rng = np.random.default_rng(seed)
    freq_cols = [f"{s}_freq" for s in FREQUENCY_STUDIES]
    normal = df["clinical_significance"].eq("normal").to_numpy()
    benign = df.loc[df["clinical_significance"].eq("benign"), freq_cols]
    mean, sd = benign.mean().to_numpy(), benign.std().fillna(0).to_numpy()
    values = mean + rng.standard_normal((normal.sum(), len(freq_cols))) * sd
    values[:, sd == 0] = np.broadcast_to(mean, values.shape)[:, sd == 0]
    values = np.clip(np.nan_to_num(values), 0.0, 1.0)
    values[rng.random(values.shape) < p_zero] = 0.0
    df.loc[normal, freq_cols] = values
    freq = df[freq_cols]
    for s in FREQUENCY_STUDIES:
        df[f"{s}_is_mentioned"] = (df[f"{s}_freq"] > 0).astype(int)
    present = freq.where(freq > 0)
    df["Total_Studies"] = (freq > 0).sum(axis=1)
    df["Avg_Frequency"] = present.mean(axis=1).fillna(0.0)
    df["Max_Frequency"] = present.max(axis=1).fillna(0.0)
    df["Min_Frequency"] = present.min(axis=1).fillna(0.0)
    df.loc[normal, VALIDATION_COLUMNS] = 1
    return df


def build_gene_table(
    export_path: Union[str, Path], ref: GeneReference, spdi: Optional[SpdiResolver] = None
) -> pd.DataFrame:
    df = expand_alleles(read_dbsnp_export(export_path))
    df = check_alleles(df, ref, spdi)
    df = add_normal_rows(df)
    # Frequencies are matched on dbSNP's own allele spelling. (The published
    # notebooks matched the SPDI-rewritten allele, which zeroed the frequency
    # features of 317 SLC19A2 / 502 SLC19A3 corrected rows.)
    tab = pd.DataFrame(
        [{**frequency_features(r.frequency, r.mutant_dbsnp), **validation_flags(r.validation_status)}
         for r in df.itertuples()] if len(df) else []
    )
    df = pd.concat([df, tab], axis=1)
    normal = df["clinical_significance"].eq("normal")
    df = impute_normal_frequencies(df)
    df = add_sequences(df, ref)
    df["label"] = df["clinical_significance"].map(normalise_label)
    log.info("%s: %d rows (%d alleles, %d rsIDs)", ref.gene.name, len(df), int((~normal).sum()), df["snp_id"].nunique())
    return df
