"""Assemble the ClinVar time-split external set as a FeatureSet.

Variants classified P/LP/B/LB in ClinVar (Oct 2026) that were unlabeled, VUS or
absent in the training data (see validation/external/CLINVAR_TIMESPLIT.md):

* TRAIN_UNLABELED / TRAIN_VUS: reuse the published features of the matching
  training-data row (real allele-frequency features),
* ABSENT small variants + the UniProt-only SLC19A2 p.Asp93His: sequences are
  built from GRCh38 coordinates and embedded; no population data (tabular = 0).
Exon-level deletions/CNVs (17) are skipped (no sequence-level representation).

    python scripts/build_clinvar_timesplit.py --out data/processed/clinvar_timesplit
"""

import argparse
import logging

import numpy as np
import pandas as pd

from triplevar.dataset import FeatureSet
from triplevar.embeddings import embed_sequences
from triplevar.external import variant_sequences
from triplevar.features import tabular_matrix

log = logging.getLogger("build_clinvar_timesplit")
KEY = ["snp_id", "pos", "original", "mutant"]


def _key(df, cols):
    return list(zip(*(df[c].astype(str).str.replace(r"\.0$", "", regex=True) for c in cols)))


def from_existing(rows: pd.DataFrame, source: FeatureSet, mask=None) -> FeatureSet:
    src_meta = source.meta if mask is None else source.meta[mask]
    index = {}
    for i, k in zip(src_meta.index, _key(src_meta, KEY)):
        index.setdefault(k, i)
    hits = [index.get(k) for k in _key(rows, ["train_snp_id", "train_pos", "train_original", "train_mutant"])]
    missing = sum(h is None for h in hits)
    if missing:
        log.warning("%d rows not found in the source feature set; dropped", missing)
    ok = [h is not None for h in hits]
    idx = np.array([h for h in hits if h is not None], dtype=int)
    sub = source.subset(idx)
    sub.meta = rows[ok].reset_index(drop=True).assign(feature_source="published_features")
    return sub


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--timesplit", default="validation/external/clinvar_timesplit.csv")
    ap.add_argument("--uniprot", default="validation/external/uniprot_variants.csv")
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--reference-dir", default="data/reference")
    ap.add_argument("--out", default="data/processed/clinvar_timesplit")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    ts = pd.read_csv(args.timesplit)
    ts = ts[ts["external_eligible"] == True].copy()  # noqa: E712
    ts["variant_id"] = "ClinVar:" + ts["variation_id"].astype(str)

    pool = FeatureSet.load(f"{args.processed}/pool")
    labeled = FeatureSet.load(f"{args.processed}/labeled")
    parts = [
        from_existing(ts[ts["training_category"] == "TRAIN_UNLABELED"], pool),
        from_existing(ts[ts["training_category"] == "TRAIN_VUS"], labeled, mask=labeled.y == 0),
    ]

    new = ts[(ts["training_category"] == "ABSENT") & (ts["small_variant"] == True)].copy()  # noqa: E712
    uni = pd.read_csv(args.uniprot)
    uni = uni[uni["external_eligible"] == True].copy()  # noqa: E712
    if len(uni):
        uni = uni.assign(
            variant_id="UniProt:" + uni["feature_id"], training_category="ABSENT", consequence="missense variant",
            is_missense=True, clinvar_classification="UniProt " + uni["humsavar_category"].astype(str),
        )
        new = pd.concat([new, uni[[c for c in uni.columns if c in new.columns or c == "variant_id"]]], ignore_index=True)
    new = new[["variant_id", "gene", "pos", "ref", "alt"] + [c for c in new.columns if c not in ("variant_id", "gene", "pos", "ref", "alt")]]
    seqs = variant_sequences(new, args.reference_dir)
    bad = seqs["status"] != "ok"
    if bad.any():
        log.warning("reference mismatch for %s", seqs.loc[bad, "variant_id"].tolist())
    seqs = seqs[~bad].reset_index(drop=True)
    X_dna, X_prot = embed_sequences(seqs["transcript_dna"], seqs["protein_sequence"], args.device)
    parts.append(
        FeatureSet(
            meta=seqs.drop(columns=["transcript_dna"]).assign(feature_source="new_embedding_no_frequency"),
            dna=X_dna, protein=X_prot, tabular=np.zeros((len(seqs), tabular_matrix(seqs).shape[1]), np.float32),
        )
    )

    meta = pd.concat([p.meta for p in parts], ignore_index=True)
    fs = FeatureSet(
        meta=meta,
        dna=np.concatenate([p.dna for p in parts]),
        protein=np.concatenate([p.protein for p in parts]),
        tabular=np.concatenate([p.tabular for p in parts]),
    )
    fs.save(args.out)
    log.info("time-split: %d variants (%d P/LP, %d B/LB) -> %s", len(fs), int((meta.label_binary == 1).sum()),
             int((meta.label_binary == 0).sum()), args.out)
    print(meta.groupby(["training_category", "feature_source"])["label_binary"].value_counts().unstack(fill_value=0))


if __name__ == "__main__":
    main()
