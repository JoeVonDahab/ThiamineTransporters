"""Build rule-labelled literature validation sets (fixed rules, set before scoring).

transport  (Li et al. 2024, [3H]-thiamine uptake in HEK293T, SLC19A2 + SLC19A3):
           loss of function if < 50 % of WT, tolerated if >= 75 %, else excluded.
binding    (Gabriel et al. 2024, nanoDSF thiamine Kd,app, SLC19A3):
           loss of function if Kd >= 5x WT or no binding, tolerated if <= 2x, else excluded.
           K44E is excluded: its defect (ER retention) is invisible to a binding assay.
Only single amino-acid substitutions are used; any variant whose protein also
occurs among the labeled training rows is dropped (flagged in_training).

    python scripts/build_literature_sets.py
"""

import argparse
import logging
import re

import numpy as np
import pandas as pd

from triplevar.dataset import FeatureSet
from triplevar.embeddings import embed_sequences
from triplevar.external import variant_sequences
from triplevar.features import TABULAR_DIM
from triplevar.labels import NORMAL, VUS

log = logging.getLogger("build_literature_sets")
SINGLE = re.compile(r"^[A-Z]\d+[A-Z]$")


def transport_table(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["assay"].astype(str).str.contains("uptake", case=False) & d["activity_pct_wt"].notna()]
    d = d[d["protein_change"].astype(str).str.match(SINGLE)].copy()
    pct = d["activity_pct_wt"].astype(float)
    d["label_binary"] = np.where(pct < 50, 1, np.where(pct >= 75, 0, np.nan))
    d["evidence"] = "uptake " + pct.round(1).astype(str) + "% of WT"
    return d[["gene", "protein_change", "label_binary", "evidence", "source"]]


def binding_table(path: str) -> pd.DataFrame:
    d = pd.read_csv(path)
    d = d[d["protein_change"].str.match(SINGLE) & (d["protein_change"] != "K44E")].copy()
    fold = d["fold_vs_wt"].astype(float)
    lof = d["no_binding"].astype(str).str.lower().eq("true") | (fold >= 5)
    d["label_binary"] = np.where(lof, 1, np.where(fold <= 2, 0, np.nan))
    d["evidence"] = np.where(d["no_binding"].astype(str).str.lower().eq("true"), "no binding",
                             "Kd " + fold.round(2).astype(str) + "x WT")
    return d[["gene", "protein_change", "label_binary", "evidence", "source"]]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--literature", default="validation/external/literature_variants.csv")
    ap.add_argument("--binding", default="validation/external/literature_binding_gabriel2024.csv")
    ap.add_argument("--labeled", default="data/processed/labeled")
    ap.add_argument("--reference-dir", default="data/reference")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--table-out", default="validation/external/literature_rule_labels.csv")
    ap.add_argument("--device")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    sets = {"lit_transport": transport_table(args.literature), "lit_binding": binding_table(args.binding)}
    labeled = FeatureSet.load(args.labeled)
    trained = labeled.meta.loc[~np.isin(labeled.y, [NORMAL, VUS]), "protein_sequence"]
    training_proteins = set(trained.dropna())

    rows = []
    for name, table in sets.items():
        table = table.assign(set=name, variant_id=name + ":" + table["gene"] + ":" + table["protein_change"])
        rows.append(table)
    table = pd.concat(rows, ignore_index=True)
    seqs = variant_sequences(table, args.reference_dir)
    seqs["in_training"] = seqs["protein_sequence"].isin(training_proteins)
    seqs.drop(columns=["transcript_dna"]).to_csv(args.table_out, index=False)
    log.info("rule labels written to %s", args.table_out)

    keep = seqs[(seqs["status"] == "ok") & seqs["label_binary"].notna() & ~seqs["in_training"]].reset_index(drop=True)
    X_dna, X_prot = embed_sequences(keep["transcript_dna"], keep["protein_sequence"], args.device)
    for name in sets:
        m = (keep["set"] == name).to_numpy()
        fs = FeatureSet(meta=keep[m].drop(columns=["transcript_dna"]).reset_index(drop=True),
                        dna=X_dna[m], protein=X_prot[m], tabular=np.zeros((int(m.sum()), TABULAR_DIM), np.float32))
        fs.save(f"{args.out_dir}/{name}")
        log.info("%s: %d variants (%d LOF, %d tolerated)", name, len(fs),
                 int((fs.meta.label_binary == 1).sum()), int((fs.meta.label_binary == 0).sum()))
    print(seqs[["set", "gene", "protein_change", "evidence", "label_binary", "in_training", "status"]].to_string())


if __name__ == "__main__":
    main()
