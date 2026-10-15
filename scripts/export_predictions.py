"""Per-variant predictions of a model for the external sets and the prioritisation pool.

    python scripts/export_predictions.py --model models/triplevar_v1.pt --out reports/predictions
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from triplevar.dataset import FeatureSet
from triplevar.ensemble import load_model
from triplevar.labels import VUS
from triplevar.predict import prediction_table

DROP = ["protein_sequence", "generated_dna", "generated_mRNA", "transcript_dna", "frequency", "normal_seq"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--processed", default="data/processed")
    ap.add_argument("--out", default="reports/predictions")
    args = ap.parse_args()
    model = load_model(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    tag = Path(args.model).stem

    for name in ("clinvar_timesplit", "literature_abstract", "lit_transport", "lit_binding"):
        fs = FeatureSet.load(f"{args.processed}/{name}")
        t = prediction_table(model, fs.X(model.layout), fs.meta)
        t.drop(columns=[c for c in DROP if c in t.columns]).to_csv(out / f"{name}.{tag}.csv", index=False)

    labeled = FeatureSet.load(f"{args.processed}/labeled")
    vus = labeled.subset(labeled.y == VUS)
    pool = FeatureSet.load(f"{args.processed}/pool")
    tables = []
    for source, fs in (("ClinVar VUS", vus), ("dbSNP unlabeled", pool)):
        t = prediction_table(model, fs.X(model.layout), fs.meta).assign(source=source)
        tables.append(t[t["clinical_significance"].astype(str) != "normal"])
    allp = pd.concat(tables, ignore_index=True)
    ranked = allp.sort_values("p_pathogenic", ascending=False, kind="mergesort")
    ranked.drop(columns=[c for c in DROP if c in ranked.columns]).to_csv(out / f"prioritisation_all.{tag}.csv", index=False)
    top = allp[allp["pathogenic_call"].astype(bool)].sort_values("p_pathogenic", ascending=False)
    top.drop(columns=[c for c in DROP if c in top.columns]).to_csv(out / f"prioritised_pathogenic.{tag}.csv", index=False)
    counts = allp.groupby(["source", "predicted_class"]).size().unstack(fill_value=0)
    counts.to_csv(out / f"prioritisation_counts.{tag}.csv")
    print(counts)
    print(f"P/LP candidates: {len(top)} of {len(allp)} variants")


if __name__ == "__main__":
    main()
