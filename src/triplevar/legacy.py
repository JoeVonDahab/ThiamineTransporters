"""Load the artifacts behind the published model (Aug-Sep 2025 notebooks).

Reproduces, row for row, the frames the training notebook built from the two
``*_model_5.pkl`` datasets, so that the saved embedding matrices (which carry
no ids) line up with their variants:

* labeled set: every row of an rsID that has a clinical label (incl. its paired
  'normal' wild-type row), labels normalised, shuffled with seed 42 -> 2,333 rows,
* unlabeled pool: 10% of the remaining 'normal' rows (random.seed(42)) followed
  by all other unlabeled rows -> 31,199 rows,
* literature set: the 22 SLC19A3 variants of ``custom_mutation.csv``.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from .dataset import FeatureSet
from .features import FeatureLayout, tabular_matrix
from .labels import CODE_BY_NAME, VUS, normalise_label

DATASETS = {
    "SLC19A2": "final_dataset/final_dataset_SLC19A2_model_5.pkl",
    "SLC19A3": "final_dataset/final_dataset_SLC19A3_model_5.pkl",
}
EMBEDDINGS = {
    "labeled_dna": "embeddings/dnabert_embeddings_1754981297.npy",
    "labeled_protein": "embeddings/X_nn_esm2_20250811_235944.npy",
    "labeled_y": "embeddings/y_nn_esm2_20250811_235944.npy",
    "pool_dna": "embeddings/dnabert_embeddings_labeling_both1754989195.npy",
    "pool_protein": "embeddings/labeling_SLC19A2_X_nn_esm2_20250812_105918.npy",  # covers both genes
    "literature_dna": "embeddings/dnabert_embeddings_labeling_external1757407509.npy",
    "literature_protein": "embeddings/labeling_external_data_20250909_014714.npy",
}
LITERATURE_CSV = "custom_mutation.csv"

META_COLUMNS = [
    "snp_id", "gene", "#chr", "pos", "original", "mutant", "variation", "variant_type",
    "function_class", "clinical_significance", "protein_sequence",
]


class LegacyArtifacts:
    """Read-only access to the original 'Both Genes' working directory."""

    def __init__(self, root: Union[str, Path]):
        self.root = Path(root)
        if not (self.root / DATASETS["SLC19A2"]).exists():
            raise FileNotFoundError(f"{self.root} does not look like the legacy 'Both Genes' directory")
        self._combined: Optional[pd.DataFrame] = None

    def path(self, rel: str) -> Path:
        return self.root / rel

    def combined(self) -> pd.DataFrame:
        """Both genes, common columns, exact duplicates dropped (55,456 rows)."""
        if self._combined is None:
            a2 = pd.read_pickle(self.path(DATASETS["SLC19A2"]))
            a3 = pd.read_pickle(self.path(DATASETS["SLC19A3"]))
            common = a2.columns.intersection(a3.columns).tolist()
            self._combined = pd.concat([a2[common], a3[common]], ignore_index=True).drop_duplicates()
        return self._combined

    def _split_labeled(self):
        df = self.combined()
        sig = df["clinical_significance"]
        labeled_ids = df.loc[sig.notna() & (sig != "nan") & (sig != "normal"), "snp_id"].unique()
        mask = df["snp_id"].isin(labeled_ids)
        return df[mask].copy(), df[~mask].copy()

    def labeled(self) -> FeatureSet:
        """The 2,333-row labeled set (VUS rows included, code 0) in embedding order."""
        lab, _ = self._split_labeled()
        lab["label"] = lab["clinical_significance"].map(normalise_label)
        lab["y"] = lab["label"].map(CODE_BY_NAME)
        lab = lab.sample(frac=1, random_state=42).reset_index(drop=True)
        y = lab["y"].to_numpy(dtype=np.int64)
        saved_y = np.load(self.path(EMBEDDINGS["labeled_y"]))
        if not np.array_equal(y, saved_y):
            raise RuntimeError("Rebuilt labels differ from the saved label vector; row order would be wrong")
        return FeatureSet(
            meta=lab[[c for c in META_COLUMNS if c in lab.columns] + ["label", "y"]],
            dna=np.load(self.path(EMBEDDINGS["labeled_dna"])),
            protein=np.load(self.path(EMBEDDINGS["labeled_protein"])),
            tabular=tabular_matrix(lab),
            y=y,
        )

    def pool(self) -> FeatureSet:
        """Unlabeled pool (31,199 rows) in embedding order."""
        _, unl = self._split_labeled()
        normal_idx = unl.index[unl["clinical_significance"] == "normal"]
        other_idx = unl.index[unl["clinical_significance"] != "normal"]
        random.seed(42)
        keep = random.sample(list(normal_idx), int(len(normal_idx) * 0.1))
        pool = unl.loc[list(keep) + list(other_idx)].reset_index(drop=True)
        dna = np.load(self.path(EMBEDDINGS["pool_dna"]))
        protein = np.load(self.path(EMBEDDINGS["pool_protein"]))
        if not (len(pool) == len(dna) == len(protein)):
            raise RuntimeError(f"Pool has {len(pool)} rows but embeddings have {len(dna)}/{len(protein)}")
        return FeatureSet(
            meta=pool[[c for c in META_COLUMNS if c in pool.columns]],
            dna=dna, protein=protein, tabular=tabular_matrix(pool),
        )

    def literature(self) -> FeatureSet:
        """The 22 literature SLC19A3 variants (no allele-frequency data: tabular block is 0)."""
        lit = pd.read_csv(self.path(LITERATURE_CSV))
        meta = lit[[c for c in META_COLUMNS if c in lit.columns]].copy()
        meta = meta.rename(columns={"snp_id": "protein_change"})
        return FeatureSet(
            meta=meta,
            dna=np.load(self.path(EMBEDDINGS["literature_dna"])),
            protein=np.load(self.path(EMBEDDINGS["literature_protein"])),
            tabular=tabular_matrix(lit),
        )


@dataclass
class Split:
    train: np.ndarray
    val: np.ndarray
    test: np.ndarray
    scaler: StandardScaler


def published_split(labeled: FeatureSet, layout: FeatureLayout) -> Split:
    """The notebook split: drop VUS, stratified 70/15/15 by row with seed 42.

    Returns row indices into ``labeled`` and the StandardScaler fitted on the
    training rows (the published checkpoints were saved without it).
    """
    keep = np.flatnonzero(labeled.y != VUS)
    y = LabelEncoder().fit_transform(labeled.y[keep])
    trval, test = train_test_split(keep, test_size=0.15, random_state=42, stratify=y)
    y_trval = LabelEncoder().fit_transform(labeled.y[trval])
    train, val = train_test_split(trval, test_size=0.1765, random_state=42, stratify=y_trval)
    scaler = StandardScaler().fit(labeled.X(layout)[train])
    return Split(train=train, val=val, test=test, scaler=scaler)
