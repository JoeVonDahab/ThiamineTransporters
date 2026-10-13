"""Feature layout of the fused TripleVar input vector.

    [ DNABERT-2 (5 x 768 = 3840) | ESM-2 (2560) | tabular (71) ]  -> 6471

The tabular block order is the column order the published model was trained
with (it came from the DataFrame, not from a list). It is fixed here and every
caller selects columns by name, so new data can never be silently permuted.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np
import pandas as pd

DNA_DIM = 3840
PROTEIN_DIM = 2560

# Population studies in dbSNP frequency strings, in published-model order.
FREQUENCY_STUDIES: List[str] = [
    "MGP", "HGDP_Stanford", "FINRISK", "Chileans", "GENOME_DK", "GnomAD_exomes", "SGDP_PRJ",
    "ExAC", "PAGE_STUDY", "Korea1K", "Estonian", "ALSPAC", "Korea4K", "KOREAN", "GENOGRAPHIC",
    "Daghestan", "GoNL", "TOMMO", "1000Genomes_30X", "HapMap", "1000Genomes", "TWINSUK",
    "NorthernSweden", "PRJEB36033", "GoESP", "Vietnamese", "TOPMED", "PRJEB37584", "ALFA",
    "GnomAD_genomes", "Siberian", "Qatari",
]
SUMMARY_COLUMNS: List[str] = ["Avg_Frequency", "Max_Frequency", "Min_Frequency", "Total_Studies"]
VALIDATION_COLUMNS: List[str] = ["by-frequency", "by-alfa", "by-cluster"]

TABULAR_COLUMNS: List[str] = (
    [c for s in FREQUENCY_STUDIES for c in (f"{s}_freq", f"{s}_is_mentioned")]
    + SUMMARY_COLUMNS
    + VALIDATION_COLUMNS
)
TABULAR_DIM = len(TABULAR_COLUMNS)  # 71


@dataclass(frozen=True)
class FeatureLayout:
    """Which modalities are present and where they sit in the fused vector."""

    use_dna: bool = True
    use_protein: bool = True
    use_tabular: bool = True

    @property
    def dim(self) -> int:
        return DNA_DIM * self.use_dna + PROTEIN_DIM * self.use_protein + TABULAR_DIM * self.use_tabular

    def slices(self) -> dict:
        out, start = {}, 0
        for name, used, width in (
            ("dna", self.use_dna, DNA_DIM),
            ("protein", self.use_protein, PROTEIN_DIM),
            ("tabular", self.use_tabular, TABULAR_DIM),
        ):
            if used:
                out[name] = slice(start, start + width)
                start += width
        return out

    def to_dict(self) -> dict:
        return {"use_dna": self.use_dna, "use_protein": self.use_protein, "use_tabular": self.use_tabular}


def tabular_matrix(df: pd.DataFrame, columns: Sequence[str] = TABULAR_COLUMNS) -> np.ndarray:
    """Select the tabular block by name (missing columns are treated as 0)."""
    block = df.reindex(columns=list(columns)).apply(pd.to_numeric, errors="coerce")
    return block.fillna(0.0).to_numpy(dtype=np.float32)


def fuse(
    dna: Optional[np.ndarray],
    protein: Optional[np.ndarray],
    tabular: Optional[np.ndarray],
    layout: FeatureLayout,
) -> np.ndarray:
    blocks = []
    for name, used, block, width in (
        ("dna", layout.use_dna, dna, DNA_DIM),
        ("protein", layout.use_protein, protein, PROTEIN_DIM),
        ("tabular", layout.use_tabular, tabular, TABULAR_DIM),
    ):
        if not used:
            continue
        if block is None or block.ndim != 2 or block.shape[1] != width:
            raise ValueError(f"{name} block must have shape (n, {width}); got {None if block is None else block.shape}")
        blocks.append(block.astype(np.float32, copy=False))
    n = {b.shape[0] for b in blocks}
    if len(n) != 1:
        raise ValueError(f"Feature blocks have different row counts: {sorted(n)}")
    return np.concatenate(blocks, axis=1)
