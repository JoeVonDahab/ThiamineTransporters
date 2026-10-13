"""In-memory and on-disk representation of featurised variants."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Union

import numpy as np
import pandas as pd

from .features import FeatureLayout, fuse


@dataclass
class FeatureSet:
    """Variant metadata plus the three per-modality feature blocks (row-aligned)."""

    meta: pd.DataFrame
    dna: np.ndarray
    protein: np.ndarray
    tabular: np.ndarray
    y: Optional[np.ndarray] = None

    def __post_init__(self) -> None:
        n = len(self.meta)
        for name in ("dna", "protein", "tabular"):
            if len(getattr(self, name)) != n:
                raise ValueError(f"{name} has {len(getattr(self, name))} rows, metadata has {n}")

    def __len__(self) -> int:
        return len(self.meta)

    def X(self, layout: FeatureLayout) -> np.ndarray:
        return fuse(self.dna, self.protein, self.tabular, layout)

    def subset(self, rows) -> "FeatureSet":
        """Rows by integer index or boolean mask."""
        rows = np.asarray(rows)
        if rows.dtype == bool:
            rows = np.flatnonzero(rows)
        return FeatureSet(
            meta=self.meta.iloc[rows].reset_index(drop=True),
            dna=self.dna[rows],
            protein=self.protein[rows],
            tabular=self.tabular[rows],
            y=None if self.y is None else self.y[rows],
        )

    def save(self, path: Union[str, Path]) -> None:
        """Write ``<path>.npz`` (features) and ``<path>.csv`` (metadata)."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {"dna": self.dna, "protein": self.protein, "tabular": self.tabular}
        if self.y is not None:
            arrays["y"] = self.y
        np.savez_compressed(path.with_suffix(".npz"), **arrays)
        self.meta.to_csv(path.with_suffix(".csv"), index=False)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "FeatureSet":
        path = Path(path)
        arrays = np.load(path.with_suffix(".npz"))
        return cls(
            meta=pd.read_csv(path.with_suffix(".csv")),
            dna=arrays["dna"],
            protein=arrays["protein"],
            tabular=arrays["tabular"],
            y=arrays["y"] if "y" in arrays else None,
        )
