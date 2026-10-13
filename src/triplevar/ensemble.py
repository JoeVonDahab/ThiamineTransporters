"""An average-of-probabilities ensemble of bundles (e.g. the cross-validation models)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import torch

from .model import BUNDLE_FORMAT, Bundle

ENSEMBLE_FORMAT = "triplevar-ensemble/1"


class Ensemble:
    def __init__(self, members: List[Bundle], metadata: Optional[Dict[str, Any]] = None):
        if not members:
            raise ValueError("empty ensemble")
        self.members = members
        self.metadata = dict(metadata or {})

    @property
    def layout(self):
        return self.members[0].layout

    @property
    def classes(self) -> List[int]:
        return self.members[0].classes

    def predict_proba(self, X: np.ndarray, device: str = "cpu") -> np.ndarray:
        return np.mean([m.predict_proba(X, device=device) for m in self.members], axis=0)

    def save(self, path: Union[str, Path], half: bool = False) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {"format": ENSEMBLE_FORMAT, "members": [m.to_dict(half=half) for m in self.members], "metadata": self.metadata},
            path,
        )


def load_model(path: Union[str, Path]):
    """Load either a single Bundle or an Ensemble."""
    raw = torch.load(str(path), map_location="cpu")
    fmt = raw.get("format") if isinstance(raw, dict) else None
    if fmt == ENSEMBLE_FORMAT:
        return Ensemble([Bundle.from_dict(m) for m in raw["members"]], raw.get("metadata"))
    if fmt == BUNDLE_FORMAT:
        return Bundle.from_dict(raw)
    raise ValueError(f"{path} is neither a TripleVar bundle nor an ensemble")
