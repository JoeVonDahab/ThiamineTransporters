"""YAML run configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Union

import yaml

from .features import FeatureLayout
from .training import TrainConfig


@dataclass
class RunConfig:
    name: str
    layout: FeatureLayout = field(default_factory=FeatureLayout)
    train: TrainConfig = field(default_factory=TrainConfig)
    split: str = "cv"  # "cv" (grouped K-fold by rsID) or "published" (row-level, seed 42)
    n_folds: int = 10
    seeds: List[int] = field(default_factory=lambda: [42])
    # fraction of synthetic 'normal' rows removed from the training fold only
    drop_normal_fraction: float = 0.0

    @classmethod
    def from_yaml(cls, path: Union[str, Path]) -> "RunConfig":
        raw: Dict[str, Any] = yaml.safe_load(Path(path).read_text())
        return cls(
            name=raw.get("name", Path(path).stem),
            layout=FeatureLayout(**raw.get("layout", {})),
            train=TrainConfig(**raw.get("train", {})),
            split=raw.get("split", "cv"),
            n_folds=int(raw.get("n_folds", 10)),
            seeds=list(raw.get("seeds", [42])),
            drop_normal_fraction=float(raw.get("drop_normal_fraction", 0.0)),
        )
