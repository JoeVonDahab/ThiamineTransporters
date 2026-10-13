"""The TripleVar classifier and its self-contained artifact ("bundle").

A bundle holds everything needed to score new variants: network weights and
architecture, the fitted standardisation, the feature layout and class order.
The published notebooks saved bare state_dicts; ``Bundle.from_legacy_state_dict``
wraps one together with the scaler re-fitted on the original training split.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import torch
from torch import nn

from .features import FeatureLayout
from .labels import TRAINING_CLASSES

BUNDLE_FORMAT = "triplevar-bundle/1"
ACTIVATIONS = {"ReLU": nn.ReLU, "LeakyReLU": nn.LeakyReLU, "GELU": nn.GELU}


class TripleVarNet(nn.Module):
    """MLP: [Linear -> BatchNorm -> activation -> Dropout] x N -> Linear.

    Parameter names (``network.<i>``) match the published checkpoints.
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dims: Sequence[int],
        num_classes: int,
        dropout: float = 0.2,
        activation: str = "LeakyReLU",
    ):
        super().__init__()
        act = ACTIVATIONS[activation]
        layers: List[nn.Module] = []
        current = input_dim
        for width in hidden_dims:
            layers += [nn.Linear(current, width), nn.BatchNorm1d(width), act(), nn.Dropout(dropout)]
            current = width
        layers.append(nn.Linear(current, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)


@dataclass
class Bundle:
    state_dict: Dict[str, torch.Tensor]
    hidden_dims: List[int]
    activation: str
    dropout: float
    layout: FeatureLayout
    scaler_mean: np.ndarray
    scaler_scale: np.ndarray
    classes: List[int] = field(default_factory=lambda: list(TRAINING_CLASSES))
    metadata: Dict[str, Any] = field(default_factory=dict)

    @property
    def input_dim(self) -> int:
        return self.layout.dim

    def build(self, device: Union[str, torch.device] = "cpu") -> TripleVarNet:
        net = TripleVarNet(self.input_dim, self.hidden_dims, len(self.classes), self.dropout, self.activation)
        net.load_state_dict(self.state_dict)
        return net.to(device).eval()

    def standardise(self, X: np.ndarray) -> np.ndarray:
        if X.shape[1] != self.input_dim:
            raise ValueError(f"Expected {self.input_dim} features, got {X.shape[1]}")
        return ((X - self.scaler_mean) / self.scaler_scale).astype(np.float32)

    @torch.no_grad()
    def predict_proba(self, X: np.ndarray, device: Union[str, torch.device] = "cpu", batch_size: int = 4096) -> np.ndarray:
        net = self.build(device)
        Xs = self.standardise(np.asarray(X, dtype=np.float64))
        out = []
        for i in range(0, len(Xs), batch_size):
            logits = net(torch.from_numpy(Xs[i : i + batch_size]).to(device))
            out.append(torch.softmax(logits, dim=1).cpu().numpy())
        return np.concatenate(out) if out else np.zeros((0, len(self.classes)), dtype=np.float32)

    # -- persistence ---------------------------------------------------------
    def to_dict(self, half: bool = False) -> Dict[str, Any]:
        """``half=True`` stores the large weight matrices in float16 (halves the file;
        they are cast back to float32 on load). BatchNorm statistics stay float32."""

        def pack(v: torch.Tensor) -> torch.Tensor:
            v = v.detach().cpu()
            return v.half() if half and v.is_floating_point() and v.dim() == 2 else v

        return {
            "format": BUNDLE_FORMAT,
            "state_dict": {k: pack(v) for k, v in self.state_dict.items()},
            "hidden_dims": list(self.hidden_dims),
            "activation": self.activation,
            "dropout": self.dropout,
            "layout": self.layout.to_dict(),
            "scaler_mean": np.asarray(self.scaler_mean, dtype=np.float64),
            "scaler_scale": np.asarray(self.scaler_scale, dtype=np.float64),
            "classes": list(self.classes),
            "metadata": self.metadata,
        }

    def save(self, path: Union[str, Path]) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.to_dict(), path)

    @classmethod
    def load(cls, path: Union[str, Path]) -> "Bundle":
        raw = torch.load(str(path), map_location="cpu")
        if not isinstance(raw, dict) or raw.get("format") != BUNDLE_FORMAT:
            raise ValueError(f"{path} is not a TripleVar bundle (see Bundle.from_legacy_state_dict)")
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, raw: Dict[str, Any]) -> "Bundle":
        return cls(
            state_dict={k: (v.float() if v.dtype == torch.float16 else v) for k, v in raw["state_dict"].items()},
            hidden_dims=list(raw["hidden_dims"]),
            activation=raw["activation"],
            dropout=float(raw["dropout"]),
            layout=FeatureLayout(**raw["layout"]),
            scaler_mean=np.asarray(raw["scaler_mean"]),
            scaler_scale=np.asarray(raw["scaler_scale"]),
            classes=[int(c) for c in raw["classes"]],
            metadata=dict(raw.get("metadata", {})),
        )

    @classmethod
    def from_legacy_state_dict(
        cls,
        state_dict: Dict[str, torch.Tensor],
        activation: str,
        layout: FeatureLayout,
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> "Bundle":
        if "model_state_dict" in state_dict:
            state_dict = state_dict["model_state_dict"]
        linear = sorted(
            int(k.split(".")[1]) for k, v in state_dict.items() if k.endswith(".weight") and v.dim() == 2
        )
        widths = [state_dict[f"network.{i}.weight"].shape[0] for i in linear]
        input_dim = state_dict[f"network.{linear[0]}.weight"].shape[1]
        if input_dim != layout.dim:
            raise ValueError(f"Checkpoint expects {input_dim} inputs but layout has {layout.dim}")
        return cls(
            state_dict=dict(state_dict),
            hidden_dims=widths[:-1],
            activation=activation,
            dropout=0.0,
            layout=layout,
            scaler_mean=np.asarray(scaler_mean),
            scaler_scale=np.asarray(scaler_scale),
            classes=list(TRAINING_CLASSES),
            metadata=dict(metadata or {}),
        )
