"""Training loop for the TripleVar classifier."""

from __future__ import annotations

import copy
import logging
import random
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from .features import FeatureLayout
from .labels import LIKELY_PATHOGENIC, PATHOGENIC, TRAINING_CLASSES
from .metrics import Scores
from .model import Bundle, TripleVarNet

log = logging.getLogger(__name__)


@dataclass
class TrainConfig:
    hidden_dims: List[int] = field(default_factory=lambda: [2048, 1024, 512, 256])
    activation: str = "LeakyReLU"
    dropout: float = 0.2
    optimizer: str = "RMSprop"
    lr: float = 5e-4
    weight_decay: float = 1e-4
    batch_size: int = 64
    max_epochs: int = 300
    early_stopping_patience: int = 100
    lr_patience: int = 25
    lr_factor: float = 0.2
    class_balance_beta: float = 0.999
    label_smoothing: float = 0.0
    # Probability of replacing a training row's allele-frequency block with the
    # "no frequency data" encoding (all zeros), so the model also learns to
    # score variants that are absent from population databases.
    tabular_dropout: float = 0.0
    # "legacy": val accuracy + 0.1 * exact pathogenic-class recall (published model)
    # "plp_auc": val ROC-AUC of P/LP vs B/LB
    selection_metric: str = "legacy"
    seed: int = 42

    def to_dict(self) -> Dict:
        return asdict(self)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class ClassBalancedCrossEntropy(nn.Module):
    """Cross-entropy weighted by the 'effective number of samples' (Cui et al., 2019)."""

    def __init__(self, samples_per_class: np.ndarray, beta: float = 0.999, label_smoothing: float = 0.0):
        super().__init__()
        counts = torch.as_tensor(samples_per_class, dtype=torch.float32)
        effective = 1.0 - torch.pow(beta, counts)
        weights = (1.0 - beta) / (effective + 1e-8)
        self.register_buffer("weights", weights / weights.sum() * len(counts))
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return nn.functional.cross_entropy(
            logits, targets, weight=self.weights.to(logits.device), label_smoothing=self.label_smoothing
        )


def _selection_score(metric: str, probs: np.ndarray, y_enc: np.ndarray, classes: List[int]) -> float:
    y = np.asarray(classes)[y_enc]
    s = Scores(probs, classes)
    if metric == "legacy":
        acc = float(np.mean(s.pred == y))
        path = np.isin(y, [PATHOGENIC, LIKELY_PATHOGENIC])
        recall = float(np.mean(s.pred[path] == y[path])) if path.any() else 0.0
        return acc + 0.1 * recall
    if metric == "plp_auc":
        variants = ~np.isin(y, [3])
        target = np.isin(y[variants], [PATHOGENIC, LIKELY_PATHOGENIC])
        if 0 < target.sum() < len(target):
            return float(roc_auc_score(target, s.pathogenic_given_variant[variants]))
        return 0.0
    raise ValueError(f"Unknown selection metric {metric!r}")


@torch.no_grad()
def _predict(net: nn.Module, X: torch.Tensor) -> np.ndarray:
    net.eval()
    return torch.softmax(net(X), dim=1).cpu().numpy()


def train(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    layout: FeatureLayout,
    config: TrainConfig,
    device: Optional[str] = None,
) -> Tuple[Bundle, Dict[str, List[float]]]:
    """Fit one model; early stopping and model selection use the validation set only."""
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    set_seed(config.seed)
    classes = list(TRAINING_CLASSES)
    enc = LabelEncoder().fit(classes)
    ytr, yva = enc.transform(y_train), enc.transform(y_val)

    scaler = StandardScaler().fit(X_train)
    Xtr = torch.tensor(scaler.transform(X_train), dtype=torch.float32)
    Xva = torch.tensor(scaler.transform(X_val), dtype=torch.float32).to(device)

    tab = layout.slices().get("tabular")
    tab_missing = None
    if tab is not None and config.tabular_dropout > 0:
        zeros = np.zeros((1, X_train.shape[1]))
        tab_missing = torch.tensor(scaler.transform(zeros)[0, tab], dtype=torch.float32).to(device)

    gen = torch.Generator().manual_seed(config.seed)
    loader = DataLoader(
        TensorDataset(Xtr, torch.tensor(ytr, dtype=torch.long)), batch_size=config.batch_size, shuffle=True, generator=gen
    )
    net = TripleVarNet(layout.dim, config.hidden_dims, len(classes), config.dropout, config.activation).to(device)
    criterion = ClassBalancedCrossEntropy(
        np.bincount(ytr, minlength=len(classes)), config.class_balance_beta, config.label_smoothing
    ).to(device)
    opt_cls = {"RMSprop": torch.optim.RMSprop, "AdamW": torch.optim.AdamW, "Adam": torch.optim.Adam}[config.optimizer]
    optimizer = opt_cls(net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=config.lr_factor, patience=config.lr_patience
    )

    history: Dict[str, List[float]] = {"train_loss": [], "val_score": []}
    best_score, best_state, stale = -np.inf, None, 0
    for epoch in range(config.max_epochs):
        net.train()
        total = 0.0
        for xb, yb in loader:
            if len(xb) < 2:  # BatchNorm needs >1 sample
                continue
            xb, yb = xb.to(device), yb.to(device)
            if tab_missing is not None:
                drop = torch.rand(len(xb), device=device) < config.tabular_dropout
                if drop.any():
                    xb = xb.clone()
                    xb[drop, tab] = tab_missing
            optimizer.zero_grad()
            loss = criterion(net(xb), yb)
            loss.backward()
            optimizer.step()
            total += loss.item() * len(xb)
        score = _selection_score(config.selection_metric, _predict(net, Xva), yva, classes)
        scheduler.step(score)
        history["train_loss"].append(total / len(Xtr))
        history["val_score"].append(score)
        if score > best_score:
            best_score, best_state, stale = score, copy.deepcopy(net.state_dict()), 0
        else:
            stale += 1
            if stale >= config.early_stopping_patience:
                break
    log.info("stopped after %d epochs; best validation %s = %.4f", epoch + 1, config.selection_metric, best_score)

    bundle = Bundle(
        state_dict={k: v.cpu() for k, v in best_state.items()},
        hidden_dims=list(config.hidden_dims),
        activation=config.activation,
        dropout=config.dropout,
        layout=layout,
        scaler_mean=scaler.mean_,
        scaler_scale=scaler.scale_,
        classes=classes,
        metadata={"train_config": config.to_dict(), "best_val_score": best_score, "epochs": epoch + 1},
    )
    return bundle, history
