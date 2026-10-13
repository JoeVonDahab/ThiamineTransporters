"""Evaluate a configuration with leakage-safe resampling.

``split: cv`` (default for new work) runs repeated grouped K-fold CV: every
labeled variant is predicted exactly once per repeat by a model that never saw
its rsID; the next fold serves as validation set for early stopping. The K
fold models form an ensemble that scores the external sets.
``split: published`` re-runs the original row-level seed-42 split.

Model selection only ever uses validation data; the yardstick for comparing
configurations is the mean validation ROC-AUC of P/LP vs B/LB ("val_auc").
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from .config import RunConfig
from .dataset import FeatureSet
from .ensemble import Ensemble
from .labels import NORMAL, VUS, is_benign, is_pathogenic
from .legacy import published_split
from .metrics import Scores, evaluate, evaluate_external
from .splits import grouped_folds
from .training import train

log = logging.getLogger(__name__)


@dataclass
class ExternalSet:
    name: str
    features: FeatureSet
    y_pathogenic: np.ndarray  # bool per row


def _val_auc(probs: np.ndarray, y: np.ndarray) -> Optional[float]:
    variants = is_pathogenic(y) | is_benign(y)
    target = is_pathogenic(y[variants])
    if 0 < target.sum() < len(target):
        return float(roc_auc_score(target, Scores(probs[variants]).pathogenic_given_variant))
    return None


def _train_rows(cfg: RunConfig, labeled: FeatureSet, rows: np.ndarray, seed: int, drop_mask: np.ndarray) -> np.ndarray:
    rows = rows[~drop_mask[rows]]
    if cfg.drop_normal_fraction > 0:
        normals = rows[labeled.y[rows] == NORMAL]
        rng = np.random.default_rng(seed)
        drop = rng.choice(normals, int(round(len(normals) * cfg.drop_normal_fraction)), replace=False)
        rows = np.setdiff1d(rows, drop)
    return rows


def _fit(cfg, labeled, X, tr, va, seed, device):
    bundle, _ = train(X[tr], labeled.y[tr], X[va], labeled.y[va], cfg.layout, replace(cfg.train, seed=seed), device=device)
    return bundle


def fit_final(
    cfg: RunConfig, labeled: FeatureSet, seed: int, device: Optional[str] = None,
    drop_from_training: Optional[np.ndarray] = None,
):
    """The release model: train on every labeled rsID except one grouped validation fold
    (used only for early stopping), with the same procedure the CV evaluates."""
    rows = np.flatnonzero(labeled.y != VUS)
    ids = labeled.meta["snp_id"].astype(str).to_numpy()[rows]
    folds = [rows[te] for _, te in grouped_folds(ids, labeled.y[rows], n_splits=cfg.n_folds, seed=seed)]
    va = folds[0]
    drop = np.zeros(len(labeled), bool) if drop_from_training is None else drop_from_training
    tr = _train_rows(cfg, labeled, np.setdiff1d(rows, va), seed, drop)
    va_fit = va[~drop[va]]
    X = labeled.X(cfg.layout)
    bundle = _fit(cfg, labeled, X, tr, va_fit, seed, device)
    bundle.metadata.update({
        "run_config": cfg.name, "release": True, "n_train_rows": int(len(tr)), "n_val_rows": int(len(va_fit)),
        "val_auc": _val_auc(bundle.predict_proba(X[va_fit]), labeled.y[va_fit]),
    })
    return bundle


def _summarise(values: List[Optional[float]]) -> Dict[str, float]:
    a = np.asarray([v for v in values if v is not None], dtype=float)
    if len(a) == 0:
        return {"mean": float("nan"), "sd": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "sd": float(a.std(ddof=1)) if len(a) > 1 else 0.0, "n": int(len(a))}


def _flatten(d: Dict, prefix: str = "") -> Dict[str, float]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(_flatten(v, key + "."))
        elif isinstance(v, (int, float)) and v is not None:
            out[key] = float(v)
    return out


def run(
    cfg: RunConfig,
    labeled: FeatureSet,
    out_dir: Path,
    externals: Optional[List[ExternalSet]] = None,
    device: Optional[str] = None,
    drop_from_training: Optional[np.ndarray] = None,
    clean_eval_mask: Optional[np.ndarray] = None,
    save_model: bool = True,
) -> Dict:
    """``drop_from_training``: rows never used to fit/early-stop (still predicted when in a test fold).
    ``clean_eval_mask``: rows whose labels are trusted; an extra '<metric>_clean' evaluation uses only these."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    externals = externals or []
    X = labeled.X(cfg.layout)
    drop = np.zeros(len(labeled), bool) if drop_from_training is None else drop_from_training
    per_seed, first_models = [], None

    for seed in cfg.seeds:
        if cfg.split == "published":
            s = published_split(labeled, cfg.layout)
            folds = [(np.concatenate([s.train, s.val]), s.val, s.test)]
        else:
            rows = np.flatnonzero(labeled.y != VUS)
            ids = labeled.meta["snp_id"].astype(str).to_numpy()[rows]
            k_folds = [rows[te] for _, te in grouped_folds(ids, labeled.y[rows], n_splits=cfg.n_folds, seed=seed)]
            folds = []
            for k in range(cfg.n_folds):
                te, va = k_folds[k], k_folds[(k + 1) % cfg.n_folds]
                tr = np.setdiff1d(rows, np.concatenate([te, va]))
                folds.append((tr, va, te))

        oof = np.full((len(labeled), 5), np.nan)
        models, val_aucs = [], []
        for tr_all, va, te in folds:
            tr = _train_rows(cfg, labeled, np.setdiff1d(tr_all, va), seed, drop)
            va_fit = va[~drop[va]]
            bundle = _fit(cfg, labeled, X, tr, va_fit, seed, device)
            val_aucs.append(_val_auc(bundle.predict_proba(X[va_fit]), labeled.y[va_fit]))
            oof[te] = bundle.predict_proba(X[te])
            models.append(bundle)
        tested = ~np.isnan(oof[:, 0])
        result = {"seed": seed, "val_auc": _summarise(val_aucs)["mean"], "test": evaluate(oof[tested], labeled.y[tested])}
        if clean_eval_mask is not None:
            m = tested & clean_eval_mask
            result["test_clean"] = evaluate(oof[m], labeled.y[m])
        ens = Ensemble(models)
        for ext in externals:
            probs = ens.predict_proba(ext.features.X(cfg.layout))
            result[f"external:{ext.name}"] = evaluate_external(probs, ext.y_pathogenic)
            if seed == cfg.seeds[0]:
                pd.DataFrame(probs, columns=["p_P", "p_LP", "p_LB", "p_B", "p_normal"]).assign(
                    **{c: ext.features.meta[c].values for c in ext.features.meta.columns if c in (
                        "variant_id", "protein_change", "gene", "label_binary", "training_category",
                        "feature_source", "consequence", "is_missense")}
                ).to_csv(out_dir / f"{cfg.name}.{ext.name}.predictions.csv", index=False)
        if seed == cfg.seeds[0]:
            first_models = ens
            pd.DataFrame(oof, columns=["p_P", "p_LP", "p_LB", "p_B", "p_normal"]).assign(
                snp_id=labeled.meta["snp_id"].values, y=labeled.y
            )[tested].to_csv(out_dir / f"{cfg.name}.oof.csv", index=False)
        per_seed.append(result)
        log.info("%s seed %d: val_auc=%.3f test P/LP AUC=%.3f", cfg.name, seed, result["val_auc"],
                 result["test"].get("PLP_vs_BLB", {}).get("auc_renormalised", float("nan")))

    flat = [_flatten({k: v for k, v in r.items() if k != "seed"}) for r in per_seed]
    keys = sorted(set().union(*flat))
    summary = {
        "config": cfg.name, "split": cfg.split, "seeds": cfg.seeds, "n_folds": cfg.n_folds,
        "metrics": {k: _summarise([f.get(k) for f in flat]) for k in keys},
    }
    if save_model and first_models is not None:
        first_models.metadata.update({"run_config": cfg.name, "split": cfg.split, "seed": cfg.seeds[0]})
        first_models.save(out_dir / f"{cfg.name}.ensemble.pt")
    (out_dir / f"{cfg.name}.per_seed.json").write_text(json.dumps(per_seed, indent=2))
    (out_dir / f"{cfg.name}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary
