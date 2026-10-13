"""Reproduce the numbers in the submitted ASCPT 2026 abstract.

The abstract model is the checkpoint saved on 2025-09-11 16:13
(``best_model__(2025_09_11)_16|13.pt``, LeakyReLU). Combined with the
published seed-42 split, it reproduces the saved abstract predictions for all
31,199 pool variants to within 3e-6.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Union

import numpy as np
import pandas as pd
import torch

from .features import FeatureLayout
from .labels import LIKELY_PATHOGENIC, PATHOGENIC, VUS
from .legacy import LegacyArtifacts, published_split
from .metrics import Scores, binary_report, evaluate
from .model import Bundle

log = logging.getLogger(__name__)

ABSTRACT_CHECKPOINT = "models/best_model__(2025_09_11)_16|13.pt"
ABSTRACT_ACTIVATION = "LeakyReLU"

# As written in the submitted abstract (Figure 2 for the per-metric values).
ABSTRACT_CLAIMS = {
    "clinvar_strict_P_vs_B": {"accuracy": 0.933, "precision": 1.0, "recall": 0.846, "specificity": 1.0, "f1": 0.917, "auc": 1.0},
    "clinvar_PLP_vs_BLB": {"accuracy": 0.937, "auc": 0.913},
    "literature_external": {"n": 15, "accuracy": 0.80, "precision": 0.90, "recall": 0.82, "specificity": 0.75, "f1": 0.857},
    "prioritised": {"pathogenic_high_confidence": 111, "likely_pathogenic_lower_confidence": 303},
    "split": {"train": 604, "val": 130, "test": 130, "synthetic_normal": 962},
}


def _round(d):
    if isinstance(d, dict):
        return {k: _round(v) for k, v in d.items()}
    if isinstance(d, float):
        return round(d, 4)
    return d


def reproduce_abstract(legacy_root: Union[str, Path], out_dir: Union[str, Path], labels_csv: Union[str, Path]) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    art = LegacyArtifacts(legacy_root)
    layout = FeatureLayout()

    labeled = art.labeled()
    split = published_split(labeled, layout)
    state = torch.load(str(art.path(ABSTRACT_CHECKPOINT)), map_location="cpu")
    bundle = Bundle.from_legacy_state_dict(
        state, ABSTRACT_ACTIVATION, layout, split.scaler.mean_, split.scaler.scale_,
        metadata={"source_checkpoint": ABSTRACT_CHECKPOINT, "trained": "2025-09-11", "split": "published row-level seed-42"},
    )
    bundle.save(out_dir / "triplevar_abstract_2025-09-11.pt")

    X = labeled.X(layout)
    report: Dict = {"claims": ABSTRACT_CLAIMS}

    n_variant = lambda idx: int(np.sum(labeled.y[idx] != 3))  # noqa: E731
    report["split"] = {
        "train": n_variant(split.train), "val": n_variant(split.val), "test": n_variant(split.test),
        "synthetic_normal": int(np.sum(labeled.y == 3)),
    }
    report["clinvar_test"] = evaluate(bundle.predict_proba(X[split.test]), labeled.y[split.test])

    # literature set, with the labels used for the abstract
    lit = art.literature()
    labels = pd.read_csv(labels_csv)
    lit_meta = lit.meta.merge(labels, on="protein_change", how="left")
    has_label = lit_meta["label_abstract"].notna().to_numpy()
    y_path = lit_meta.loc[has_label, "label_abstract"].str.contains("Pathogenic").to_numpy()
    s = Scores(bundle.predict_proba(lit.X(layout)))
    report["literature_external"] = {
        "rule_strict_ratio_gt_0.5": binary_report(y_path, (s.strict_pathogenic > 0.5)[has_label], s.strict_pathogenic[has_label]).as_dict(),
        "rule_grouped_argmax": binary_report(y_path, s.call_grouped()[has_label], s.pathogenic_given_variant[has_label]).as_dict(),
    }
    pd.DataFrame({
        "protein_change": lit_meta["protein_change"], "label_abstract": lit_meta["label_abstract"],
        "predicted_class": s.pred, "p_strict": s.strict_pathogenic, "p_pathogenic": s.pathogenic_given_variant,
    }).to_csv(out_dir / "literature_predictions.csv", index=False)

    # prioritisation: VUS rows of the labeled set + the unlabeled dbSNP pool
    vus = labeled.y == VUS
    pool = art.pool()
    pred_vus = Scores(bundle.predict_proba(X[vus])).pred
    pred_pool = Scores(bundle.predict_proba(pool.X(layout))).pred
    report["prioritised"] = {
        "pathogenic_high_confidence": int((pred_vus == PATHOGENIC).sum() + (pred_pool == PATHOGENIC).sum()),
        "likely_pathogenic_lower_confidence": int((pred_vus == LIKELY_PATHOGENIC).sum() + (pred_pool == LIKELY_PATHOGENIC).sum()),
        "vus_rows": int(vus.sum()),
        "pool_rows": len(pool),
    }

    report = _round(report)
    (out_dir / "abstract_reproduction.json").write_text(json.dumps(report, indent=2))
    log.info("wrote %s", out_dir / "abstract_reproduction.json")
    return report
