import numpy as np
import pytest
import torch

from triplevar.features import FeatureLayout, TABULAR_COLUMNS, TABULAR_DIM, fuse, tabular_matrix
from triplevar.model import Bundle, TripleVarNet
from triplevar.splits import grouped_train_val_test
from triplevar.training import TrainConfig, train

import pandas as pd


def test_tabular_layout():
    assert TABULAR_DIM == 71
    assert FeatureLayout().dim == 6471
    assert FeatureLayout(use_tabular=False).dim == 6400
    df = pd.DataFrame({"TOPMED_freq": [0.5], "unrelated": [1.0]})
    row = tabular_matrix(df)[0]
    assert row[TABULAR_COLUMNS.index("TOPMED_freq")] == 0.5
    assert row.sum() == 0.5


def test_fuse_checks_shapes():
    layout = FeatureLayout()
    with pytest.raises(ValueError):
        fuse(np.zeros((2, 10)), np.zeros((2, 2560)), np.zeros((2, 71)), layout)


def test_legacy_parameter_names_load():
    net = TripleVarNet(6471, [2048, 1024, 512, 256], 5)
    keys = set(net.state_dict())
    assert {"network.0.weight", "network.1.running_mean", "network.16.weight"} <= keys


def test_bundle_roundtrip(tmp_path):
    layout = FeatureLayout(use_dna=False, use_protein=False, use_tabular=True)
    net = TripleVarNet(layout.dim, [8], 5, activation="ReLU")
    b = Bundle(net.state_dict(), [8], "ReLU", 0.0, layout, np.zeros(layout.dim), np.ones(layout.dim))
    b.save(tmp_path / "m.pt")
    b2 = Bundle.load(tmp_path / "m.pt")
    X = np.random.default_rng(0).normal(size=(4, layout.dim))
    np.testing.assert_allclose(b.predict_proba(X), b2.predict_proba(X), atol=1e-6)


def test_half_precision_ensemble_roundtrip(tmp_path):
    from triplevar.ensemble import Ensemble, load_model

    layout = FeatureLayout(use_dna=False, use_protein=False, use_tabular=True)
    members = []
    for seed in (0, 1):
        torch.manual_seed(seed)
        net = TripleVarNet(layout.dim, [16], 5)
        members.append(Bundle(net.state_dict(), [16], "LeakyReLU", 0.0, layout, np.zeros(layout.dim), np.ones(layout.dim)))
    ens = Ensemble(members)
    ens.save(tmp_path / "e.pt", half=True)
    loaded = load_model(tmp_path / "e.pt")
    X = np.random.default_rng(1).normal(size=(8, layout.dim))
    np.testing.assert_allclose(ens.predict_proba(X), loaded.predict_proba(X), atol=2e-3)
    assert loaded.members[0].state_dict["network.0.weight"].dtype == torch.float32


def test_grouped_split_keeps_rsids_together():
    rng = np.random.default_rng(0)
    ids = np.repeat(np.arange(300), 2).astype(str)
    y = np.where(np.arange(600) % 2 == 0, rng.choice([-2, -1, 1, 2], 600), 3)
    tr, va, te = grouped_train_val_test(ids, y, seed=1)
    assert not (set(ids[tr]) & set(ids[te])) and not (set(ids[va]) & set(ids[te]))
    assert len(tr) + len(va) + len(te) == 600


def test_training_smoke():
    rng = np.random.default_rng(0)
    layout = FeatureLayout(use_dna=False, use_protein=False, use_tabular=True)
    y = np.array([-2, -1, 1, 2, 3] * 40)
    X = rng.normal(size=(len(y), layout.dim)) + (y[:, None] * 0.5)
    cfg = TrainConfig(hidden_dims=[32], max_epochs=5, early_stopping_patience=5, tabular_dropout=0.2)
    bundle, history = train(X[:150], y[:150], X[150:], y[150:], layout, cfg, device="cpu")
    assert len(history["val_score"]) == 5
    assert bundle.predict_proba(X[:3]).shape == (3, 5)
