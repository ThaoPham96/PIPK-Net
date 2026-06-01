"""Smoke tests for the pipknet package.

Fast, dependency-light checks that each module imports and its core functions
behave. Run with ``pytest`` from the repo root, or directly:
    python tests/test_smoke.py
"""
import numpy as np
import torch
from torch_geometric.loader import DataLoader

import pipknet
from pipknet.featurizers import smiles_to_pyg, ION_MAP
from pipknet.models import GNNMultitask
from pipknet.engine import to_original_units
from pipknet.utils import TaskScaler, get_generic_scaffold, extract_targets
from pipknet.data import apply_pk_transforms, build_dataset
from pipknet import training


def test_package_exports():
    assert pipknet.__version__
    for name in ("GNNMultitask", "smiles_to_pyg", "train_from_csv", "to_original_units"):
        assert name in pipknet.__all__


def test_smiles_to_pyg_feature_dim():
    data = smiles_to_pyg("CCO", "neutral")
    assert data is not None
    # 14 atom types + 5 degrees + 3 formal charges + 1 aromatic flag
    assert data.x.shape[1] == 23
    assert int(data.ion_feat.item()) == ION_MAP["neutral"]
    assert data.edge_index.shape[0] == 2


def test_smiles_to_pyg_invalid_returns_none():
    assert smiles_to_pyg("not_a_smiles", "neutral") is None


def test_model_forward_shape_and_no_dead_params():
    model = GNNMultitask(hidden_dim=32, ion_emb_dim=8, use_ion=True, dropout=0.1)
    batch = next(iter(DataLoader([smiles_to_pyg("CCO", "neutral"),
                                  smiles_to_pyg("c1ccccc1", "anionic")], batch_size=2)))
    out = model(batch)
    assert out.shape == (2, 7)
    # the model must carry no scalar parameters outside the conv/embedding/MLP layers
    names = dict(model.named_parameters())
    for extra in ("cons_logvar", "pbpk_a", "pbpk_b"):
        assert extra not in names


def test_to_original_units_roundtrip():
    # log10/logit10 inputs -> physical units
    p = np.zeros((1, 7))                      # all zeros
    out = to_original_units(p)
    assert abs(out["Vd (L)"][0] - 1.0) < 1e-9        # 10**0 = 1
    assert abs(out["F (%)"][0] - 50.0) < 1e-6        # logit10(0) -> 50%


def test_get_generic_scaffold():
    s = get_generic_scaffold("c1ccccc1CCN")
    assert isinstance(s, str) and len(s) > 0
    assert get_generic_scaffold(None) is None


def test_taskscaler_fit_from_loader():
    df = _toy_df()
    df = apply_pk_transforms(df).reset_index(drop=True)
    df["ORIG_INDEX"] = np.arange(len(df))
    df = df.set_index("ORIG_INDEX", drop=False)
    ds = build_dataset(df)
    scaler = TaskScaler().fit_from_loader(DataLoader(ds, batch_size=4), device="cpu")
    assert scaler.means.shape == (7,)
    assert scaler.stds.shape == (7,) and (scaler.stds > 0).all()


def test_extract_targets_mask():
    df = apply_pk_transforms(_toy_df())
    y, mask = extract_targets(df.iloc[0])
    assert y.shape == (7,) and mask.shape == (7,)
    assert mask.dtype == torch.bool


def test_training_and_predictor_roundtrip(tmp_path=None):
    import tempfile, pathlib
    from pipknet.inference import PIPKNetPredictor
    out = pathlib.Path(tmp_path or tempfile.mkdtemp())
    df = _toy_df()
    csv = out / "toy.csv"
    df.to_csv(csv, index=False)
    res = training.train_from_csv(csv, "C_physio", out / "ckpt", quick=True,
                                  max_epochs=1, patience=1, batch_size=4)
    assert len(res) > 0
    variant_dir = out / "ckpt" / "C_physio"
    assert (variant_dir / "fold_1" / "best_outer_model.pt").exists()
    assert (variant_dir / "fold_1" / "config.json").exists()

    # load the freshly trained ensemble and predict single + batch
    predictor = PIPKNetPredictor(variant_dir, device="cpu")
    single = predictor.predict("CCO", ion_type="neutral")
    assert list(single.index) == list(pipknet.TASKS_WITH_UNITS)
    assert {"Mean", "Std"} <= set(single.columns)
    batch = predictor.predict_batch(df)
    assert len(batch) == len(df)
    assert "Vd (L)_mean" in batch.columns


def _toy_df():
    """Six structurally distinct drugs with partial PK labels."""
    import pandas as pd
    rows = [
        ("DRUGA", "CCO", "neutral", 5.0, 50.0, 7.0, 80.0, 90.0, 1.2, 2.0),
        ("DRUGB", "c1ccccc1", "anionic", 2.0, 20.0, 6.0, 60.0, 70.0, 0.5, 1.0),
        ("DRUGC", "C1CCCCC1", "cationic", 12.0, 300.0, 18.0, 40.0, 30.0, 2.0, 3.0),
        ("DRUGD", "CC(=O)O", "anionic", 1.0, 10.0, 9.0, 95.0, 20.0, 3.0, 0.5),
        ("DRUGE", "c1ccncc1", "cationic", 8.0, 150.0, 12.0, 55.0, 60.0, 1.0, 1.5),
        ("DRUGF", "CCN(CC)CC", "cationic", 20.0, 500.0, 25.0, 70.0, 50.0, 0.8, 2.5),
    ]
    cols = ["Name", "SMILES", "IonType", "t1/2(hour)", "VD(liter)", "Cl(liter/hour)",
            "F(percentage)", "PPB(percentage)", "Cmax_uM", "Tmax"]
    return pd.DataFrame(rows, columns=cols)


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in fns:
        try:
            fn()
            print(f"PASS {fn.__name__}")
            passed += 1
        except Exception as e:
            print(f"FAIL {fn.__name__}: {type(e).__name__}: {e}")
    print(f"\n{passed}/{len(fns)} smoke tests passed")
