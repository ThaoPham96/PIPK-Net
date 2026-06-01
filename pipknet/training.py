"""Nested 5-fold scaffold cross-validation training driver.

Reproduces the PIPK-Net training protocol for a single architecture variant:

  outer GroupKFold(5) over Bemis-Murcko scaffolds
    -> inner GroupKFold(5) grid search to select hyperparameters
    -> retrain the best configuration on the full outer-train split
       (with a 90/10 GroupShuffleSplit held out for early stopping)
    -> evaluate on the outer-test fold

Each outer fold writes a checkpoint directory compatible with the inference
CLI: ``best_outer_model.pt``, ``config.json`` (with ``best_hp``), ``scaler.json``,
``predictions.csv`` and ``metrics.csv``. The five outer-fold models form the
inference ensemble.

The default grid matches the published search space; it is large, so training
all variants is compute-intensive. Use ``--quick`` (or pass a reduced
``hp_space``) for a fast smoke run.
"""
from __future__ import annotations

import itertools
import json
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from torch_geometric.loader import DataLoader

from .models import GNNMultitask
from .engine import fit_one, evaluate

# Published hyperparameter search space (manuscript Methods 2.1.4).
FULL_HP_SPACE = {
    "hidden_dim": [64, 96, 128],
    "dropout": [0.1, 0.2],
    "lr": [1e-3, 5e-4],
    "weight_decay": [1e-4, 1e-3],
    "ion_emb_dim": [16, 32],         # variants B_ion / C_physio only
    "lambda_cons": [0.2, 0.5, 1.0],  # variant C_physio only
}
# Small grid for smoke testing.
QUICK_HP_SPACE = {"hidden_dim": [64], "dropout": [0.1], "lr": [1e-3],
                  "weight_decay": [1e-4], "ion_emb_dim": [16], "lambda_cons": [0.2]}


def hp_grid(variant: str, space: Dict[str, list]) -> List[dict]:
    """Enumerate hyperparameter combinations relevant to a variant."""
    keys = ["hidden_dim", "dropout", "lr", "weight_decay"]
    if variant != "A_baseline":
        keys.append("ion_emb_dim")
    if variant == "C_physio":
        keys.append("lambda_cons")
    combos = [dict(zip(keys, vals)) for vals in itertools.product(*[space[k] for k in keys])]
    return combos


def _build_model(variant: str, hp: dict, device) -> GNNMultitask:
    return GNNMultitask(
        hidden_dim=hp["hidden_dim"], ion_dim=4,
        ion_emb_dim=hp.get("ion_emb_dim", 16), out_dim=7,
        use_ion=(variant != "A_baseline"), dropout=hp["dropout"],
    ).to(device)


def _loader(items, batch_size=128, shuffle=False):
    return DataLoader(items, batch_size=batch_size, shuffle=shuffle)


def _val_score(model, loader, df_ref) -> float:
    metr, _ = evaluate(model, loader, df_ref)
    return float(np.nanmean(metr["Pearson"].values))


def nested_cv(dataset, df_ref, groups, variant: str, out_dir, *,
              hp_space: Optional[dict] = None, n_outer: int = 5, n_inner: int = 5,
              batch_size: int = 128, max_epochs: int = 200, patience: int = 30,
              device=None, seed: int = 42) -> pd.DataFrame:
    """Run nested CV for one variant and write per-fold checkpoints.

    Returns a long DataFrame of per-fold, per-task outer-test metrics.
    """
    assert variant in ("A_baseline", "B_ion", "C_physio")
    hp_space = hp_space or FULL_HP_SPACE
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(out_dir) / variant
    out_dir.mkdir(parents=True, exist_ok=True)
    mode = "C_physio" if variant == "C_physio" else variant

    idx_all = np.arange(len(dataset))
    groups = np.asarray(groups, dtype=object)
    combos = hp_grid(variant, hp_space)
    n_groups = len(np.unique(groups))
    if n_outer > n_groups:
        print(f"[train] only {n_groups} scaffold groups; reducing outer folds {n_outer} -> {n_groups}")
        n_outer = n_groups
    outer = GroupKFold(n_splits=n_outer)
    all_metrics = []

    for fold_i, (trv_idx, test_idx) in enumerate(outer.split(idx_all, groups=groups), 1):
        trv_groups = groups[trv_idx]

        # --- inner grid search ---
        best_hp, best_inner = None, -np.inf
        if len(combos) == 1:
            best_hp = combos[0]
        else:
            inner = GroupKFold(n_splits=min(n_inner, len(np.unique(trv_groups))))
            for hp in combos:
                scores = []
                for itr, ival in inner.split(trv_idx, groups=trv_groups):
                    tr = [dataset[trv_idx[i]] for i in itr]
                    va = [dataset[trv_idx[i]] for i in ival]
                    model = _build_model(variant, hp, device)
                    model, _ = fit_one(model, _loader(tr, batch_size, True), _loader(va, batch_size),
                                       df_ref, mode=mode, lambda_cons=hp.get("lambda_cons", 0.2),
                                       patience=patience, lr=hp["lr"],
                                       weight_decay=hp["weight_decay"], max_epochs=max_epochs)
                    scores.append(_val_score(model, _loader(va, batch_size), df_ref))
                mean_score = float(np.nanmean(scores))
                if mean_score > best_inner:
                    best_inner, best_hp = mean_score, hp
        best_hp = {**best_hp, "batch_size": batch_size}

        # --- retrain best HP on full outer-train (90/10 early-stopping split) ---
        gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=seed)
        rel_tr, rel_va = next(gss.split(trv_idx, groups=trv_groups))
        tr = [dataset[trv_idx[i]] for i in rel_tr]
        va = [dataset[trv_idx[i]] for i in rel_va]
        model = _build_model(variant, best_hp, device)
        model, scaler_stats = fit_one(model, _loader(tr, batch_size, True), _loader(va, batch_size),
                                      df_ref, mode=mode, lambda_cons=best_hp.get("lambda_cons", 0.2),
                                      patience=patience, lr=best_hp["lr"],
                                      weight_decay=best_hp["weight_decay"], max_epochs=max_epochs)

        # --- evaluate on outer-test + save artefacts ---
        test_items = [dataset[i] for i in test_idx]
        metr, preds = evaluate(model, _loader(test_items, batch_size), df_ref)
        metr["outer_fold"], metr["config"] = fold_i, variant
        all_metrics.append(metr)

        fold_dir = out_dir / f"fold_{fold_i}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        torch.save({"model_state": model.state_dict()}, fold_dir / "best_outer_model.pt")
        json.dump({"variant": variant, "best_hp": best_hp, "outer_fold": fold_i},
                  open(fold_dir / "config.json", "w"), indent=2)
        json.dump(scaler_stats, open(fold_dir / "scaler.json", "w"))
        preds.to_csv(fold_dir / "predictions.csv")
        metr.to_csv(fold_dir / "metrics.csv", index=False)
        print(f"[train] {variant} fold {fold_i}: best_hp={best_hp} "
              f"-> mean test Pearson={np.nanmean(metr['Pearson'].values):.3f}")

    results = pd.concat(all_metrics, ignore_index=True)
    results.to_csv(out_dir / "nested_cv_results.csv", index=False)
    return results


def train_from_csv(csv_path, variant: str, out_dir, *, quick: bool = False,
                   max_epochs: int = 200, patience: int = 30, batch_size: int = 128):
    """Convenience entry point: load a CSV and run nested CV for one variant."""
    from .data import load_dataset_from_csv
    dataset, df_ref, groups = load_dataset_from_csv(csv_path)
    print(f"[train] loaded {len(dataset)} molecules, {len(np.unique(groups))} scaffolds")
    return nested_cv(dataset, df_ref, groups, variant, out_dir,
                     hp_space=QUICK_HP_SPACE if quick else FULL_HP_SPACE,
                     max_epochs=max_epochs, patience=patience, batch_size=batch_size)
