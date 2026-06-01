"""High-level inference API for PIPK-Net.

Load a trained 5-fold ensemble once with :class:`PIPKNetPredictor`, then predict
the seven systemic PK parameters for a single SMILES or a batch. Predictions are
returned in physical units as the ensemble mean +/- standard deviation across the
five folds (a simple model-uncertainty estimate).

Example
-------
>>> from pipknet.inference import PIPKNetPredictor
>>> predictor = PIPKNetPredictor("checkpoints/C_physio")
>>> predictor.predict("CN1C=NC2=C1C(=O)N(C(=O)N2C)C", ion_type="neutral")
              Mean       Std
Parameter
t_half (h)    5.57      2.19
Vd (L)      126.06     56.30
...
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Union

import numpy as np
import pandas as pd
import torch
from torch_geometric.loader import DataLoader

from .models import GNNMultitask, TASKS_WITH_UNITS
from .featurizers import smiles_to_pyg
from .engine import to_original_units

ION_CHOICES = ("anionic", "cationic", "neutral", "zwitterionic")


class PIPKNetPredictor:
    """A loaded PIPK-Net fold ensemble ready for repeated prediction.

    Args:
        weights_dir: path to a variant checkpoint folder (e.g. ``checkpoints/C_physio``)
            containing ``fold_{1..5}/`` with ``best_outer_model.pt`` and ``config.json``.
        device: torch device; defaults to CUDA when available, else CPU.
    """

    def __init__(self, weights_dir: Union[str, Path], device: Optional[torch.device] = None):
        self.weights_dir = Path(weights_dir)
        self.variant = self.weights_dir.name
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.models = self._load_folds()
        if not self.models:
            raise FileNotFoundError(f"No fold checkpoints found under {self.weights_dir}")

    def _load_folds(self):
        models = []
        for fold in range(1, 6):
            fold_dir = self.weights_dir / f"fold_{fold}"
            model_path = fold_dir / "best_outer_model.pt"
            config_path = fold_dir / "config.json"
            if not (model_path.exists() and config_path.exists()):
                continue
            hp = json.load(open(config_path))["best_hp"]
            model = GNNMultitask(
                hidden_dim=hp["hidden_dim"], ion_dim=4,
                ion_emb_dim=hp.get("ion_emb_dim", 16), out_dim=7,
                use_ion=(self.variant != "A_baseline"), dropout=hp["dropout"],
            ).to(self.device)
            ck = torch.load(model_path, map_location=self.device, weights_only=False)
            model.load_state_dict(ck["model_state"], strict=False)
            model.eval()
            models.append(model)
        return models

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict(self, smiles: str, ion_type: str = "neutral",
                as_frame: bool = True) -> Union[pd.DataFrame, dict]:
        """Predict the 7 PK parameters for one molecule.

        Returns a DataFrame indexed by parameter with ``Mean``/``Std`` columns
        (or, if ``as_frame=False``, a ``{parameter: (mean, std)}`` dict).
        """
        g = smiles_to_pyg(smiles, ion_type)
        if g is None:
            raise ValueError(f"Could not parse SMILES: {smiles!r}")
        g = g.to(self.device)
        fold_rows = []
        for model in self.models:
            phys = to_original_units(model(g).cpu().numpy())   # dict of length-1 arrays
            fold_rows.append({k: float(v[0]) for k, v in phys.items()})
        fold_df = pd.DataFrame(fold_rows)[TASKS_WITH_UNITS]
        summary = pd.DataFrame({"Mean": fold_df.mean(), "Std": fold_df.std(ddof=0)})
        summary.index.name = "Parameter"
        if as_frame:
            return summary
        return {p: (float(summary.loc[p, "Mean"]), float(summary.loc[p, "Std"])) for p in summary.index}

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, molecules: Union[pd.DataFrame, str, Path, Iterable[dict]],
                      smiles_col: str = "SMILES", ion_col: str = "IonType",
                      name_col: str = "Name", batch_size: int = 64) -> pd.DataFrame:
        """Predict PK parameters for many molecules.

        ``molecules`` may be a DataFrame, a path to a CSV, or an iterable of dicts,
        each providing a SMILES (and optionally an IonType and a Name). Returns one
        row per (parseable) molecule with ``<parameter>_mean``/``<parameter>_std``
        columns. Unparseable SMILES are skipped with a warning.
        """
        df = self._coerce_to_frame(molecules, smiles_col, ion_col, name_col)

        graphs, kept = [], []
        for i, row in df.iterrows():
            g = smiles_to_pyg(row[smiles_col], row.get(ion_col, "neutral"))
            if g is None:
                print(f"[predict_batch] skipping unparseable SMILES: {row[smiles_col]!r}")
                continue
            graphs.append(g)
            kept.append(i)
        if not graphs:
            return pd.DataFrame()

        loader = DataLoader(graphs, batch_size=batch_size, shuffle=False)
        # per fold: (N, 7) in log/logit space, concatenated in loader order
        fold_logits = []
        for model in self.models:
            outs = [model(b.to(self.device)).cpu().numpy() for b in loader]
            fold_logits.append(np.vstack(outs))
        # convert each fold to physical units, then mean/std across folds per parameter
        fold_phys = [to_original_units(P) for P in fold_logits]   # list of dicts of (N,) arrays
        meta = df.loc[kept, [c for c in (name_col, smiles_col, ion_col) if c in df.columns]].reset_index(drop=True)
        out = meta.copy()
        for param in TASKS_WITH_UNITS:
            stack = np.stack([fp[param] for fp in fold_phys])     # (n_folds, N)
            out[f"{param}_mean"] = stack.mean(axis=0)
            out[f"{param}_std"] = stack.std(axis=0, ddof=0)
        return out

    @staticmethod
    def _coerce_to_frame(molecules, smiles_col, ion_col, name_col) -> pd.DataFrame:
        if isinstance(molecules, (str, Path)):
            df = pd.read_csv(molecules)
        elif isinstance(molecules, pd.DataFrame):
            df = molecules.copy()
        else:
            df = pd.DataFrame(list(molecules))
        if smiles_col not in df.columns:
            raise ValueError(f"input must contain a '{smiles_col}' column; got {list(df.columns)}")
        if ion_col not in df.columns:
            df[ion_col] = "neutral"
        df[ion_col] = df[ion_col].fillna("neutral")
        if name_col not in df.columns:
            df[name_col] = [f"mol_{i+1}" for i in range(len(df))]
        return df
