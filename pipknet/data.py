"""CSV loading and PyG dataset construction for training/evaluation.

Takes a CSV of clinical PK records (one row per drug, with a SMILES column,
an optional IonType column, and the raw physical-unit PK columns), applies the
log10/logit10 target transforms, validates the required columns, and builds a
list of PyTorch Geometric ``Data`` objects carrying the targets, a visibility
mask, and a stable ``row_idx`` for joining back to the source table.
"""
from __future__ import annotations

from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch_geometric.data import Data

from .featurizers import smiles_to_pyg
from .utils import extract_targets, get_generic_scaffold

# Raw physical-unit PK columns and the outlier ceilings used to null out
# non-physiological label values before transformation.
RAW_PK_COLUMNS = ["t1/2(hour)", "VD(liter)", "Cl(liter/hour)",
                  "F(percentage)", "PPB(percentage)", "Cmax_uM", "Tmax"]
OUTLIER_THRESHOLDS = {"t1/2(hour)": 10000, "VD(liter)": 20000,
                      "Cl(liter/hour)": 1000, "Cmax_uM": 2000, "Tmax": 100}


def apply_pk_transforms(df: pd.DataFrame) -> pd.DataFrame:
    """Add log10 (and logit10 for F/PPB) target columns, nulling extreme outliers.

    Reproduces the manuscript preprocessing: physiologically implausible label
    values are set to NaN (kept as rows, excluded per-task), log10 is applied to
    the dispositional endpoints, and logit10 to the bounded percentage endpoints.
    """
    out = df.copy()
    if "Tmax" not in out.columns and "Tmax(hour)" in out.columns:
        out["Tmax"] = out["Tmax(hour)"]

    for col, limit in OUTLIER_THRESHOLDS.items():
        if col in out.columns:
            out[col] = pd.to_numeric(out[col], errors="coerce")
            out.loc[out[col] > limit, col] = np.nan

    for task in ["t1/2(hour)", "VD(liter)", "Cl(liter/hour)", "Cmax_uM", "Tmax"]:
        if task in out.columns:
            v = pd.to_numeric(out[task], errors="coerce")
            out[f"log_{task if task != 'Tmax' else 'Tmax'}"] = np.log10(v + 1e-3)
    # normalise the Cmax/Tmax transformed column names to those used downstream
    if "log_Tmax" not in out.columns and "Tmax" in out.columns:
        out["log_Tmax"] = np.log10(pd.to_numeric(out["Tmax"], errors="coerce") + 1e-3)

    for task in ["F(percentage)", "PPB(percentage)"]:
        if task in out.columns:
            p = pd.to_numeric(out[task], errors="coerce").clip(0.1, 99.9)
            out[f"logit_{task}"] = np.log10(p / (100 - p))
    return out


def build_dataset(df_ref: pd.DataFrame, smiles_col: str = "SMILES",
                  ion_col: str = "IonType", id_col: str = "ORIG_INDEX",
                  name_col: str = "Name") -> List[Data]:
    """Build a list of PyG graphs with targets, mask, row_idx and drug name."""
    data_list: List[Data] = []
    for _, row in df_ref.iterrows():
        smi = row.get(smiles_col)
        if pd.isna(smi):
            continue
        g = smiles_to_pyg(smi, row.get(ion_col, "neutral"))
        if g is None:
            continue
        y, mask = extract_targets(row)
        g.y = y.view(1, -1)
        g.y_mask = mask.view(1, -1)
        g.row_idx = torch.tensor([int(row[id_col])], dtype=torch.long)
        g.drug_name = str(row.get(name_col, ""))
        data_list.append(g)
    return data_list


def scaffold_groups(df_ref: pd.DataFrame, smiles_col: str = "SMILES",
                    id_col: str = "ORIG_INDEX") -> np.ndarray:
    """Generic Bemis-Murcko scaffold group key per row (falls back to a unique id)."""
    keys = []
    for _, row in df_ref.iterrows():
        if pd.isna(row.get(smiles_col)):
            continue
        if smiles_to_pyg(row.get(smiles_col), row.get("IonType", "neutral")) is None:
            continue
        sc = get_generic_scaffold(row[smiles_col])
        keys.append(sc if isinstance(sc, str) and sc else f"IDX_{int(row[id_col])}")
    return np.array(keys, dtype=object)


def load_dataset_from_csv(path, smiles_col: str = "SMILES", ion_col: str = "IonType"
                          ) -> Tuple[List[Data], pd.DataFrame, np.ndarray]:
    """Load a CSV, apply transforms, and return ``(dataset, df_ref, groups)``.

    ``df_ref`` is indexed by a stable ``ORIG_INDEX`` (row position) and ``groups``
    is the scaffold group key aligned to ``dataset`` order.
    """
    df = pd.read_csv(path)
    if smiles_col not in df.columns:
        raise ValueError(f"CSV must contain a '{smiles_col}' column; found {list(df.columns)[:10]}…")
    if ion_col not in df.columns:
        df[ion_col] = "neutral"
    df[ion_col] = df[ion_col].fillna("neutral")

    df = apply_pk_transforms(df)
    if "ORIG_INDEX" not in df.columns:
        df = df.reset_index(drop=True)
        df["ORIG_INDEX"] = np.arange(len(df))
    df = df.set_index("ORIG_INDEX", drop=False)
    df.index.name = "ORIG_INDEX"

    dataset = build_dataset(df, smiles_col=smiles_col, ion_col=ion_col)
    groups = scaffold_groups(df, smiles_col=smiles_col)
    return dataset, df, groups
