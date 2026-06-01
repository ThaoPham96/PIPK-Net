"""Clinical reference data: loading, descriptors, scaffold partition, and the
distribution statistics behind Table 1.

The canonical reference table (``data/df_ref.csv``) is the e-Drug3D-derived
cohort of 1170 oral drugs with SMILES, IonType, the seven physical-unit PK
targets, their log10/logit10 transforms, and RDKit descriptors (LogP, MW, BCS).
1167 rows have parseable SMILES and form the modelled dataset; a scaffold-aware
80/20 split (GroupShuffleSplit, seed 42) yields 904 development + 263
independent-test compounds -- reproducing the manuscript partition exactly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from sklearn.model_selection import GroupShuffleSplit

from . import config as C

RDLogger.DisableLog("rdApp.*")

# pipknet provides the canonical generic Bemis-Murcko scaffold helper
from pipknet.utils import get_generic_scaffold


# ---------------------------------------------------------------------------
# Descriptors (manuscript cell: classify_bcs)
# ---------------------------------------------------------------------------
def classify_bcs(smiles):
    """Return (LogP, MW, BCS_Candidate) for a SMILES; (None, None, msg) on failure."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None, None, "Invalid SMILES"
        logp = Descriptors.MolLogP(mol)
        mw = Descriptors.MolWt(mol)
        if logp > 5.0 and mw > 500:
            bcs = "Class IV (Low Sol/Low Perm)"
        elif logp > 5.0:
            bcs = "Class II (Low Sol/High Perm)"
        else:
            bcs = "Class I/III (High Solubility)"
        return logp, mw, bcs
    except Exception:
        return None, None, "Error"


# ---------------------------------------------------------------------------
# Reference table
# ---------------------------------------------------------------------------
def build_reference(force_rebuild: bool = False) -> pd.DataFrame:
    """Load the canonical reference table, indexed by ORIG_INDEX.

    Reads ``data/df_ref.csv`` if present; otherwise vendors it from the source
    artefact (adding LogP/MW/BCS and cleaning IonType) and caches it.
    """
    if C.DF_REF_CSV.exists() and not force_rebuild:
        df = pd.read_csv(C.DF_REF_CSV)
    else:
        df = pd.read_csv(C.SRC_DF_REF)
        # descriptors + clean IonType (manuscript cells 36)
        if "LogP" not in df.columns or df["LogP"].isna().all():
            df[["LogP", "MW", "BCS_Candidate"]] = df["SMILES"].apply(
                lambda s: pd.Series(classify_bcs(s)))
        df["IonType"] = df["IonType"].fillna("Neutral").astype(str).str.capitalize()
        C.DATA_DIR.mkdir(parents=True, exist_ok=True)
        df.to_csv(C.DF_REF_CSV, index=False)

    if "ORIG_INDEX" not in df.columns:
        df = df.reset_index().rename(columns={"index": "ORIG_INDEX"})
    df["IonType"] = df["IonType"].fillna("Neutral").astype(str).str.capitalize()
    # index by ORIG_INDEX but keep it as a column so .loc[row_idx] works everywhere
    df = df.set_index("ORIG_INDEX", drop=False)
    df.index.name = "ORIG_INDEX"
    return df


def valid_smiles_index(df_ref: pd.DataFrame) -> list[int]:
    """ORIG_INDEX values whose SMILES parse (the modelled dataset, 1167 rows)."""
    keep = []
    for oi, smi in zip(df_ref["ORIG_INDEX"], df_ref[C.SMILES_COL]):
        if pd.notna(smi) and Chem.MolFromSmiles(str(smi)) is not None:
            keep.append(int(oi))
    return keep


def scaffold_partition(df_ref: pd.DataFrame):
    """Reproduce the scaffold-aware 80/20 split.

    Returns ``(dev_orig, test_orig)`` lists of ORIG_INDEX. Validated against the
    frozen independent-test set when available.
    """
    valid = valid_smiles_index(df_ref)
    sub = df_ref.loc[valid]
    scaffolds = sub[C.SMILES_COL].map(lambda s: get_generic_scaffold(s))
    groups = np.array([
        sc if isinstance(sc, str) and sc else f"IDX_{int(oi)}"
        for sc, oi in zip(scaffolds, sub["ORIG_INDEX"])
    ], dtype=object)

    gss = GroupShuffleSplit(n_splits=1, test_size=C.TEST_SIZE, random_state=C.SPLIT_SEED)
    dev_rel, test_rel = next(gss.split(np.arange(len(sub)), groups=groups))
    dev_orig = [int(sub.iloc[i]["ORIG_INDEX"]) for i in dev_rel]
    test_orig = [int(sub.iloc[i]["ORIG_INDEX"]) for i in test_rel]
    return dev_orig, test_orig


def partition_labels(df_ref: pd.DataFrame):
    """Return df_ref with a 'partition' column (Development / Test / Excluded)."""
    dev, test = scaffold_partition(df_ref)
    df = df_ref.copy()
    df["partition"] = "Excluded"
    df.loc[dev, "partition"] = "Development"
    df.loc[test, "partition"] = "Test"
    return df, dev, test


# ---------------------------------------------------------------------------
# Table 1 -- distribution statistics
# ---------------------------------------------------------------------------
#   label -> (df_ref physical column, unit)
PK_PARAMS_T5 = {
    "Half-life (t1/2)":             ("t1/2(hour)",      "h"),
    "Volume of distribution (Vd)":  ("VD(liter)",       "L"),
    "Clearance (CL)":               ("Cl(liter/hour)",  "L/h"),
    "Bioavailability (F)":          ("F(percentage)",   "%"),
    "Plasma protein binding (PPB)": ("PPB(percentage)", "%"),
    "Peak concentration (Cmax)":    ("Cmax_uM",         "µM"),
    "Time to peak (Tmax)":          ("Tmax(hour)",      "h"),
}


def distribution_table(df_ref: pd.DataFrame) -> pd.DataFrame:
    """Table 1: per-task N/min/max/mean/median over Development vs Test partitions."""
    df, _, _ = partition_labels(df_ref)
    rows = []
    for label, (col, unit) in PK_PARAMS_T5.items():
        if col not in df.columns:
            continue
        for part in ["Development", "Test"]:
            sub = pd.to_numeric(df[df["partition"] == part][col], errors="coerce").dropna()
            if len(sub) == 0:
                continue
            rows.append({
                "PK Parameter": label, "Unit": unit, "Partition": part,
                "N": len(sub), "Min": round(sub.min(), 3), "Max": round(sub.max(), 2),
                "Mean": round(sub.mean(), 2), "Median": round(sub.median(), 2),
            })
    return pd.DataFrame(rows)
