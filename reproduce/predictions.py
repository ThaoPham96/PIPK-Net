"""Independent-test predictions for every benchmarked model.

``freeze()`` runs (once) the per-fold and ensemble inference for the GNN
ablation variants (A_baseline/B_ion/C_physio) and the ChemBERTa benchmark from
their saved checkpoints, loads the Chemprop and ChemLLM predictions from disk,
and writes everything to ``data/predictions/`` as tidy CSVs in physical units.

After freezing, the lightweight ``load_*`` helpers let the figure/table modules
run with **no torch / transformers / chemprop dependency** -- they read only the
frozen CSVs. Heavy imports (torch, PyG) are deferred into the inference
functions so importing this module stays cheap.

Schema
------
data/predictions/
  truth_indeptest.csv                 row_idx, SMILES, Name, IonType, LogP, MW, <physical truth cols>
  perfold/gnn_<variant>.csv           row_idx, fold, <7 tasks>           (physical units)
  perfold/chemberta_A_baseline.csv    row_idx, fold, <7 tasks>           (physical units)
  perfold/chemprop.csv                smiles, fold, <7 tasks>            (physical units)
  ensemble/pipknet.csv                row_idx, <7 tasks>                 (= GNN C_physio fold-mean)
  ensemble/gnn_<variant>.csv          row_idx, <7 tasks>
  ensemble/chemberta.csv              row_idx, <7 tasks>                 (log-mean then back-transform)
  ensemble/chemprop.csv               smiles, <7 tasks>
  ensemble/chemllm.csv                row_idx, <7 tasks>                 (name-aligned, single set)
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

from . import config as C
from . import data as D

PERFOLD_DIR = C.PRED_DIR / "perfold"
ENS_DIR = C.PRED_DIR / "ensemble"


# ---------------------------------------------------------------------------
# Inverse transforms (log10 / logit10 -> physical units)
# ---------------------------------------------------------------------------
def _inv_log(v):
    return 10.0 ** v


def _inv_logit_pct(v):
    return 100.0 / (1.0 + 10.0 ** (-v))


def log_logit_to_physical(arr2d) -> pd.DataFrame:
    """(N, 7) array in log10/logit10 space -> DataFrame with task columns (physical)."""
    a = np.atleast_2d(np.asarray(arr2d, dtype=float))
    out = {}
    for t in C.TASKS:
        col = a[:, C.TIDX[t]]
        out[t] = _inv_logit_pct(col) if t in C.LOGIT_TASKS else _inv_log(col)
    return pd.DataFrame(out)


def backtransform_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Back-transform a DataFrame whose task columns hold log10/logit10 values."""
    out = df.copy()
    for t in C.TASKS:
        if t in out.columns:
            out[t] = _inv_logit_pct(out[t]) if t in C.LOGIT_TASKS else _inv_log(out[t])
    return out


# ===========================================================================
# Freeze step (requires torch + PyG; run once)
# ===========================================================================
def freeze(validate: bool = True):
    """Generate every prediction CSV under ``data/predictions/``."""
    import torch
    from torch_geometric.loader import DataLoader
    from pipknet.models import GNNMultitask
    from pipknet.featurizers import smiles_to_pyg
    from .models_chemberta import (ChemBERTa_PK_MLP, load_chemberta_feature_matrix,
                                   attach_chemberta_features)

    PERFOLD_DIR.mkdir(parents=True, exist_ok=True)
    ENS_DIR.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    df_ref = D.build_reference()
    dev_orig, test_orig = D.scaffold_partition(df_ref)
    print(f"[freeze] dev={len(dev_orig)} test={len(test_orig)} (device={device})")

    # --- truth table for the independent test set -------------------------
    sub = df_ref.loc[test_orig]
    truth = pd.DataFrame({"row_idx": sub["ORIG_INDEX"].astype(int).values})
    truth["SMILES"] = sub[C.SMILES_COL].values
    truth["Name"] = sub["Name"].values
    truth["IonType"] = sub["IonType"].values
    truth["LogP"] = pd.to_numeric(sub["LogP"], errors="coerce").values
    truth["MW"] = pd.to_numeric(sub["MW"], errors="coerce").values
    for t in C.TASKS:
        raw = C.RAW_TRUTH_COLS[t]
        std = C.TRUTH_COLS[t]
        truth[std] = pd.to_numeric(sub[raw], errors="coerce").values if raw in sub.columns else np.nan
    truth.to_csv(C.TRUTH_CSV, index=False)
    print(f"[freeze] wrote truth ({len(truth)} rows)")

    # --- build PyG graphs for the independent test set --------------------
    graphs, ok_orig = [], []
    for oi in test_orig:
        row = df_ref.loc[oi]
        g = smiles_to_pyg(row[C.SMILES_COL], row["IonType"])
        if g is None:
            continue
        g.row_idx = torch.tensor([int(oi)], dtype=torch.long)
        graphs.append(g)
        ok_orig.append(int(oi))

    # ===================== GNN variants ==================================
    @torch.no_grad()
    def gnn_fold_predict(variant, fold, loader):
        d = C.CHECKPOINTS_DIR / variant / f"fold_{fold}"
        hp = json.load(open(d / "config.json"))["best_hp"]
        model = GNNMultitask(
            hidden_dim=hp["hidden_dim"], ion_dim=4,
            ion_emb_dim=hp.get("ion_emb_dim", 16), out_dim=7,
            use_ion=(variant != "A_baseline"), dropout=hp["dropout"],
        ).to(device)
        ck = torch.load(d / "best_outer_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=False)
        model.eval()
        preds, idx = [], []
        for batch in loader:
            batch = batch.to(device)
            yh = model(batch).cpu().numpy()   # raw log10/logit10 outputs (no clipping)
            preds.append(yh)
            idx.append(batch.row_idx.view(-1).cpu().numpy())
        P = np.vstack(preds)
        rid = np.concatenate(idx)
        phys = log_logit_to_physical(P)
        phys.insert(0, "row_idx", rid)
        return phys

    loader = DataLoader(graphs, batch_size=64, shuffle=False)
    for variant in C.GNN_VARIANTS:
        folds = []
        for f in range(1, C.N_SPLITS + 1):
            pf = gnn_fold_predict(variant, f, loader)
            pf.insert(1, "fold", f)
            folds.append(pf)
        perfold = pd.concat(folds, ignore_index=True)
        perfold.to_csv(PERFOLD_DIR / f"gnn_{variant}.csv", index=False)
        # ensemble = mean across folds of physical predictions
        ens = perfold.groupby("row_idx")[C.TASKS].mean().reset_index()
        ens.to_csv(ENS_DIR / f"gnn_{variant}.csv", index=False)
        if variant == "C_physio":
            ens.to_csv(ENS_DIR / "pipknet.csv", index=False)
        print(f"[freeze] GNN {variant}: per-fold {perfold.shape}, ensemble {ens.shape}")

    # ===================== Case-study drugs (Figure 7 / Table 5) ==========
    # Predict with the C_physio (PIPK-Net) ensemble for the four case-study drugs
    # regardless of which split they fall in, so F8/T4 are reproducible.
    import re

    def _norm(s):
        return re.sub(r"[^A-Z0-9]", "", str(s).upper())

    df_ref["_nm"] = df_ref["Name"].map(_norm)
    case_rows = []
    for drug in C.CASE_STUDY_DRUGS:
        hit = df_ref[df_ref["_nm"] == _norm(drug)]
        if hit.empty:
            print(f"[freeze] WARNING case drug not found: {drug}")
            continue
        row = hit.iloc[0]
        case_rows.append((drug, int(row["ORIG_INDEX"]), row[C.SMILES_COL], row["IonType"], row))
    if case_rows:
        cgraphs = []
        for drug, oi, smi, ion, _row in case_rows:
            g = smiles_to_pyg(smi, ion)
            g.row_idx = torch.tensor([oi], dtype=torch.long)
            cgraphs.append(g)
        cloader = DataLoader(cgraphs, batch_size=len(cgraphs), shuffle=False)
        fold_phys = [gnn_fold_predict("C_physio", f, cloader).set_index("row_idx")
                     for f in range(1, C.N_SPLITS + 1)]
        case_pred = pd.concat(fold_phys).groupby("row_idx")[C.TASKS].mean()
        out_rows = []
        for drug, oi, smi, ion, row in case_rows:
            rec = {"ID": C.CASE_STUDY_DRUGS.index(drug) + 1, "Name": drug, "SMILES": smi,
                   "IonType": ion, "LogP": pd.to_numeric(row.get("LogP"), errors="coerce")}
            for t in C.TASKS:
                rec[f"pred_{t}"] = case_pred.loc[oi, t]
                raw = C.RAW_TRUTH_COLS[t]
                rec[f"obs_{t}"] = pd.to_numeric(row.get(raw), errors="coerce") if raw in row else np.nan
            out_rows.append(rec)
        pd.DataFrame(out_rows).to_csv(C.PRED_DIR / "casestudy_pipknet.csv", index=False)
        print(f"[freeze] case-study predictions: {len(out_rows)} drugs")

    # ===================== ChemBERTa (A_baseline benchmark) ===============
    feat_mat = load_chemberta_feature_matrix(C.SRC_CHEMBERTA_FEATS, df_ref)
    cb_graphs = []
    for oi in ok_orig:
        row = df_ref.loc[oi]
        g = smiles_to_pyg(row[C.SMILES_COL], row["IonType"])
        g.row_idx = torch.tensor([int(oi)], dtype=torch.long)
        cb_graphs.append(g)
    attach_chemberta_features(cb_graphs, feat_mat, df_ref)

    @torch.no_grad()
    def chemberta_fold_predict(variant, fold, glist):
        d = C.SRC_CHEMBERTA_ROOT / variant / f"fold_{fold}"
        hp = json.load(open(d / "config.json"))["best_hp"]
        model = ChemBERTa_PK_MLP(
            input_dim=768, hidden_dim=hp["hidden_dim"], ion_dim=len(["a", "c", "n", "z"]),
            ion_emb_dim=hp.get("ion_emb_dim", 0), out_dim=7,
            use_ion=(variant != "A_baseline"), dropout=hp["dropout"],
        ).to(device)
        ck = torch.load(d / "best_outer_model.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"], strict=False)
        model.eval()
        ld = DataLoader(glist, batch_size=hp.get("batch_size", 128), shuffle=False)
        out, idx = [], []
        for batch in ld:
            batch = batch.to(device)
            out.append(model(batch).cpu().numpy())   # log/logit space
            idx.append(batch.row_idx.view(-1).cpu().numpy())
        raw = pd.DataFrame(np.vstack(out), columns=C.TASKS)
        raw.insert(0, "row_idx", np.concatenate(idx))
        return raw

    cb_variant = "A_baseline"   # the ChemBERTa benchmark reported in the paper
    raw_folds = [chemberta_fold_predict(cb_variant, f, cb_graphs) for f in range(1, C.N_SPLITS + 1)]
    # per-fold physical (back-transform each fold) -> for the F3 boxplot
    cb_perfold = []
    for f, raw in enumerate(raw_folds, 1):
        phys = backtransform_columns(raw.set_index("row_idx")).reset_index()
        phys.insert(1, "fold", f)
        cb_perfold.append(phys)
    cb_perfold = pd.concat(cb_perfold, ignore_index=True)
    cb_perfold.to_csv(PERFOLD_DIR / "chemberta_A_baseline.csv", index=False)
    # ensemble = mean of RAW log/logit across folds, then back-transform
    raw_concat = pd.concat([r.set_index("row_idx") for r in raw_folds])
    raw_mean = raw_concat.groupby("row_idx")[C.TASKS].mean()
    cb_ens = backtransform_columns(raw_mean).reset_index()
    cb_ens.to_csv(ENS_DIR / "chemberta.csv", index=False)
    print(f"[freeze] ChemBERTa: per-fold {cb_perfold.shape}, ensemble {cb_ens.shape}")

    # ===================== Chemprop (from saved fold CSVs) ================
    cp_map = {"t_half": "log_t1/2(hour)", "Vd": "log_VD(liter)", "CL": "log_Cl(liter/hour)",
              "Cmax": "log_Cmax_uM", "Tmax": "log_Tmax",
              "F": "logit_F(percentage)", "PPB": "logit_PPB(percentage)"}
    cp_perfold, cp_raw_folds = [], []
    for f in range(1, C.N_SPLITS + 1):
        path = C.SRC_CHEMPROP_PREDS / f"fold{f}_indep_preds.csv"
        df = pd.read_csv(path)
        df.columns = [c.lower() if c.lower() == "smiles" else c for c in df.columns]
        raw = pd.DataFrame({"smiles": df["smiles"]})
        for t, col in cp_map.items():
            raw[t] = pd.to_numeric(df[col], errors="coerce")
        cp_raw_folds.append(raw)
        phys = backtransform_columns(raw.copy())
        phys.insert(1, "fold", f)
        cp_perfold.append(phys)
    cp_perfold = pd.concat(cp_perfold, ignore_index=True)
    cp_perfold.to_csv(PERFOLD_DIR / "chemprop.csv", index=False)
    # ensemble = mean raw log/logit per SMILES across folds, then back-transform
    cp_long = pd.concat(cp_raw_folds, ignore_index=True)
    cp_mean = cp_long.groupby("smiles")[C.TASKS].mean()
    cp_ens = backtransform_columns(cp_mean).reset_index()
    cp_ens.to_csv(ENS_DIR / "chemprop.csv", index=False)
    # Transparency: the test set has duplicate canonical SMILES (the same drug under
    # multiple e-Drug3D records). Chemprop is SMILES-keyed, so its ensemble collapses
    # them to unique SMILES; GNN/ChemBERTa are index-keyed and keep all records.
    # Guard against a *genuine* future prediction failure (a test SMILES with no pred).
    test_smiles = set(sub[C.SMILES_COL])
    true_misses = test_smiles - set(cp_ens["smiles"])
    n_dup_records = len(sub) - sub[C.SMILES_COL].nunique()  # duplicate-SMILES records
    print(f"[freeze] Chemprop: per-fold {cp_perfold.shape}, ensemble {cp_ens.shape} "
          f"({len(test_orig)} test records -> {cp_ens.shape[0]} unique SMILES; "
          f"{n_dup_records} duplicate-SMILES records collapsed; "
          f"{len(true_misses)} SMILES with no prediction)")
    if true_misses:
        print(f"[freeze] WARNING Chemprop is missing predictions for {len(true_misses)} "
              f"test SMILES: {sorted(true_misses)[:5]}{' ...' if len(true_misses) > 5 else ''}")

    # ===================== ChemLLM (reference only) ======================
    cl = pd.read_csv(C.SRC_CHEMLLM)
    cl["_name"] = cl["Name"].astype(str).str.strip().str.upper()
    cl = cl[~cl["_name"].duplicated(keep="first")].set_index("_name")
    cl_cols = {"t_half": "t_half", "Vd": "Vd", "CL": "CL", "F": "F",
               "PPB": "PPB", "Cmax": "Cmax", "Tmax": "Tmax"}
    tr_name = sub["Name"].astype(str).str.strip().str.upper()
    cl_ens = pd.DataFrame({"row_idx": sub["ORIG_INDEX"].astype(int).values})
    for t, col in cl_cols.items():
        if col in cl.columns:
            cl_ens[t] = tr_name.map(cl[col]).apply(pd.to_numeric, errors="coerce").values
    cl_ens.to_csv(ENS_DIR / "chemllm.csv", index=False)
    print(f"[freeze] ChemLLM: ensemble {cl_ens.shape}")

    # ===================== Nested-CV predictions (Figure 2) ==============
    # Vendor the CV prediction table into data/ so Fig 2 reads only from data/
    # (consistent with the other figures and robust to the checkpoints copy).
    import shutil
    cv_src = None
    for cand in (C.SRC_GNN_ROOT / "all_drug_predictions.csv", C.NESTED_CV_FALLBACK):
        if cand.exists():
            cv_src = cand
            break
    if cv_src is not None:
        shutil.copyfile(cv_src, C.NESTED_CV_PRED_CSV)
        print(f"[freeze] nested-CV predictions vendored from {cv_src.name}")
    else:
        print("[freeze] WARNING nested-CV predictions not found (Figure 2 will be unavailable)")

    if validate:
        _validate_freeze(df_ref)
    print("[freeze] done.")


def _validate_freeze(df_ref):
    """Optional sanity check: compare the C_physio ensemble against a reference file.

    Enabled by setting the environment variable ``PIPKNET_VALIDATION_REF`` to a
    ``gnn_physio_vs_truth.csv`` (row_idx-indexed) with reference predictions;
    skipped cleanly otherwise (so it never fails on a fresh clone).
    """
    ref_env = os.environ.get("PIPKNET_VALIDATION_REF")
    ref_path = Path(ref_env) if ref_env else (C.SRC / "gnn_physio_vs_truth.csv")
    if not ref_path.exists():
        print("[validate] no validation reference (set PIPKNET_VALIDATION_REF to enable); skipping")
        return
    ref = pd.read_csv(ref_path).set_index("row_idx")
    ens = pd.read_csv(ENS_DIR / "pipknet.csv").set_index("row_idx")
    common = ens.index.intersection(ref.index)
    if len(common) == 0:
        print("[validate] no overlapping row_idx; skipping")
        return
    diff = (np.log10(ens.loc[common, "Vd"].clip(lower=1e-9))
            - np.log10(ref.loc[common, "Vd"].clip(lower=1e-9))).abs().median()
    print(f"[validate] PIPK-Net Vd median |Δlog10| vs frozen ref = {diff:.4f} (n={len(common)})")


# ===========================================================================
# Lightweight loaders (no torch) -- used by figures/tables
# ===========================================================================
def load_truth() -> pd.DataFrame:
    """Independent-test ground truth, indexed by row_idx."""
    return pd.read_csv(C.TRUTH_CSV).set_index("row_idx")


def load_perfold(model: str, variant: str | None = None):
    """Return a list of 5 per-fold prediction DataFrames (physical units).

    ``model`` in {'gnn','chemberta','chemprop'}. For GNN pass ``variant``.
    GNN/ChemBERTa frames are indexed by row_idx; Chemprop by smiles.
    """
    if model == "gnn":
        df = pd.read_csv(PERFOLD_DIR / f"gnn_{variant}.csv")
        key = "row_idx"
    elif model == "chemberta":
        df = pd.read_csv(PERFOLD_DIR / "chemberta_A_baseline.csv")
        key = "row_idx"
    elif model == "chemprop":
        df = pd.read_csv(PERFOLD_DIR / "chemprop.csv")
        key = "smiles"
    else:
        raise ValueError(model)
    return [g.drop(columns="fold").set_index(key)[C.TASKS]
            for _, g in df.groupby("fold")]


def load_ensemble(model: str) -> pd.DataFrame:
    """Ensemble-mean predictions in physical units.

    ``model`` in {'pipknet','gnn_A_baseline','gnn_B_ion','chemberta','chemprop','chemllm'}.
    Indexed by row_idx (or smiles for chemprop).
    """
    fname = {"pipknet": "pipknet.csv", "chemberta": "chemberta.csv",
             "chemprop": "chemprop.csv", "chemllm": "chemllm.csv",
             "gnn_A_baseline": "gnn_A_baseline.csv", "gnn_B_ion": "gnn_B_ion.csv",
             "gnn_C_physio": "gnn_C_physio.csv"}[model]
    df = pd.read_csv(ENS_DIR / fname)
    key = "smiles" if model == "chemprop" else "row_idx"
    return df.set_index(key)


def load_nested_cv() -> pd.DataFrame:
    """Per-fold cross-validation predictions (Figure 2).

    Reads the vendored copy under ``data/predictions/``; falls back to the
    checkpoints mirror if the vendored file is absent.
    """
    path = C.NESTED_CV_PRED_CSV if C.NESTED_CV_PRED_CSV.exists() else C.NESTED_CV_FALLBACK
    if not path.exists():
        raise FileNotFoundError(
            "Nested-CV predictions not found. Run `python -m reproduce.predictions` to "
            f"vendor them to {C.NESTED_CV_PRED_CSV}, or restore "
            f"{C.NESTED_CV_FALLBACK} from your GNN run directory.")
    return pd.read_csv(path)


if __name__ == "__main__":
    freeze()
