"""Central configuration for the PIPK-Net reproducibility package.

All paths, task definitions, plotting styles and constants live here so that no
analysis/figure module hard-codes a machine-specific path. Override the source
data root (used only by the one-time ``predictions`` freeze step) with the
environment variable ``PIPKNET_SRC`` if your raw artefacts live elsewhere.
"""
from __future__ import annotations

import os
from pathlib import Path

# ---------------------------------------------------------------------------
# Repository layout (self-contained; resolved relative to this file)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent          # .../PIPK_Net_Release
CHECKPOINTS_DIR = REPO_ROOT / "checkpoints"                 # GNN A/B/C fold checkpoints
DATA_DIR = REPO_ROOT / "data"
PRED_DIR = DATA_DIR / "predictions"                         # frozen prediction CSVs
SPLITS_DIR = DATA_DIR / "splits"
BENCH_DIR = DATA_DIR / "benchmarks"                         # static reference tables (T4)
OUTPUTS_DIR = REPO_ROOT / "outputs"
FIG_DIR = OUTPUTS_DIR / "figures"
TAB_DIR = OUTPUTS_DIR / "tables"

DF_REF_CSV = DATA_DIR / "df_ref.csv"                        # canonical clinical reference
TRUTH_CSV = PRED_DIR / "truth_indeptest.csv"               # independent-test ground truth
NESTED_CV_PRED_CSV = PRED_DIR / "nested_cv_predictions.csv"  # vendored CV preds (used by Fig 2)
NESTED_CV_FALLBACK = CHECKPOINTS_DIR / "all_drug_predictions.csv"  # release-mirror fallback

for _d in (DATA_DIR, PRED_DIR, BENCH_DIR, OUTPUTS_DIR, FIG_DIR, TAB_DIR):
    _d.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Source artefacts -- used ONLY by predictions.freeze() (the one-time export).
# Figures/tables never touch these; they read DATA_DIR. Override with $PIPKNET_SRC.
# ---------------------------------------------------------------------------
SRC = Path(os.environ.get("PIPKNET_SRC", r"D:/PK_prediction_vscode/PK_prediction_vscode"))
SRC_DF_REF = SRC / "df_ref_final_with_bcs.csv"             # clinical reference (1170 rows)
SRC_CHEMBERTA_FEATS = SRC / "oral_drug_data_with_smiles_chemberta.csv"  # 768-dim feats, CAS-indexed
SRC_CHEMBERTA_ROOT = SRC / "runs" / "chemberta_updated_sets_v6"          # ChemBERTa checkpoints
SRC_GNN_ROOT = SRC / "runs" / "gnn_updated_sets_v6"                       # GNN run dir (CV preds source)
SRC_CHEMPROP_PREDS = Path(os.environ.get("PIPKNET_CHEMPROP", r"D:/chemprop_matched_models"))
SRC_CHEMLLM = Path(os.environ.get("PIPKNET_CHEMLLM", r"D:/chemllm_benchmark_independent_predictions.csv"))

# ---------------------------------------------------------------------------
# Tasks
# ---------------------------------------------------------------------------
TASKS = ["t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax"]
TIDX = {t: i for i, t in enumerate(TASKS)}

# tasks modelled in log10 space (vs logit10 for the percentage endpoints)
LOG_TASKS = {"t_half", "Vd", "CL", "Cmax", "Tmax"}
LOGIT_TASKS = {"F", "PPB"}

# transformed-target column names in df_ref (used for F1 KDEs / training)
TRANSFORMED_COLS = [
    "log_t1/2(hour)", "log_VD(liter)", "log_Cl(liter/hour)",
    "logit_F(percentage)", "logit_PPB(percentage)", "log_Cmax_uM", "log_Tmax",
]

# Two-layer column-name mapping (NOT a duplication):
#   RAW_TRUTH_COLS  task -> column name in the SOURCE df_ref. Used ONLY by
#                   predictions.freeze() when reading clinical labels out of df_ref.
#   TRUTH_COLS      task -> column name in the FROZEN truth table
#                   (data/predictions/truth_indeptest.csv). Used by every figure/table.
# freeze() reads df_ref[RAW_TRUTH_COLS[t]] and writes it under TRUTH_COLS[t]; thereafter
# all downstream code uses the frozen table via TRUTH_COLS only.
TRUTH_COLS = {
    "t_half": "t_half", "Vd": "Vd", "CL": "CL",
    "F": "F(percentage)", "PPB": "PPB(percentage)",
    "Cmax": "Cmax_uM", "Tmax": "Tmax",
}
RAW_TRUTH_COLS = {
    "t_half": "t1/2(hour)", "Vd": "VD(liter)", "CL": "Cl(liter/hour)",
    "F": "F(percentage)", "PPB": "PPB(percentage)",
    "Cmax": "Cmax_uM", "Tmax": "Tmax(hour)",
}

TASK_DISPLAY = {
    "t_half": r"$t_{1/2}$ (h)", "Vd": r"$V_d$ (L)", "CL": "CL (L/h)",
    "F": "F (%)", "PPB": "PPB (%)", "Cmax": r"$C_{max}$ (µM)", "Tmax": r"$T_{max}$ (h)",
}
TASK_UNITS = {"t_half": "h", "Vd": "L", "CL": "L/h", "F": "%", "PPB": "%", "Cmax": "µM", "Tmax": "h"}

# ---------------------------------------------------------------------------
# Models & styling
# ---------------------------------------------------------------------------
GNN_VARIANTS = ["A_baseline", "B_ion", "C_physio"]          # C_physio == PIPK-Net
SMILES_COL = "SMILES"

# ablation palette (Figures 2 & 4)
ABLATION_PALETTE = {"A_baseline": "#a9a9a9", "B_ion": "#6baed6", "C_physio": "#31a354"}
ABLATION_LABELS = [r"$A_{baseline}$", r"$B_{ion}$", r"$C_{physio}$ (PIPK-Net)"]

# benchmark palette (Figure 4)
BENCH_PALETTE = {"ChemBERTa": "#d6a36b", "Chemprop": "#9e7ac4", "PIPK-Net": "#31a354"}

# ionisation colours (Figures 6 & 8) -- keyed by capitalised IonType
ION_COLORS = {"Neutral": "#1976D2", "Cationic": "#F57C00", "Anionic": "#388E3C", "Zwitterionic": "#C2185B"}

DEV_COLOR, TEST_COLOR = "#1f77b4", "#d62728"

# ---------------------------------------------------------------------------
# Analysis constants
# ---------------------------------------------------------------------------
# Vd cohorts (Table 4 / Figure 6)
VD_COHORTS = {
    "Low Vd (<50 L)":              lambda v: v < 50,
    "Medium-High Vd (300-2000 L)": lambda v: (v >= 300) & (v <= 2000),
    "Extreme Vd (>2000 L)":        lambda v: v > 2000,
}
N_BOOT = 10000           # paired bootstrap iterations (Table 4)
BOOT_SEED = 0            # RNG seed for the bootstrap
MIN_N_FOR_STATS = 5      # cohorts smaller than this report point estimates only

REF_BODY_WEIGHT_KG = 70.0   # reference weight for L -> L/kg conversion (Table 5; Valentin 2002)

# Case-study drugs (Figure 7 diamonds / Table 5); names match df_ref "Name"
CASE_STUDY_DRUGS = ["PENICILLIN V", "LOSARTAN", "SECOBARBITAL", "VENLAFAXINE"]

N_SPLITS = 5
SPLIT_SEED = 42          # GroupShuffleSplit seed that reproduces the 263-compound test set
TEST_SIZE = 0.20


def set_pub_style():
    """Apply the manuscript's 300-DPI publication matplotlib style."""
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "DejaVu Sans"],
        "font.size": 11,
        "axes.titlesize": 17, "axes.labelsize": 14,
        "xtick.labelsize": 12, "ytick.labelsize": 12,
        "legend.fontsize": 12, "legend.title_fontsize": 13,
        "pdf.fonttype": 42, "ps.fonttype": 42,   # editable text in vector output
    })
