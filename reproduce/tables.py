"""Manuscript and supplementary tables, each written as a tidy CSV.

Tables (manuscript numbering)
-----------------------------
Table 1  -- PK distribution statistics (Development vs Test).
Table 3  -- ablation prediction accuracy per PK target (AAFE/%2FE/%5FE) for A/B/C.
Table 4  -- Vd cohort comparison (mean pred, AAFE, %2FE, ΔAAFE 95% CI, Wilcoxon p).
Table 5  -- case studies vs ADMET-AI / DeepPK (L/kg); PIPK-Net column computed.
Tables S1-S3 -- optimised hyperparameters per fold, per GNN variant.
Tables S4-S6 -- per-task AAFE / Pearson / Spearman across the 4 models (incl. ChemLLM).

(Manuscript Table 2 is the architecture-specification table, written by hand, not
generated here. The macro-average across the 4 models -- quoted in Results 2.2.1 and
the bottom row of Tables S4-S6 -- is written to ``macro_average_summary.csv``.)

Output: outputs/tables/*.csv
"""
from __future__ import annotations

import json
import numpy as np
import pandas as pd

from . import config as C
from . import data as D
from . import metrics as M
from . import predictions as P

CHEMLLM_CLIP = 1e-4


def _save(df, name):
    path = C.TAB_DIR / f"{name}.csv"
    df.to_csv(path, index=False)
    print(f"[table] {name}: {df.shape[0]}x{df.shape[1]} -> {path.name}")
    return df


# ---------------------------------------------------------------------------
# Table 1 -- distribution statistics
# ---------------------------------------------------------------------------
def table_distribution(df_ref):
    return _save(D.distribution_table(df_ref), "table1_distribution_stats")


# ---------------------------------------------------------------------------
# Table 3 -- ablation prediction accuracy
# ---------------------------------------------------------------------------
def table_accuracy(truth):
    rows = []
    for variant in C.GNN_VARIANTS:
        ens = P.load_ensemble(f"gnn_{variant}")
        for task in C.TASKS:
            pred, obs = M.align_pred_obs(task, ens, truth)
            yt = pd.to_numeric(obs, errors="coerce").to_numpy()
            yp = pd.to_numeric(pred, errors="coerce").to_numpy()
            _, aafe, p2, p5, n = M.calculate_fold_metrics(yt, yp)
            rows.append({"PK Parameter": task, "n": n, "Model Variant": variant,
                         "AAFE": round(aafe, 2), "%2FE": round(p2, 1), "%5FE": round(p5, 1)})
    return _save(pd.DataFrame(rows), "table3_ablation_accuracy")


# ---------------------------------------------------------------------------
# Tables S4-S6 (per-task metrics) + macro-average summary
# ---------------------------------------------------------------------------
def _per_task_all_models(truth):
    models = [("PIPK-Net", P.load_ensemble("pipknet"), None, None),
              ("Chemprop", P.load_ensemble("chemprop"), C.SMILES_COL, None),
              ("ChemBERTa", P.load_ensemble("chemberta"), None, None),
              ("ChemLLM", P.load_ensemble("chemllm"), None, CHEMLLM_CLIP)]
    rows = []
    for name, df, sc, clip in models:
        for task in C.TASKS:
            pred, obs = M.align_pred_obs(task, df, truth, smiles_col=sc)
            m = M.metrics_unified(pred, obs, task, clip_floor=clip)
            if m is None:
                continue
            rows.append({"Model": name, "Task": task, **m})
    return pd.DataFrame(rows)


def tables_per_task_and_macro(truth):
    per_task = _per_task_all_models(truth)
    model_order = ["ChemBERTa", "Chemprop", "PIPK-Net", "ChemLLM"]
    task_order = ["Vd", "CL", "t_half", "F", "PPB", "Cmax", "Tmax"]

    out = {}
    for metric, fname in [("AAFE", "tableS4_aafe"), ("Pearson_r", "tableS5_pearson"),
                          ("Spearman_rho", "tableS6_spearman")]:
        piv = per_task.pivot(index="Task", columns="Model", values=metric).reindex(task_order)
        piv = piv[[m for m in model_order if m in piv.columns]]
        # macro-average; ChemLLM AAFE excluded (clipping-dominated, per supp footnote)
        avg = {}
        for m in piv.columns:
            if metric == "AAFE" and m == "ChemLLM":
                avg[m] = np.nan
            else:
                avg[m] = piv[m].mean()
        piv.loc["Macro-average"] = avg
        out[metric] = piv
        _save(piv.round(3).reset_index().rename(columns={"index": "Task"}), fname)

    # Macro-average summary across the three metrics (quoted in Results 2.2.1; not a
    # numbered table). The Notes column flags that ChemLLM AAFE is deliberately blank.
    models = ["ChemBERTa", "Chemprop", "PIPK-Net", "ChemLLM"]
    summary = pd.DataFrame({
        "Metric": ["AAFE", "Pearson_r", "Spearman_rho"],
        **{m: [out["AAFE"].loc["Macro-average", m] if m in out["AAFE"].columns else np.nan,
               out["Pearson_r"].loc["Macro-average", m],
               out["Spearman_rho"].loc["Macro-average", m]]
           for m in models},
    }).round(3)
    summary["Notes"] = ["ChemLLM AAFE excluded: zero-prediction clipping (see Table S4 footnote)", "", ""]
    _save(summary, "macro_average_summary")
    return per_task


# ---------------------------------------------------------------------------
# Table 4 -- Vd cohort comparison
# ---------------------------------------------------------------------------
def table_vd_cohorts(truth):
    pipk = P.load_ensemble("pipknet")
    benches = [("Chemprop", P.load_ensemble("chemprop"), C.SMILES_COL),
               ("ChemBERTa", P.load_ensemble("chemberta"), None)]
    rows = []
    for cohort, fn in C.VD_COHORTS.items():
        for bench_name, bench_df, sc in benches:
            # aligned (pipk, bench, obs) triples, restricted to the observed-Vd cohort
            tri = M.align_three("Vd", pipk, bench_df, truth, smiles_col=sc)
            tri = tri[tri["obs"].apply(fn)]
            n = len(tri)
            if n == 0:
                continue
            pk, bn, ob = tri["pipk"].to_numpy(), tri["bench"].to_numpy(), tri["obs"].to_numpy()
            rec = {"Cohort": cohort, "Benchmark": bench_name, "n": n,
                   "Obs mean Vd (L)": round(ob.mean(), 1),
                   "PIPK-Net mean (L)": round(pk.mean(), 1),
                   "Bench mean (L)": round(bn.mean(), 1),
                   "PIPK-Net AAFE": round(M.aafe(pk, ob), 2),
                   "Bench AAFE": round(M.aafe(bn, ob), 2),
                   "PIPK-Net %2FE": round(M.pct_within(pk, ob, 2), 1),
                   "Bench %2FE": round(M.pct_within(bn, ob, 2), 1)}
            # dAAFE = AAFE(PIPK-Net) - AAFE(benchmark); negative => PIPK-Net closer
            if n >= C.MIN_N_FOR_STATS:
                d, lo, hi = M.paired_diff_ci(pk, bn, ob, M.aafe)
                rec["dAAFE (PIPK-bench)"] = round(d, 3)
                rec["dAAFE 95% CI"] = f"({lo:+.2f}, {hi:+.2f})"
                rec["Wilcoxon p"] = round(M.wilcoxon_logfe(pk, bn, ob), 4)
                rec["Favours PIPK-Net"] = bool(hi < 0)
            else:
                rec["dAAFE (PIPK-bench)"] = round(M.aafe(pk, ob) - M.aafe(bn, ob), 3)
                rec["dAAFE 95% CI"] = f"n.s. (n<{C.MIN_N_FOR_STATS})"
                rec["Wilcoxon p"] = np.nan
                rec["Favours PIPK-Net"] = np.nan
            rows.append(rec)
    return _save(pd.DataFrame(rows), "table4_vd_cohorts")


# ---------------------------------------------------------------------------
# Table 5 -- case studies vs ADMET-AI / DeepPK
# ---------------------------------------------------------------------------
def table_case_studies():
    ref = pd.read_csv(C.BENCH_DIR / "admet_deeppk_casestudy.csv")
    case = pd.read_csv(C.PRED_DIR / "casestudy_pipknet.csv")
    case["_nm"] = case["Name"].str.upper().str.strip()
    ref["_nm"] = ref["Drug"].str.upper().str.strip()
    merged = ref.merge(case[["_nm", "pred_Vd"]], on="_nm", how="left")
    merged["PIPK-Net_Lkg"] = (merged["pred_Vd"] / C.REF_BODY_WEIGHT_KG).round(2)

    def _in_range(val, rng):
        if not isinstance(rng, str) or "-" not in rng:
            return ""
        lo, hi = [float(x) for x in rng.split("-")]
        return "in range" if (pd.notna(val) and lo <= val <= hi) else "outside"

    merged["PIPK-Net_status"] = merged.apply(
        lambda r: _in_range(r["PIPK-Net_Lkg"], r["Clinical_range_Lkg"]), axis=1)
    cols = ["ID", "Drug", "Clinical_Vd_Lkg", "Clinical_range_Lkg", "Clinical_range_source",
            "PIPK-Net_Lkg", "PIPK-Net_status", "ADMET_AI_Lkg", "DeepPK_Lkg"]
    out = merged[cols].copy()
    out.attrs["note"] = f"PIPK-Net L/kg = predicted Vd (L) / {C.REF_BODY_WEIGHT_KG:g} kg"
    return _save(out, "table5_case_studies")


# ---------------------------------------------------------------------------
# Tables S1-S3 -- optimised hyperparameters
# ---------------------------------------------------------------------------
def tables_hyperparams():
    for si, variant in zip([1, 2, 3], C.GNN_VARIANTS):
        cols = {}
        for fold in range(1, C.N_SPLITS + 1):
            cfg = json.load(open(C.CHECKPOINTS_DIR / variant / f"fold_{fold}" / "config.json"))
            cols[f"Ensemble {fold}"] = cfg["best_hp"]
        df = pd.DataFrame(cols)
        df.index.name = "Hyperparameter"
        _save(df.reset_index(), f"tableS{si}_hyperparams_{variant}")


# ---------------------------------------------------------------------------
# Orchestrator
# ---------------------------------------------------------------------------
def make():
    df_ref = D.build_reference()
    truth = P.load_truth()
    table_distribution(df_ref)        # Table 1
    table_accuracy(truth)             # Table 3
    table_vd_cohorts(truth)           # Table 4
    table_case_studies()              # Table 5
    tables_per_task_and_macro(truth)  # Tables S4-S6 (+ macro_average_summary)
    tables_hyperparams()              # Tables S1-S3
    print("[table] all tables written to", C.TAB_DIR)


if __name__ == "__main__":
    make()
