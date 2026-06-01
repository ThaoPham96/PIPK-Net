"""Metrics and alignment helpers shared by the figure and table modules.

All metrics follow the manuscript conventions:
  * AAFE  = 10 ** mean(|log10(pred/obs)|)            (geometric fold error)
  * %kFE  = fraction of |fold error| within k-fold
  * Pearson r computed on log10 values for log-distributed tasks, linear for %
  * Spearman rho is rank-based (transform-invariant)
Predictions and observations are aligned on row_idx (GNN/ChemBERTa/ChemLLM,
indexed by ORIG_INDEX) or on SMILES (Chemprop), and restricted to strictly
positive pairs so the log transform is defined.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr, wilcoxon

from . import config as C


# ---------------------------------------------------------------------------
# Scalar metrics
# ---------------------------------------------------------------------------
def aafe(pred, obs) -> float:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    return float(10 ** np.mean(np.abs(np.log10(pred / obs))))


def afe(pred, obs) -> float:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    return float(10 ** np.mean(np.log10(pred / obs)))


def pct_within(pred, obs, k) -> float:
    pred, obs = np.asarray(pred, float), np.asarray(obs, float)
    fe = np.maximum(pred / obs, obs / pred)
    return float(100.0 * np.mean(fe <= k))


def _pearson(p, o, task):
    if task in C.LOG_TASKS:
        return pearsonr(np.log10(p), np.log10(o))[0]
    return pearsonr(p, o)[0]


def calculate_fold_metrics(yt, yp):
    """(AFE, AAFE, %2FE, %5FE, n) on positive aligned pairs (Table 3 helper)."""
    yt, yp = np.asarray(yt, float), np.asarray(yp, float)
    valid = ~np.isnan(yt) & ~np.isnan(yp) & (yt > 0) & (yp > 0)
    yt, yp = yt[valid], yp[valid]
    n = len(yt)
    if n == 0:
        return np.nan, np.nan, np.nan, np.nan, 0
    return afe(yp, yt), aafe(yp, yt), pct_within(yp, yt, 2.0), pct_within(yp, yt, 5.0), n


def metrics_unified(pred: pd.Series, obs: pd.Series, task: str, clip_floor=None):
    """{n, AAFE, Pearson_r, Spearman_rho} on positive pairs (Tables S4-S6).

    If ``clip_floor`` is given, non-positive predictions are clipped to it
    (ChemLLM convention) instead of dropped.
    """
    p = pd.to_numeric(pred, errors="coerce")
    o = pd.to_numeric(obs, errors="coerce")
    df = pd.DataFrame({"p": p.values, "o": o.values}).dropna().astype(float)
    n_clipped = 0
    if clip_floor is not None:
        n_clipped = int((df["p"] <= 0).sum())
        df.loc[df["p"] <= 0, "p"] = clip_floor
        df = df[df["o"] > 0]
    else:
        df = df[(df["p"] > 0) & (df["o"] > 0)]
    if len(df) < 2:
        return None
    p, o = df["p"].to_numpy(), df["o"].to_numpy()
    out = {"n": len(df), "AAFE": aafe(p, o), "Pearson_r": _pearson(p, o, task),
           "Spearman_rho": spearmanr(p, o)[0]}
    if clip_floor is not None:
        out["n_clipped"] = n_clipped
    return out


def distribution_metrics(pred: pd.Series, obs: pd.Series):
    """{n, obs_mean, pred_mean, AAFE, %2FE, %5FE} on positive pairs (distribution/cohort helper)."""
    p = pd.to_numeric(pred, errors="coerce")
    o = pd.to_numeric(obs, errors="coerce")
    mask = (p > 0) & (o > 0) & p.notna() & o.notna()
    p, o = p[mask].to_numpy(), o[mask].to_numpy()
    if len(o) == 0:
        return None
    fold = p / o
    return {"n": len(o), "obs_mean": o.mean(), "pred_mean": p.mean(),
            "AAFE": aafe(p, o),
            "%2FE": 100 * ((fold >= 0.5) & (fold <= 2.0)).mean(),
            "%5FE": 100 * ((fold >= 0.2) & (fold <= 5.0)).mean()}


# ---------------------------------------------------------------------------
# Alignment (index- vs SMILES-based joins)
# ---------------------------------------------------------------------------
def align_pred_obs(task, pred_df, truth_df, smiles_col=None):
    """Return aligned (pred, obs) Series for a task. truth_df indexed by row_idx."""
    truth_col = C.TRUTH_COLS[task]
    if smiles_col is None:
        obs = truth_df[truth_col]
        pred = pred_df[task].reindex(obs.index)
    else:
        t = truth_df[[smiles_col, truth_col]].copy()
        t = t[~t[smiles_col].duplicated(keep="first")]
        pr = pred_df[[task]].rename(columns={task: "_p"})
        pr = pr[~pr.index.duplicated(keep="first")]
        m = t.merge(pr, left_on=smiles_col, right_index=True, how="left")
        obs, pred = m[truth_col], m["_p"]
    return pred, obs


def align_three(task, pipk_df, bench_df, truth_df, smiles_col=None):
    """Aligned (pipk, bench, obs) DataFrame on strictly-positive triples (Table 4)."""
    truth_col = C.TRUTH_COLS[task]
    if smiles_col is None:
        t = truth_df[[truth_col]].rename(columns={truth_col: "obs"}).copy()
        t["pipk"] = pipk_df[task].reindex(t.index).values
        t["bench"] = bench_df[task].reindex(t.index).values
    else:
        t = truth_df[[smiles_col, truth_col]].rename(columns={truth_col: "obs"}).copy()
        t["pipk"] = pipk_df[task].reindex(t.index).values
        t["smiles"] = t[smiles_col].astype(str)
        t = t[~t["smiles"].duplicated(keep="first")]
        bd = bench_df[[task]].rename(columns={task: "bench"})
        bd = bd[~bd.index.duplicated(keep="first")]
        t = t.merge(bd, left_on="smiles", right_index=True, how="left")
    t = t[["pipk", "bench", "obs"]].dropna()
    return t[(t["pipk"] > 0) & (t["bench"] > 0) & (t["obs"] > 0)]


def cohort_indices(task: str, truth_df: pd.DataFrame, cohort_filter) -> pd.Index:
    """row_idx of compounds whose observed ``task`` value falls in a cohort.

    e.g. ``cohort_indices("Vd", truth, lambda v: v < 50)`` for the low-Vd cohort.
    """
    obs = pd.to_numeric(truth_df[C.TRUTH_COLS[task]], errors="coerce").dropna()
    return obs[obs.apply(cohort_filter)].index


# ---------------------------------------------------------------------------
# Per-fold metrics (Figures 2/3/4 boxplots)
# ---------------------------------------------------------------------------
def per_fold_metrics(fold_dfs, truth_df: pd.DataFrame, label: str,
                     smiles_col: str | None = None) -> pd.DataFrame:
    """Pearson r and Spearman rho per (fold, task) for an ensemble's fold list."""
    rows = []
    for fi, df in enumerate(fold_dfs):
        for task in C.TASKS:
            pred, obs = align_pred_obs(task, df, truth_df, smiles_col)
            m = (pred > 0) & (obs > 0) & pred.notna() & obs.notna()
            p, o = pred[m].to_numpy(), obs[m].to_numpy()
            if len(o) < 2:
                continue
            rows.append({"Fold": fi, "Task": task, "config": label,
                         "Pearson": _pearson(p, o, task),
                         "Spearman": spearmanr(p, o)[0], "n": len(o)})
    return pd.DataFrame(rows)


def per_fold_metrics_nested_cv(df_all, pred_obs_cols, variant_col="config", fold_col="outer_fold"):
    """Per (variant, fold, task) metrics from the wide CV prediction table (Figure 2)."""
    rows = []
    for (variant, fold), grp in df_all.groupby([variant_col, fold_col]):
        for task in C.TASKS:
            pcol, ocol = pred_obs_cols[task]
            pred = pd.to_numeric(grp[pcol], errors="coerce")
            obs = pd.to_numeric(grp[ocol], errors="coerce")
            m = (pred > 0) & (obs > 0) & pred.notna() & obs.notna()
            p, o = pred[m].to_numpy(), obs[m].to_numpy()
            if len(o) < 2:
                continue
            rows.append({"Fold": fold, "Task": task, "config": variant,
                         "Pearson": _pearson(p, o, task),
                         "Spearman": spearmanr(p, o)[0], "n": len(o)})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Paired bootstrap (Table 4 ΔAAFE confidence intervals)
# ---------------------------------------------------------------------------
def paired_diff_ci(pipk, bench, obs, metric_fn, n_boot=C.N_BOOT, seed=C.BOOT_SEED):
    """Bootstrap 95% CI for metric(pipk,obs) - metric(bench,obs) (paired resamples).

    Negative ΔAAFE => PIPK-Net is closer to observed.
    """
    pipk, bench, obs = map(lambda a: np.asarray(a, float), (pipk, bench, obs))
    point = metric_fn(pipk, obs) - metric_fn(bench, obs)
    rng = np.random.default_rng(seed)
    n = len(obs)
    diffs = np.empty(n_boot)
    for b in range(n_boot):
        idx = rng.integers(0, n, n)
        diffs[b] = metric_fn(pipk[idx], obs[idx]) - metric_fn(bench[idx], obs[idx])
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return point, lo, hi


def wilcoxon_logfe(pipk, bench, obs):
    """Wilcoxon signed-rank p on per-compound absolute log-fold errors."""
    pipk, bench, obs = map(lambda a: np.asarray(a, float), (pipk, bench, obs))
    e_p = np.abs(np.log10(pipk / obs))
    e_b = np.abs(np.log10(bench / obs))
    try:
        return wilcoxon(e_p, e_b).pvalue
    except ValueError:
        return np.nan


# ---------------------------------------------------------------------------
# Mass-balance residual (Figure 6)
# ---------------------------------------------------------------------------
def mass_balance_residual(pred_df: pd.DataFrame) -> pd.Series:
    """|log10(t_half) - (log10(ln2) + log10(Vd) - log10(CL))| per compound."""
    d = pred_df[["t_half", "Vd", "CL"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d > 0).all(axis=1)]
    expected = np.log10(np.log(2)) + np.log10(d["Vd"]) - np.log10(d["CL"])
    return (np.log10(d["t_half"]) - expected).abs()
