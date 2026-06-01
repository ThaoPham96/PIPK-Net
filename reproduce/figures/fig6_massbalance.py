"""Figure 6 -- mass-balance constraint residual across Vd cohorts.

For each compound and model, the absolute residual
    |log10(t_half) - (log10(ln2) + log10(Vd) - log10(CL))|
is computed on the model's ensemble predictions; values near zero indicate the
predicted triad respects t1/2 = ln(2)*Vd/CL. Compounds are stratified into Low
(<50 L), Medium-High (300-2000 L) and Extreme (>2000 L) observed-Vd cohorts.
The physiology-informed loss makes PIPK-Net's residuals collapse toward zero
relative to the benchmarks, which have no mechanism coupling the triad.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .. import config as C
from .. import predictions as P

MODEL_ORDER = ["PIPK-Net", "Chemprop", "ChemBERTa"]
COLORS = {"PIPK-Net": "#31a354", "Chemprop": "#9e7ac4", "ChemBERTa": "#d6a36b"}


def _residual_frame(model_df, truth, obs_vd, smiles_col=None):
    """Per-compound absolute mass-balance residual + observed Vd (for cohorting)."""
    if smiles_col is None:
        d = model_df[["t_half", "Vd", "CL"]].copy()
        d["obs_vd"] = obs_vd.reindex(d.index)
    else:
        t = truth[[smiles_col]].copy()
        t["obs_vd"] = obs_vd.values
        t = t[~t[smiles_col].duplicated(keep="first")]
        pr = model_df[["t_half", "Vd", "CL"]]
        pr = pr[~pr.index.duplicated(keep="first")]
        d = t.merge(pr, left_on=smiles_col, right_index=True, how="left")
    d = d[["t_half", "Vd", "CL", "obs_vd"]].apply(pd.to_numeric, errors="coerce").dropna()
    d = d[(d[["t_half", "Vd", "CL"]] > 0).all(axis=1)]
    expected = np.log10(np.log(2)) + np.log10(d["Vd"]) - np.log10(d["CL"])
    d["resid"] = (np.log10(d["t_half"]) - expected).abs()
    return d[["obs_vd", "resid"]]


def make():
    truth = P.load_truth()
    obs_vd = pd.to_numeric(truth[C.TRUTH_COLS["Vd"]], errors="coerce")
    frames = {
        "PIPK-Net": _residual_frame(P.load_ensemble("pipknet"), truth, obs_vd),
        "ChemBERTa": _residual_frame(P.load_ensemble("chemberta"), truth, obs_vd),
        "Chemprop": _residual_frame(P.load_ensemble("chemprop"), truth, obs_vd, smiles_col=C.SMILES_COL),
    }

    rows = []
    for model, d in frames.items():
        for cohort, fn in C.VD_COHORTS.items():
            sub = d[d["obs_vd"].apply(fn)]
            for v in sub["resid"]:
                rows.append({"Model": model, "Cohort": cohort, "Abs Residual": v})
    plot_df = pd.DataFrame(rows)
    cohort_order = list(C.VD_COHORTS.keys())

    C.set_pub_style()
    plt.rcParams.update({"axes.titlesize": 11})
    fig, axes = plt.subplots(1, 3, figsize=(14, 5.5), sharey=True)
    for ax, cohort in zip(axes, cohort_order):
        sub = plot_df[plot_df["Cohort"] == cohort]
        counts = sub.groupby("Model").size().reindex(MODEL_ORDER).fillna(0).astype(int)
        sns.boxplot(data=sub, x="Model", y="Abs Residual", hue="Model",
                    order=MODEL_ORDER, hue_order=MODEL_ORDER, palette=COLORS,
                    ax=ax, width=0.55, fliersize=0, legend=False)
        sns.stripplot(data=sub, x="Model", y="Abs Residual", order=MODEL_ORDER,
                      ax=ax, color="black", size=3, alpha=0.45, jitter=0.15)
        for y_ref in [0.05, 0.10, 0.30]:
            ax.axhline(y_ref, color="grey", ls="--", lw=0.6, alpha=0.6)
        if ax is axes[-1]:
            for y_ref, lab in [(0.05, "1.12×"), (0.10, "1.26×"), (0.30, "2.0×")]:
                ax.text(1.02, y_ref, lab, va="center", ha="left", fontsize=9,
                        color="grey", transform=ax.get_yaxis_transform())
        ax.set_xticks(range(len(MODEL_ORDER)))
        ax.set_xticklabels([f"{m}\nn={counts[m]}" for m in MODEL_ORDER], rotation=0, fontsize=9)
        ax.set_title(cohort, fontsize=11)
        ax.set_xlabel(""); ax.set_ylim(0, 0.4); ax.grid(axis="y", alpha=0.2)
    axes[0].set_ylabel("Absolute mass-balance residual\n(log$_{10}$ scale)", fontsize=10)
    fig.suptitle(r"Mass-balance constraint residual across V$_d$ cohorts", fontsize=12, y=1.02)
    plt.tight_layout(); plt.subplots_adjust(left=0.08, right=0.95, top=0.90)

    base = C.FIG_DIR / "figure6_mass_balance_residual"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    med = plot_df.groupby(["Model", "Cohort"])["Abs Residual"].median().round(3)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf")
    print("[fig] median residuals:\n" + med.to_string())


if __name__ == "__main__":
    make()
