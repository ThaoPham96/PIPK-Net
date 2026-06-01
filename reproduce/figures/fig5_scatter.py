"""Figure 5 -- predicted vs observed Vd, CL and t1/2 on the independent test set,
stratified by ionisation state.

A 5x3 grid matching the published Figure 5:
  rows  (A) A_baseline, (B) B_ion, (C) PIPK-Net (C_physio), (D) ChemBERTa, (E) Chemprop
  cols  Vd, CL, t1/2
Points are coloured by ionisation state; R^2 is the coefficient of determination
on log10-transformed values, with the identity line for reference. Stepwise R^2
gains down the GNN rows (A->B->C) show the contribution of the ionisation
embedding and the mass-balance constraint to the high-Vd regime.
"""
from __future__ import annotations

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import r2_score

from .. import config as C
from .. import predictions as P

PARAMS = [("Vd", r"Volume of Distribution ($V_d$)"),
          ("CL", r"Clearance ($CL$)"),
          ("t_half", r"Half-life ($t_{1/2}$)")]

# rows in published order: model label -> (loader key, smiles_col)
ROWS = [("$A_{baseline}$", "gnn_A_baseline", None),
        ("$B_{ion}$", "gnn_B_ion", None),
        ("PIPK-Net ($C_{physio}$)", "pipknet", None),
        ("ChemBERTa", "chemberta", None),
        ("Chemprop", "chemprop", C.SMILES_COL)]


def _align(task, pred_df, truth_df, smiles_col=None):
    truth_col = C.TRUTH_COLS[task]
    if smiles_col is None:
        df = truth_df[[truth_col, "IonType"]].rename(columns={truth_col: "obs", "IonType": "ion"}).copy()
        df["pred"] = pred_df[task].reindex(df.index)
    else:
        df = truth_df[[smiles_col, truth_col, "IonType"]].rename(
            columns={smiles_col: "smiles", truth_col: "obs", "IonType": "ion"}).copy()
        df = df[~df["smiles"].duplicated(keep="first")]
        pr = pred_df[[task]].rename(columns={task: "pred"})
        pr = pr[~pr.index.duplicated(keep="first")]
        df = df.merge(pr, left_on="smiles", right_index=True, how="left")
    df = df[["pred", "obs", "ion"]].dropna()
    return df[(df["pred"] > 0) & (df["obs"] > 0)]


def make():
    truth = P.load_truth()
    C.set_pub_style()
    plt.rcParams.update({"axes.titlesize": 9, "axes.labelsize": 9})
    n_rows = len(ROWS)
    fig, axes = plt.subplots(n_rows, 3, figsize=(11, 3.6 * n_rows))

    for r, (model_name, key, smiles_arg) in enumerate(ROWS):
        pred_df = P.load_ensemble(key)
        for c, (task, task_label) in enumerate(PARAMS):
            ax = axes[r, c]
            d = _align(task, pred_df, truth, smiles_col=smiles_arg)
            log_pred = np.log10(d["pred"]).to_numpy()
            log_obs = np.log10(d["obs"]).to_numpy()
            for ion_label, color in C.ION_COLORS.items():
                m = (d["ion"] == ion_label).to_numpy()
                if m.sum():
                    ax.scatter(log_pred[m], log_obs[m], c=color, s=16, alpha=0.75,
                               edgecolors="none", label=ion_label)
            lo, hi = min(log_pred.min(), log_obs.min()), max(log_pred.max(), log_obs.max())
            mg = 0.1 * (hi - lo)
            ax.plot([lo - mg, hi + mg], [lo - mg, hi + mg], color="grey", ls="--", lw=0.8, alpha=0.6)
            ax.set_xlim(lo - mg, hi + mg); ax.set_ylim(lo - mg, hi + mg)
            ax.set_aspect("equal", adjustable="box")
            r2 = r2_score(log_obs, log_pred)
            panel = f"({chr(65 + r)}{c + 1})"
            ax.set_title(f"{panel} {model_name}: {task_label}\n$R^2$ = {r2:.3f} | N = {len(d)}",
                         fontsize=8.5, fontweight="bold")
            ax.set_xlabel(r"Predicted log$_{10}$ value", fontsize=8.5)
            ax.set_ylabel(r"Observed log$_{10}$ value", fontsize=8.5)
            ax.tick_params(labelsize=7.5)
            ax.grid(True, alpha=0.2)

    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=4, fontsize=10,
               bbox_to_anchor=(0.5, -0.015), frameon=True,
               title="Ionisation State", title_fontsize=10)
    fig.suptitle(r"Predicted vs. Observed $V_d$, $CL$, and $t_{1/2}$ stratified by ionisation state",
                 fontsize=13, y=0.995, fontweight="bold")
    plt.tight_layout(); plt.subplots_adjust(bottom=0.06)

    base = C.FIG_DIR / "figure5_predicted_vs_observed"
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf")


if __name__ == "__main__":
    make()
