"""Figure 7 and Figure S1 -- prediction error across lipophilicity space.

Reproduces manuscript Figure 7 and supplementary Figure S1 from the frozen
PIPK-Net ensemble predictions:

* Figure 7  -- per-compound log2(pred/obs) vs LogP for Vd, CL and t1/2, with
  2-fold (grey) and 10-fold (black) reference bands, points coloured by
  ionisation state, and the four case-study drugs marked as numbered diamonds
  with an appended observed/predicted Vd table.
* Figure S1 -- the same log2(pred/obs)-vs-LogP view for all seven PK endpoints.

Ionisation classes use the supplementary colour scheme (Anionic≈acidic: green,
Cationic≈basic: orange, Neutral: grey, Zwitterionic: purple).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .. import config as C
from .. import predictions as P

LOG2_2FE = np.log2(2.0)      # = 1.0
LOG2_10FE = np.log2(10.0)    # ≈ 3.3219

ION_COLORS_LOGP = {"Anionic": "#2ca25f", "Cationic": "#F57C00",
                   "Neutral": "#7f7f7f", "Zwitterionic": "#7b3294"}


def _error_frame(task, pred_df, truth, smiles_col=None):
    """Per-compound log2(pred/obs) with LogP and ionisation, for one task."""
    truth_col = C.TRUTH_COLS[task]
    base = truth[[truth_col, "LogP", "IonType"]].rename(columns={truth_col: "obs"}).copy()
    if smiles_col is None:
        base["pred"] = pred_df[task].reindex(base.index)
    else:
        base["smiles"] = truth[smiles_col].values
        base = base[~base["smiles"].duplicated(keep="first")]
        pr = pred_df[[task]].rename(columns={task: "pred"})
        pr = pr[~pr.index.duplicated(keep="first")]
        base = base.merge(pr, left_on="smiles", right_index=True, how="left")
    base = base[["pred", "obs", "LogP", "IonType"]].apply(
        lambda s: pd.to_numeric(s, errors="coerce") if s.name != "IonType" else s).dropna()
    base = base[(base["pred"] > 0) & (base["obs"] > 0)]
    base["log2fe"] = np.log2(base["pred"] / base["obs"])
    return base


def _scatter_panel(ax, df, title):
    for ion, color in ION_COLORS_LOGP.items():
        m = df["IonType"] == ion
        if m.any():
            ax.scatter(df.loc[m, "LogP"], df.loc[m, "log2fe"], c=color, s=22,
                       alpha=0.75, edgecolors="none")
    for y in (LOG2_2FE, -LOG2_2FE):
        ax.axhline(y, color="grey", ls="--", lw=0.9, alpha=0.7)
    for y in (LOG2_10FE, -LOG2_10FE):
        ax.axhline(y, color="black", ls="--", lw=0.9, alpha=0.7)
    ax.axhline(0, color="black", ls="-", lw=0.6, alpha=0.4)
    ax.set_ylabel(title, fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.15)


def make_figure7():
    truth = P.load_truth()
    pipk = P.load_ensemble("pipknet")
    tasks = [("Vd", r"$V_d$  Median $\log_2(\frac{Pred}{Obs})$"),
             ("CL", r"$CL$  Median $\log_2(\frac{Pred}{Obs})$"),
             ("t_half", r"$t_{1/2}$  Median $\log_2(\frac{Pred}{Obs})$")]
    frames = {t: _error_frame(t, pipk, truth) for t, _ in tasks}

    case = pd.read_csv(C.PRED_DIR / "casestudy_pipknet.csv") if (C.PRED_DIR / "casestudy_pipknet.csv").exists() else None

    C.set_pub_style()
    plt.rcParams.update({"axes.titlesize": 12})
    fig = plt.figure(figsize=(11, 12))
    gs = fig.add_gridspec(4, 1, height_ratios=[1, 1, 1, 0.45], hspace=0.28)
    axes = [fig.add_subplot(gs[i]) for i in range(3)]

    within2, within10 = {}, {}
    for ax, (task, ylab) in zip(axes, tasks):
        df = frames[task]
        _scatter_panel(ax, df, ylab)
        within2[task] = 100 * (df["log2fe"].abs() <= LOG2_2FE).mean()
        within10[task] = 100 * (df["log2fe"].abs() <= LOG2_10FE).mean()
        # Case-study diamonds appear on ALL three panels at each drug's task-specific
        # log2(pred/obs) (matching manuscript Figure 7); the appended table lists the
        # Vd values only, as in the manuscript.
        if case is not None:
            for _, cr in case.iterrows():
                pred, obs = cr.get(f"pred_{task}"), cr.get(f"obs_{task}")
                if pd.notna(pred) and pd.notna(obs) and pred > 0 and obs > 0:
                    y = np.log2(pred / obs)
                    ax.scatter(cr["LogP"], y, marker="D", s=90, facecolors="none",
                               edgecolors="black", linewidths=1.6, zorder=5)
                    ax.annotate(str(int(cr["ID"])), (cr["LogP"], y), fontsize=9,
                                fontweight="bold", ha="center", va="center", zorder=6)
                    ax.axvline(cr["LogP"], color="grey", ls=":", lw=0.7, alpha=0.5)
    axes[-1].set_xlabel("LogP (Lipophilicity)", fontsize=12, fontweight="bold")

    # ionisation counts for legend (test set)
    counts = truth["IonType"].value_counts()
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=col,
                          label=f"{ion} (n={int(counts.get(ion, 0))})")
               for ion, col in ION_COLORS_LOGP.items()]
    handles += [plt.Line2D([0], [0], marker="D", ls="", markerfacecolor="none",
                           markeredgecolor="black", label="Case-study drug")]
    axes[0].legend(handles=handles, loc="upper left", fontsize=8, frameon=True, ncol=1)
    axes[0].set_title("Prediction Error Across Lipophilicity Space", fontsize=13, fontweight="bold")

    # appended case-study Vd table
    ax_tab = fig.add_subplot(gs[3]); ax_tab.axis("off")
    if case is not None:
        cell = [[int(r["ID"]), r["Name"].title(), f"{r['obs_Vd']:.2f}", f"{r['pred_Vd']:.2f}"]
                for _, r in case.sort_values("ID").iterrows()]
        tbl = ax_tab.table(cellText=cell,
                           colLabels=["ID", "Case Study Drug", "Observed Vd (L)", "Predicted Vd (L)"],
                           loc="center", cellLoc="center")
        tbl.auto_set_font_size(False); tbl.set_fontsize(9); tbl.scale(1, 1.4)

    base = C.FIG_DIR / "figure7_logp_fold_error"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf")
    print("[fig] within-2-fold: " + ", ".join(f"{k}={within2[k]:.1f}%" for k, _ in tasks))
    print("[fig] within-10-fold: " + ", ".join(f"{k}={within10[k]:.1f}%" for k, _ in tasks))


def make_figureS1():
    truth = P.load_truth()
    pipk = P.load_ensemble("pipknet")
    C.set_pub_style()
    plt.rcParams.update({"axes.titlesize": 11})
    fig, axes = plt.subplots(4, 2, figsize=(12, 14))
    axes = axes.flatten()
    for i, task in enumerate(C.TASKS):
        df = _error_frame(task, pipk, truth)
        _scatter_panel(axes[i], df, rf"{C.TASK_DISPLAY[task]}  $\log_2(\frac{{Pred}}{{Obs}})$")
        axes[i].set_xlabel("LogP", fontsize=10, fontweight="bold")
        axes[i].set_title(C.TASK_DISPLAY[task], fontsize=11, fontweight="bold")
    # legend in the spare 8th panel
    axes[7].axis("off")
    counts = truth["IonType"].value_counts()
    handles = [plt.Line2D([0], [0], marker="o", ls="", color=col,
                          label=f"{ion} (n={int(counts.get(ion, 0))})")
               for ion, col in ION_COLORS_LOGP.items()]
    axes[7].legend(handles=handles, loc="center", fontsize=12, frameon=True,
                   title="Ionisation State", title_fontsize=13)
    fig.suptitle("PIPK-Net Prediction Error Across LogP Space", fontsize=15, fontweight="bold", y=1.0)
    plt.tight_layout()

    base = C.FIG_DIR / "figureS1_logp_error_all_tasks"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf")


def make():
    make_figure7()
    make_figureS1()


if __name__ == "__main__":
    make()
