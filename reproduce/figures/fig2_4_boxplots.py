"""Figures 2, 3 and 4 -- Pearson/Spearman box-and-whisker performance plots.

* Figure 2 -- nested 5-fold CV, ablation variants (A_baseline/B_ion/C_physio).
* Figure 3 -- independent test, ablation variants.
* Figure 4 -- independent test, benchmark comparison (PIPK-Net/Chemprop/ChemBERTa).

All three share the same two-panel (Pearson top, Spearman bottom) box plot with
the seven endpoints plus a macro-average column.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from .. import config as C
from .. import metrics as M
from .. import predictions as P


def _plot_box(df_plot, configs_order, legend_labels, palette, save_name,
              title_prefix, legend_title):
    """Two-panel box plot (Pearson/Spearman) across endpoints + macro-average."""
    C.set_pub_style()
    tasks_with_avg = [C.TASK_DISPLAY[t] for t in C.TASKS] + ["Average"]

    df_avg = (df_plot.groupby(["config", "Fold"])[["Pearson", "Spearman"]]
              .mean().reset_index())
    df_avg["Task"] = "AVERAGE"
    df_full = pd.concat([df_plot, df_avg], ignore_index=True)
    display_map = {**C.TASK_DISPLAY, "AVERAGE": "Average"}
    df_full["TaskDisplay"] = df_full["Task"].map(display_map)

    fig, axes = plt.subplots(2, 1, figsize=(18, 13), sharex=False, dpi=300)
    for ax, metric, ylabel in [(axes[0], "Pearson", "Pearson (r)"),
                               (axes[1], "Spearman", "Spearman (ρ)")]:
        sns.boxplot(data=df_full, x="TaskDisplay", y=metric, hue="config",
                    order=tasks_with_avg, hue_order=configs_order,
                    palette=palette, ax=ax, width=0.65, linewidth=1.3, fliersize=4)
        ax.set_title(f"{title_prefix} ({metric})", fontsize=17, fontweight="bold", pad=18)
        ax.set_ylabel(ylabel, fontsize=14, fontweight="bold")
        ax.set_xlabel("Pharmacokinetic Parameters", fontsize=14, fontweight="bold")
        ax.set_ylim([-0.1, 1.0])
        ax.set_xticks(range(len(tasks_with_avg)))
        ax.set_xticklabels(tasks_with_avg, fontsize=12, fontweight="bold")
        ax.tick_params(axis="y", labelsize=12)
        ax.grid(axis="y", linestyle="-", alpha=0.4, color="lightgray")
        ax.set_axisbelow(True)
        for x_pos in np.arange(0.5, 6.5, 1):
            ax.axvline(x=x_pos, color="gray", linestyle="--", alpha=0.5, linewidth=1.2)
        ax.axvline(x=6.5, color="black", linestyle="-.", alpha=0.7, linewidth=1.8)
        if ax.get_legend() is not None:
            ax.get_legend().remove()

    handles, _ = axes[1].get_legend_handles_labels()
    legend = fig.legend(handles, legend_labels, loc="lower center",
                        bbox_to_anchor=(0.5, -0.01), ncol=len(configs_order),
                        fontsize=12, framealpha=1.0, edgecolor="black",
                        title=legend_title, title_fontsize=13)
    legend.get_title().set_fontweight("bold")
    plt.tight_layout()
    fig.subplots_adjust(bottom=0.10, hspace=0.32)

    base = C.FIG_DIR / save_name
    fig.savefig(base.with_suffix(".png"), bbox_inches="tight", dpi=300)
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf")
    return df_full


def make_figure2():
    """Figure 2 -- nested 5-fold CV ablation box plots."""
    df_all = P.load_nested_cv()
    col_map = {t: (f"{disp}", f"{disp}_true") for t, disp in {
        "t_half": "t_half (h)", "Vd": "Vd (L)", "CL": "CL (L/h)", "F": "F (%)",
        "PPB": "PPB (%)", "Cmax": "Cmax (uM)", "Tmax": "Tmax (h)"}.items()}
    df = M.per_fold_metrics_nested_cv(df_all, col_map)
    return _plot_box(df, C.GNN_VARIANTS, C.ABLATION_LABELS, C.ABLATION_PALETTE,
                     "figure2_nestedcv_ablation",
                     "Nested 5-Fold Cross-Validation Performance", "Ensemble Variants")


def make_figure3():
    """Figure 3 -- independent-test ablation box plots."""
    truth = P.load_truth()
    parts = [M.per_fold_metrics(P.load_perfold("gnn", v), truth, v) for v in C.GNN_VARIANTS]
    df = pd.concat(parts, ignore_index=True)
    return _plot_box(df, C.GNN_VARIANTS, C.ABLATION_LABELS, C.ABLATION_PALETTE,
                     "figure3_ablation_indep_test",
                     "Ensemble Performance on Independent Test Set", "Ensemble Variants")


def make_figure4():
    """Figure 4 -- independent-test benchmark box plots."""
    truth = P.load_truth()
    df = pd.concat([
        M.per_fold_metrics(P.load_perfold("gnn", "C_physio"), truth, "PIPK-Net"),
        M.per_fold_metrics(P.load_perfold("chemprop"), truth, "Chemprop", smiles_col=C.SMILES_COL),
        M.per_fold_metrics(P.load_perfold("chemberta"), truth, "ChemBERTa"),
    ], ignore_index=True)
    order = ["ChemBERTa", "Chemprop", "PIPK-Net"]
    return _plot_box(df, order, order, C.BENCH_PALETTE,
                     "figure4_benchmark_indep_test",
                     "Ensemble Performance on Independent Test Set", "Benchmark Models")


def make():
    make_figure2()
    make_figure3()
    make_figure4()


if __name__ == "__main__":
    make()
