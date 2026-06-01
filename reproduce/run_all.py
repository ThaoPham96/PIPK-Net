"""Orchestrator: regenerate every PIPK-Net manuscript/supplementary figure and table.

Typical use
-----------
    python -m reproduce.run_all              # figures + tables from frozen predictions
    python -m reproduce.run_all --freeze      # (re)run model inference first, then all
    python -m reproduce.run_all --figures     # figures only
    python -m reproduce.run_all --tables      # tables only

The ``--freeze`` step requires torch / torch-geometric and the model checkpoints;
figures and tables otherwise read only the frozen CSVs under ``data/predictions/``.
"""
from __future__ import annotations

import argparse

from . import config as C
from . import predictions as P
from .figures import fig1_dataset, fig2_4_boxplots, fig5_scatter, fig6_massbalance, fig7_logp
from . import tables


def run_figures():
    print("\n=== Figures ===")
    fig1_dataset.make()        # Figure 1A
    fig2_4_boxplots.make()     # Figures 2, 3, 4
    fig5_scatter.make()        # Figure 5
    fig6_massbalance.make()    # Figure 6
    fig7_logp.make()           # Figures 7, S1


def run_tables():
    print("\n=== Tables ===")
    tables.make()              # Tables 1, 3, 4, 5, S1-S6 (+ macro_average_summary)


def main():
    ap = argparse.ArgumentParser(description="Reproduce PIPK-Net figures and tables")
    ap.add_argument("--freeze", action="store_true", help="re-run model inference first")
    ap.add_argument("--figures", action="store_true", help="figures only")
    ap.add_argument("--tables", action="store_true", help="tables only")
    args = ap.parse_args()

    if args.freeze or not C.TRUTH_CSV.exists():
        print("=== Freezing predictions (model inference) ===")
        P.freeze()

    # Default (neither flag) runs both; --figures or --tables restricts to one stage.
    do_fig = args.figures or not args.tables
    do_tab = args.tables or not args.figures
    if do_fig:
        run_figures()
    if do_tab:
        run_tables()
    print(f"\nDone. Figures -> {C.FIG_DIR}\n      Tables  -> {C.TAB_DIR}")


if __name__ == "__main__":
    main()
