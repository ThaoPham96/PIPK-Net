# `reproduce/` — PIPK-Net figure & table reproduction

This package regenerates **only** the figures and tables that appear in the
PIPK-Net manuscript and supplementary information, from saved artefacts
(model checkpoints + frozen prediction CSVs). It contains no training code —
the GNN architecture, featurisers and inference engine are reused from the
sibling [`pipknet`](../pipknet) package.

## Quick start

```powershell
conda activate pk_project
cd D:\PIPK_Net_Release

# 1) one-time: run model inference and freeze predictions (needs torch + checkpoints)
python -m reproduce.predictions          # writes data/predictions/*.csv

# 2) regenerate everything from the frozen predictions
python -m reproduce.run_all              # outputs/figures/*.{png,pdf} + outputs/tables/*.csv
```

`run_all.py` auto-runs the freeze step if `data/predictions/` is empty; pass
`--freeze` to force it, `--figures` / `--tables` to run one stage. Individual
figures can also be run directly, e.g. `python -m reproduce.figures.fig5_scatter`.

## What maps to what

| Output | Paper | Module |
| --- | --- | --- |
| `figure1_dataset_diagnostics` | Fig 1A | `figures/fig1_dataset.py` |
| `figure2_nestedcv_ablation` | Fig 2 | `figures/fig2_4_boxplots.py` |
| `figure3_ablation_indep_test` | Fig 3 | `figures/fig2_4_boxplots.py` |
| `figure4_benchmark_indep_test` | Fig 4 | `figures/fig2_4_boxplots.py` |
| `figure5_predicted_vs_observed` (5×3) | Fig 5 | `figures/fig5_scatter.py` |
| `figure6_mass_balance_residual` | Fig 6 | `figures/fig6_massbalance.py` |
| `figure7_logp_fold_error` | Fig 7 | `figures/fig7_logp.py` |
| `figureS1_logp_error_all_tasks` | Fig S1 | `figures/fig7_logp.py` |
| `table1_distribution_stats` | Table 1 | `tables.py` |
| `table3_ablation_accuracy` | Table 3 | `tables.py` |
| `table4_vd_cohorts` | Table 4 | `tables.py` |
| `table5_case_studies` | Table 5 | `tables.py` |
| `tableS1…tableS3_hyperparams_*` | Tables S1–S3 | `tables.py` |
| `tableS4_aafe`, `tableS5_pearson`, `tableS6_spearman` | Tables S4–S6 | `tables.py` |
| `macro_average_summary` | Results §2.2.1 (text) | `tables.py` |

> Manuscript **Table 2** (architecture specification) is a hand-authored table and is not
> generated here. The cross-model macro-average quoted in the text is written to
> `macro_average_summary.csv` (and is the bottom row of Tables S4–S6).

## Module overview

- `config.py` — all paths, task definitions, palettes and constants.
- `data.py` — clinical reference table, RDKit descriptors, scaffold split (reproduces
  the 904/263 partition exactly), Table 1 statistics.
- `predictions.py` — per-fold + ensemble inference (GNN A/B/C, ChemBERTa) and loading of
  Chemprop/ChemLLM predictions; freezes everything to `data/predictions/`. Lightweight
  `load_*` helpers let figures/tables run without torch.
- `models_chemberta.py` — the ChemBERTa MLP head + feature attachment.
- `metrics.py` — AAFE, %kFE, Pearson/Spearman, alignment, paired bootstrap, mass-balance.
- `figures/` — one module per figure (group).
- `tables.py` — every table, written as CSV.

## Reproducibility notes

- The scaffold split (`GroupShuffleSplit`, seed 42, generic Bemis-Murcko scaffolds)
  reproduces the manuscript's 904 development / 263 independent-test partition and
  647 unique scaffolds exactly.
- The frozen GNN C_physio (PIPK-Net) ensemble matches the original predictions to
  `|Δlog10| = 0` on the independent test set.
- `macro_average_summary.csv` is the principled macro-average of the per-task metrics
  (Tables S4–S6); ChemLLM is excluded from the AAFE macro-average (its predictions are
  clipping-dominated, per the supplementary footnote).
- **Table 5** L/kg uses a 70 kg reference body weight (per the Methods text and the
  published values 0.33 / 0.48 / 1.31 / 7.08; the "72 kg" in the Table 5 footnote is a
  manuscript typo — the displayed values are the 70 kg conversions).
- ADMET-AI and DeepPK values in Table 5 are transcribed from the manuscript
  (`data/benchmarks/admet_deeppk_casestudy.csv`); only the PIPK-Net column is computed.
- Figure 7 and Figure S1 were authored here (no source existed in the working notebook);
  the within-fold accuracies match the manuscript caption (2-fold 39.5/41.6/40.5 %).
- Minor differences from the printed tables reflect manuscript inconsistencies, e.g.
  the supplementary Tmax `n` and the Table 1 half-life median (printed 0.00; the data
  median is 9.00 h).
- The 263-record independent test set contains **257 unique canonical SMILES** — three
  drugs appear under multiple e-Drug3D records (Protokylol ×2, Methylphenidate ×3,
  Tramadol ×4). Chemprop is SMILES-keyed, so its ensemble/alignment collapses these to
  the 257 unique SMILES (matching the original benchmark's `drop_duplicates`), while the
  index-keyed GNN/ChemBERTa variants evaluate on all 263 records. There are **0** test
  SMILES without a Chemprop prediction; the freeze step asserts this and warns otherwise.
