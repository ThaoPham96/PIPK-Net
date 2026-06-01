"""Reproducibility package for the PIPK-Net manuscript.

Pipeline
--------
1. ``data``        -- load the clinical reference table (e-Drug3D derived).
2. ``predictions`` -- run/freeze per-fold and ensemble predictions for every
   model (GNN ablation variants, ChemBERTa, Chemprop, ChemLLM) on the
   independent test set.
3. ``figures.*``   -- one module per manuscript figure (group).
4. ``tables``      -- every manuscript/supplementary table, written as CSV.

Run ``python -m reproduce.run_all`` to regenerate everything.
"""

__version__ = "1.0.0"
