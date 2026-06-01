"""ChemBERTa benchmark model.

The ChemBERTa benchmark uses frozen 768-dim transformer embeddings
(``seyonec/PubChem10M_SMILES_BPE_450k``) fed into a task-specific MLP, fine-tuned
with the same nested 5-fold protocol as PIPK-Net. This module defines the MLP
head and the helper that attaches the precomputed embeddings (and the ionisation
index) to the PyG graphs so the same fold checkpoints can be reloaded.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from pipknet.featurizers import ION_TYPES


class ChemBERTa_PK_MLP(nn.Module):
    """MLP head over 768-dim ChemBERTa features (optionally + ionisation embedding)."""

    def __init__(self, input_dim=768, hidden_dim=128, ion_dim=4, ion_emb_dim=16,
                 out_dim=7, use_ion=False, dropout=0.2):
        super().__init__()
        self.use_ion = use_ion
        if self.use_ion and ion_emb_dim > 0:
            self.ion_embedding = nn.Embedding(ion_dim, ion_emb_dim)
            combined = input_dim + ion_emb_dim
        else:
            self.ion_embedding = None
            combined = input_dim

        self.mlp = nn.Sequential(
            nn.Linear(combined, hidden_dim),
            nn.BatchNorm1d(hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, out_dim),
        )

    def forward(self, batch):
        x = batch.chemberta_feat.view(-1, 768).float()
        if self.use_ion and self.ion_embedding is not None:
            ion = self.ion_embedding(batch.ion_idx.long())
            x = torch.cat([x, ion], dim=1)
        return self.mlp(x)


def load_chemberta_feature_matrix(feats_csv, df_ref) -> np.ndarray:
    """Build the [N, 768] feature matrix aligned to df_ref rows, via CAS lookup.

    Returns a matrix indexed positionally by df_ref's ORIG_INDEX (zero vector
    where a CAS has no embedding).
    """
    chem = pd.read_csv(feats_csv, low_memory=False)
    feat_cols = [f"feat_{i}" for i in range(768)]
    lookup = (chem.drop_duplicates(subset=["CAS"]).set_index("CAS")[feat_cols]
              .to_dict("index"))

    def vec(cas):
        res = lookup.get(cas)
        return np.array(list(res.values()), dtype=np.float32) if res else np.zeros(768, np.float32)

    n = int(df_ref["ORIG_INDEX"].max()) + 1
    mat = np.zeros((n, 768), dtype=np.float32)
    for oi, cas in zip(df_ref["ORIG_INDEX"], df_ref["CAS"]):
        mat[int(oi)] = vec(cas)
    return mat


def attach_chemberta_features(pyg_list, feature_matrix, df_ref):
    """Attach ``chemberta_feat`` and ``ion_idx`` to each PyG graph (in place).

    Contract: ``data.row_idx`` MUST equal the drug's ORIG_INDEX in ``df_ref``
    (both ``feature_matrix`` and ``df_ref`` are ORIG_INDEX-indexed). The
    bounds check below turns a broken row_idx assignment into a hard error
    rather than a silent feature/label mismatch.
    """
    ion_map = {t: i for i, t in enumerate(ION_TYPES)}
    for data in pyg_list:
        idx = int(data.row_idx.item())
        if not (0 <= idx < feature_matrix.shape[0]) or idx not in df_ref.index:
            raise IndexError(
                f"row_idx {idx} is not a valid ORIG_INDEX in df_ref / feature matrix; "
                "the row_idx<->ORIG_INDEX contract is violated.")
        data.chemberta_feat = torch.tensor(feature_matrix[idx], dtype=torch.float32).view(1, -1)
        ion_str = str(df_ref.loc[idx, "IonType"]).strip().lower()
        data.ion_idx = torch.tensor([ion_map.get(ion_str, ion_map["neutral"])], dtype=torch.long)
    return pyg_list
