from typing import Optional, Tuple

import torch
import numpy as np
import pandas as pd
import re
from rdkit import Chem
from rdkit.Chem.Scaffolds import MurckoScaffold
from sklearn.model_selection import GroupShuffleSplit

# --- 1. Chemistry Utilities ---

def get_generic_scaffold(smi: str) -> Optional[str]:
    """Generates a generic (all atoms to C, all bonds to single) Bemis-Murcko scaffold."""
    if pd.isna(smi): return None
    mol = Chem.MolFromSmiles(str(smi))
    if not mol: return None
    scaf = MurckoScaffold.GetScaffoldForMol(mol)
    scaf = MurckoScaffold.MakeScaffoldGeneric(scaf)
    return Chem.MolToSmiles(scaf, isomericSmiles=False)

def norm_name(s):
    """Normalizes drug names by removing punctuation/spaces and upper-casing."""
    return re.sub(r"[^A-Z0-9]", "", str(s).upper()) if pd.notna(s) else None

# --- 2. Dataset Utilities ---

def scaffold_split(dataset, groups, test_size=0.2, seed=42):
    """Performs a scaffold-aware split to ensure structural independence."""
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    dev_idx, test_idx = next(gss.split(np.arange(len(dataset)), groups=groups))
    return [dataset[i] for i in dev_idx], [dataset[i] for i in test_idx]

def extract_targets(row) -> Tuple[torch.Tensor, torch.Tensor]:
    """Extract the 7 transformed PK targets and a visibility mask from a row.

    Returns ``(y, mask)`` where ``y`` is a length-7 float tensor in log10/logit10
    space and ``mask`` flags which targets are observed.
    """
    cols = ["log_t1/2(hour)", "log_VD(liter)", "log_Cl(liter/hour)",
            "logit_F(percentage)", "logit_PPB(percentage)", "log_Cmax_uM", "log_Tmax"]
    vals = [row.get(c) for c in cols]
    mask = [pd.notna(v) for v in vals]
    # Missing targets stored as 0.0 (placeholder); excluded from the loss via the mask.
    y = torch.tensor([float(v) if m else 0.0 for v, m in zip(vals, mask)], dtype=torch.float)
    return y, torch.tensor(mask, dtype=torch.bool)

# --- 3. Normalization Logic ---

class TaskScaler:
    """Computes and stores Z-score statistics while ignoring masked (missing) data."""
    def __init__(self, means=None, stds=None):
        self.means = torch.tensor(means) if means is not None else None
        self.stds  = torch.tensor(stds) if stds is not None else None

    @torch.no_grad()
    def fit_from_loader(self, loader, device):
        """Compute per-target mean/std from a PyG DataLoader, ignoring masked values."""
        S, SS, N = torch.zeros(7, device=device), torch.zeros(7, device=device), torch.zeros(7, device=device)
        for batch in loader:
            y, m = batch.y.to(device).view(-1, 7), batch.y_mask.to(device).view(-1, 7)
            S  += (y * m).sum(0)
            SS += (y.pow(2) * m).sum(0)
            N  += m.sum(0)

        self.means = (S / N.clamp_min(1)).cpu()
        self.stds  = torch.sqrt((SS / N.clamp_min(1)) - self.means.to(device).pow(2)).clamp_min(1e-6).cpu()
        return self

    def to(self, device):
        if self.means is not None: self.means = self.means.to(device)
        if self.stds is not None: self.stds = self.stds.to(device)
        return self