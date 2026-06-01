import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GraphConv, global_mean_pool

TASKS = ["t_half", "Vd", "CL", "F", "PPB", "Cmax", "Tmax"]
TASKS_WITH_UNITS = ["t_half (h)", "Vd (L)", "CL (L/h)", "F (%)", "PPB (%)",
                    "Cmax (uM)", "Tmax (h)"]


class GNNMultitask(nn.Module):
    """Multi-task GraphConv regressor for the seven PK endpoints.

    Two GraphConv layers + global mean pooling produce a structural embedding;
    when ``use_ion`` is set, a learnable ionisation-state embedding is
    concatenated before a two-layer MLP head. The physiology-informed
    mass-balance constraint is applied in the loss (see ``engine.py``), not in
    the architecture, so no extra parameters are needed here.

    Args:
        hidden_dim: width of the GraphConv layers and MLP hidden layer.
        ion_dim: number of ionisation classes (anionic/cationic/neutral/zwitterionic).
        ion_emb_dim: dimensionality of the learnable ionisation embedding.
        out_dim: number of regression targets (7).
        use_ion: whether to concatenate the ionisation embedding (variants B/C).
        dropout: dropout probability applied after pooling and the first MLP layer.
    """

    def __init__(self, hidden_dim: int = 128, ion_dim: int = 4, ion_emb_dim: int = 16,
                 out_dim: int = len(TASKS), use_ion: bool = True, dropout: float = 0.1):
        super().__init__()
        self.use_ion = use_ion
        self.conv1 = GraphConv(-1, hidden_dim)   # auto infer input dim
        self.conv2 = GraphConv(hidden_dim, hidden_dim)
        self.dropout = nn.Dropout(dropout)

        self.ion_embedding = nn.Embedding(ion_dim, ion_emb_dim) if use_ion else None
        fc_in = hidden_dim + (ion_emb_dim if use_ion else 0)

        self.fc1 = nn.Linear(fc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, out_dim)

    def forward(self, data: Data) -> torch.Tensor:
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x = F.relu(self.conv2(x, edge_index))
        x = global_mean_pool(x, batch)
        x = self.dropout(x)

        if self.use_ion and hasattr(data, "ion_feat"):
            ion = data.ion_feat
            if ion.dim()==2: 
                ion = ion.view(-1)
            x = torch.cat([x, self.ion_embedding(ion)], dim=1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)  # (B, 7)