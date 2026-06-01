# pipknet/__init__.py
"""PIPK-Net: physiology-informed GNN for multitask pharmacokinetic prediction."""

from .models import GNNMultitask, TASKS, TASKS_WITH_UNITS
from .engine import (fit_one, evaluate, to_original_units,
                     loss_supervised, loss_with_t12_consistency)
from .utils import (TaskScaler, scaffold_split, get_generic_scaffold,
                    extract_targets, norm_name)
from .featurizers import smiles_to_pyg, ION_TYPES, ION_MAP
from .data import load_dataset_from_csv, build_dataset, apply_pk_transforms
from .training import nested_cv, train_from_csv
from .inference import PIPKNetPredictor

__version__ = "0.1.0"
__author__ = "Thao Pham"

__all__ = [
    "GNNMultitask", "TASKS", "TASKS_WITH_UNITS",
    "fit_one", "evaluate", "to_original_units",
    "loss_supervised", "loss_with_t12_consistency",
    "TaskScaler", "scaffold_split", "get_generic_scaffold", "extract_targets", "norm_name",
    "smiles_to_pyg", "ION_TYPES", "ION_MAP",
    "load_dataset_from_csv", "build_dataset", "apply_pk_transforms",
    "nested_cv", "train_from_csv",
    "PIPKNetPredictor",
]

