"""PIPK-Net quick-start: predict systemic PK parameters for small molecules.

Run from anywhere inside the repository:

    python examples/quickstart.py

It demonstrates the high-level :class:`pipknet.PIPKNetPredictor` API for
single-molecule and batch prediction (each compound predicted at its own
ionisation state) and the mass-balance consistency of the physiology-informed
(C_physio) variant (PIPK-Net).
"""
from pathlib import Path

import numpy as np
import pandas as pd

from pipknet import PIPKNetPredictor

# ---------------------------------------------------------------------------
# Locate the repository root (the folder containing checkpoints/ and example_drugs.csv)
# ---------------------------------------------------------------------------
REPO = Path(__file__).resolve().parent.parent
CKPT = REPO / "checkpoints" / "C_physio"   # the recommended physiology-informed variant


def main():
    pd.set_option("display.width", 120)

    # 1) Load the 5-fold ensemble once.
    predictor = PIPKNetPredictor(CKPT)
    print(f"Loaded PIPK-Net variant '{predictor.variant}' "
          f"({len(predictor.models)} folds) on {predictor.device}.\n")

    # 2) Single-molecule prediction (mean +/- std across the 5 folds, physical units).
    #    Selumetinib is a MEK inhibitor; its ionisation state at pH 7.4 is neutral.
    #    IonType is a fixed molecular property (set by pKa) supplied as an input feature,
    #    not a tunable parameter -- always pass the compound's true class.
    selumetinib = "CN1C=NC2=C1C=C(C(=C2F)NC3=C(C=C(C=C3)Br)Cl)C(=O)NOCCO"
    print("Selumetinib (predicted as neutral):")
    print(predictor.predict(selumetinib, ion_type="neutral").round(2), "\n")

    # 3) Ionisation state is a fixed molecular property set by the compound's pKa at
    #    pH 7.4. Supply the correct class for your compound; do not vary it as a tunable
    #    parameter. To show the model on a different ionisation class, here is venlafaxine,
    #    a cationic drug with extensive tissue distribution (clinical Vd 308-525 L).
    venlafaxine = "COc1ccc(C(CN(C)C)C2(O)CCCCC2)cc1.Cl"   # hydrochloride salt, as stored in e-Drug3D
    print("Venlafaxine (cationic; held-out test compound):")
    r = predictor.predict(venlafaxine, ion_type="cationic").round(2)
    print(r)
    print(f"  -> predicted Vd {r.loc['Vd (L)', 'Mean']:.0f} L (clinical 308-525 L)\n")

    # 4) Batch prediction from a CSV (SMILES [, IonType, Name]); each drug is predicted
    #    at its own ionisation state.
    batch = predictor.predict_batch(REPO / "example_drugs.csv")
    cols = ["Name", "IonType", "Vd (L)_mean", "CL (L/h)_mean", "t_half (h)_mean", "F (%)_mean"]
    print("Batch predictions (example_drugs.csv):")
    print(batch[cols].round(1).to_string(index=False), "\n")

    # 5) Mass-balance consistency of the physiology-informed variant:
    #    t_half should be close to ln(2) * Vd / CL.
    vd, cl, th = batch["Vd (L)_mean"], batch["CL (L/h)_mean"], batch["t_half (h)_mean"]
    implied = np.log(2) * vd / cl
    fold_dev = 10 ** np.abs(np.log10(th / implied))
    print("Mass-balance check (median fold deviation of t_half from ln2*Vd/CL): "
          f"{np.median(fold_dev):.2f}x")


if __name__ == "__main__":
    main()
