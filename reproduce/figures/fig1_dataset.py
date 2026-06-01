"""Figure 1 (Panel A) -- dataset characterisation diagnostics.

Four sub-panels:
  (1) Example generic Bemis-Murcko scaffolds + representative drugs (CV vs Test).
  (2) Tanimoto similarity of each test compound to its nearest CV neighbour.
  (3) Physicochemical space (LogP vs molecular weight), CV vs Test.
  (4) KDE distributions of the seven transformed PK targets, CV vs Test.

(Panels B and C of manuscript Figure 1 -- the data-partition and model-overview
schematics -- are drawn externally and are not reproduced here.)
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import FormatStrFormatter
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold

from .. import config as C
from .. import data as D

RDLogger.DisableLog("rdApp.*")


def _generic_scaffold(smiles):
    mol = Chem.MolFromSmiles(str(smiles))
    if mol is None:
        return None
    try:
        scaff = MurckoScaffold.MakeScaffoldGeneric(MurckoScaffold.GetScaffoldForMol(mol))
        return Chem.MolToSmiles(scaff)
    except Exception:
        return None


def _scaffold_examples(df_subset, priority_names, label_prefix, count=2):
    """Pick `count` generic scaffolds (prioritising named drugs) + 2 example drugs each."""
    out, seen = [], set()
    for name in priority_names:
        match = df_subset[df_subset["Name"].str.contains(name, case=False, na=False)]
        if match.empty:
            continue
        scaff_smi = _generic_scaffold(match.iloc[0]["SMILES"])
        if not scaff_smi or scaff_smi in seen:
            continue
        members = df_subset[df_subset["SMILES"].apply(_generic_scaffold) == scaff_smi]
        ex_mols = [Chem.MolFromSmiles(match.iloc[0]["SMILES"])]
        ex_names = [name.upper()]
        others = members[~members["Name"].str.contains(name, case=False, na=False)]
        if not others.empty:
            ex_mols.append(Chem.MolFromSmiles(others.iloc[0]["SMILES"]))
            ex_names.append(str(others.iloc[0]["Name"]).upper())
        out.append((Chem.MolFromSmiles(scaff_smi), f"{label_prefix} (N={len(members)})", ex_mols, ex_names))
        seen.add(scaff_smi)
        if len(out) == count:
            return out
    # fill remaining slots with most common frameworks
    tmp = df_subset.copy()
    tmp["frame"] = tmp["SMILES"].apply(_generic_scaffold)
    for frame_smi in tmp["frame"].value_counts().index:
        if not frame_smi or frame_smi in seen:
            continue
        members = tmp[tmp["frame"] == frame_smi]
        ex_mols = [Chem.MolFromSmiles(s) for s in members["SMILES"][:2]]
        ex_names = [str(n).upper() for n in members["Name"][:2].tolist()]
        out.append((Chem.MolFromSmiles(frame_smi), f"{label_prefix} (N={len(members)})", ex_mols, ex_names))
        seen.add(frame_smi)
        if len(out) == count:
            break
    return out


def make(cv_example_drugs=("Lidocaine", "Methamphetamine"),
         test_example_drugs=("Venlafaxine", "Selumetinib")):
    """Render and save Figure 1 panel A.

    ``cv_example_drugs`` / ``test_example_drugs`` are the drugs highlighted in the
    scaffold panel. The defaults are the manuscript's choices for the seed-42
    scaffold split; if a drug is absent from a partition (e.g. after a different
    split), ``_scaffold_examples`` falls back to the most common frameworks.
    """
    df_ref = D.build_reference()
    df, dev, test = D.partition_labels(df_ref)
    df_dev, df_test = df.loc[dev], df.loc[test]
    cd, ct = C.DEV_COLOR, C.TEST_COLOR

    cv_scaffs = _scaffold_examples(df_dev, list(cv_example_drugs), "Nested CV", 2)
    test_scaffs = _scaffold_examples(df_test, list(test_example_drugs), "Test Set", 2)

    # Tanimoto: each test compound vs its nearest CV neighbour
    fps_dev = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in df_dev["SMILES"]]
    fps_test = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in df_test["SMILES"]]
    tanimoto = [max(DataStructs.BulkTanimotoSimilarity(fp, fps_dev)) for fp in fps_test]

    C.set_pub_style()
    fig = plt.figure(figsize=(24, 9), dpi=300, facecolor="white")
    title_size = 20
    gs_master = GridSpec(2, 1, height_ratios=[1.2, 1.0], hspace=0.45)

    # --- (1) scaffolds ---
    gs_a = gs_master[0].subgridspec(1, 4, wspace=0.25)
    fig.text(0.5, 0.96, "(1) Example of scaffolds and corresponding drugs",
             ha="center", fontweight="bold", fontsize=title_size)
    for i, (scaff_mol, label, ex_mols, ex_names) in enumerate(cv_scaffs + test_scaffs):
        ax = fig.add_subplot(gs_a[i]); ax.axis("off")
        is_cv = i < 2
        ax.add_patch(plt.Rectangle((-0.05, -0.05), 1.1, 1.20, linewidth=2.0, fill=False,
                                   edgecolor=cd if is_cv else ct, linestyle="--",
                                   transform=ax.transAxes, clip_on=False))
        ax.text(0.5, 1.00, label, ha="center", va="bottom", fontsize=14,
                fontweight="bold", transform=ax.transAxes)
        gs_box = gs_a[i].subgridspec(2, 1, height_ratios=[2.2, 1], hspace=0.05)
        ax_s = fig.add_subplot(gs_box[0]); ax_s.axis("off")
        if scaff_mol is not None:
            ax_s.imshow(Draw.MolToImage(scaff_mol, size=(600, 600), fitImage=True))
        gs_ex = gs_box[1].subgridspec(1, 2, wspace=0.05)
        for j in range(2):
            if j < len(ex_mols) and ex_mols[j] is not None:
                ax_ex = fig.add_subplot(gs_ex[j]); ax_ex.axis("off")
                ax_ex.imshow(Draw.MolToImage(ex_mols[j], size=(400, 400), fitImage=True))
                ax_ex.set_title(ex_names[j], fontsize=10, pad=1)

    # --- bottom row ---
    gs_bottom = gs_master[1].subgridspec(1, 3, width_ratios=[0.8, 0.8, 1.8], wspace=0.3)
    ty = 0.44
    fig.text(0.19, ty, "(2) Tanimoto Similarity", fontweight="bold", fontsize=title_size, ha="center")
    fig.text(0.42, ty, "(3) Physicochemical Space", fontweight="bold", fontsize=title_size, ha="center")
    fig.text(0.73, ty, "(4) PK Target Distributions", fontweight="bold", fontsize=title_size, ha="center")

    # (2) Tanimoto
    ax_b = fig.add_subplot(gs_bottom[0])
    sns.kdeplot(tanimoto, fill=True, color=ct, alpha=0.4, linewidth=2, ax=ax_b)
    ax_b.axvline(0.4, color="#333333", linestyle=":", lw=1.0, alpha=0.8)
    ax_b.text(0.42, ax_b.get_ylim()[1] * 0.8, "Distinct\nScaffolds", fontsize=10,
              fontstyle="italic", color="#333333")
    ax_b.axvline(0.8, color="#000000", linestyle="--", lw=1.0, alpha=0.8)
    ax_b.set_xlabel("Max Tanimoto to Nested Cross-CV Set", fontsize=14, fontweight="bold")
    ax_b.set_ylabel("Density", fontsize=14, fontweight="bold")
    ax_b.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)

    # (3) physicochemical space
    ax_c = fig.add_subplot(gs_bottom[1])
    ax_c.scatter(df_dev["MW"], df_dev["LogP"], alpha=0.15, color=cd, s=15, label="Nested CV")
    ax_c.scatter(df_test["MW"], df_test["LogP"], alpha=0.7, color=ct, s=35,
                 edgecolors="white", lw=0.5, label="Independent Test")
    ax_c.axvline(500, color="#000000", linestyle="--", lw=1.2, alpha=0.7)
    ax_c.axhline(5, color="#000000", linestyle="--", lw=1.2, alpha=0.7)
    ax_c.set_xlabel("Molecular Weight", fontsize=14, fontweight="bold")
    ax_c.set_ylabel("LogP", fontsize=14, fontweight="bold")
    ax_c.legend(loc="lower right", fontsize=9, frameon=True, facecolor="white", framealpha=0.8)
    ax_c.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)

    # (4) PK target distributions (transformed space)
    gs_d = gs_bottom[2].subgridspec(2, 4, hspace=0.70, wspace=0.45)
    titles = [r"$t_{1/2}$", r"$V_d$", r"$CL$", r"$F$", r"$PPB$", r"$C_{max}$", r"$T_{max}$"]
    for i, (col, title) in enumerate(zip(C.TRANSFORMED_COLS, titles)):
        ax = fig.add_subplot(gs_d[i // 4, i % 4])
        d_vals = pd.to_numeric(df_dev[col], errors="coerce").dropna()
        t_vals = pd.to_numeric(df_test[col], errors="coerce").dropna()
        if len(d_vals):
            sns.kdeplot(d_vals, fill=True, color=cd, alpha=0.3, ax=ax, lw=1.0)
        if len(t_vals):
            sns.kdeplot(t_vals, fill=True, color=ct, alpha=0.3, ax=ax, lw=1.0)
        ax.set_title(title, fontweight="bold", fontsize=12)
        ax.set_xlabel(r"$\log_{10}$ Values", fontsize=10, fontweight="bold", labelpad=4)
        ax.set_ylabel("Density", fontsize=10, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.5, linewidth=0.5)
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

    ax_leg = fig.add_subplot(gs_d[1, 3]); ax_leg.axis("off")
    ax_leg.legend(handles=[plt.Rectangle((0, 0), 1, 1, color=cd, alpha=0.5, label="Nested CV"),
                           plt.Rectangle((0, 0), 1, 1, color=ct, alpha=0.5, label="Independent Test")],
                  loc="center left", fontsize=10, frameon=False)

    base = C.FIG_DIR / "figure1_dataset_diagnostics"
    fig.savefig(base.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(base.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)
    print(f"[fig] saved {base.with_suffix('.png').name} / .pdf  "
          f"(median Tanimoto={np.median(tanimoto):.3f})")


if __name__ == "__main__":
    make()
