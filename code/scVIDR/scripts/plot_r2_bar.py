"""
Main-figure bar plot (journal quality, 300 DPI): scVIDR prediction R2 (all HVGs
and top-100 DEGs, mean +/- SD over the seed run) for the top-5 most numerous
taPVAT cell types, per age/sex stratum (8W/24W x M/F).

Inputs (tracked):
  results/target_manifest.csv   (rank cell types by abundance)
  results/r2_results.csv        (mean/SD R2 per cell type x stratum)
Output:
  results/figures/r2_top5_celltypes_bar.{png,pdf}
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update({
    "savefig.dpi": 300,
    "figure.dpi": 300,
    "font.size": 14,
    "font.weight": "bold",
    "axes.titlesize": 19,
    "axes.titleweight": "bold",
    "axes.labelsize": 17,
    "axes.labelweight": "bold",
    "xtick.labelsize": 14,
    "ytick.labelsize": 15,
    "legend.fontsize": 15,
    "axes.linewidth": 1.5,
    "pdf.fonttype": 42,   # editable text in Illustrator
    "ps.fonttype": 42,
})

RESULTS = Path(__file__).resolve().parents[3] / "code/scVIDR/results"
OUTDIR = RESULTS / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

# ---- top-5 most numerous cell types (total cells, from the manifest) ----
man = pd.read_csv(RESULTS / "target_manifest.csv")
man["total"] = man["control_count"] + man["treated_count"]
both = man[man["sex_subset"] == "Both"]
src = both if len(both) else man[man["sex_subset"].isin(["M", "F"])]
abund = src.groupby("target")["total"].sum().sort_values(ascending=False)
TOP5 = list(abund.head(5).index)
print("Top-5 most numerous cell types:")
for ct in TOP5:
    print(f"  {ct}: {int(abund[ct])} cells")

# ---- meaningful display names (edit here if you want different wording) ----
LABELS = {
    "Adipocytes_Brown": "Brown adipocytes",
    "ECs_Cap": "Capillary endothelial cells",
    "Fibroblasts_Bmper+_Nrxn1+": "Bmper$^{+}$Nrxn1$^{+}$ fibroblasts",
    "Adipocytes_3": "White adipocytes (Adipocytes_3)",
    "Pericytes": "Pericytes",
}

d = pd.read_csv(RESULTS / "r2_results.csv")
STRATA = [("8W", "M"), ("8W", "F"), ("24W", "M"), ("24W", "F")]
labels = [LABELS.get(ct, ct) for ct in TOP5]
x = np.arange(len(TOP5))
w = 0.40
C_ALL, C_DEG = "#1F7A8C", "#D9774B"   # vibrant but refined: teal + terracotta

fig, axes = plt.subplots(2, 2, figsize=(16, 12), sharey=True)
for ax, (age, sex) in zip(axes.flat, STRATA):
    sub = d[(d["age"] == age) & (d["sex"] == sex)].set_index("cell_type")
    def col(ct, k):
        return sub.loc[ct, k] if ct in sub.index else np.nan
    allm = [col(ct, "r2_all_hvgs_mean") for ct in TOP5]
    alls = [col(ct, "r2_all_hvgs_sd") if ct in sub.index else 0 for ct in TOP5]
    degm = [col(ct, "r2_top100_degs_mean") for ct in TOP5]
    degs = [col(ct, "r2_top100_degs_sd") if ct in sub.index else 0 for ct in TOP5]
    ekw = dict(capsize=5, capthick=2, elinewidth=2, ecolor="black")
    ax.bar(x - w / 2, allm, w, yerr=alls, error_kw=ekw, color=C_ALL,
           edgecolor="black", linewidth=1.2, label="All HVGs")
    ax.bar(x + w / 2, degm, w, yerr=degs, error_kw=ekw, color=C_DEG,
           edgecolor="black", linewidth=1.2, label="Top-100 DEGs")
    # overlay the 5 individual per-seed R2 values on each bar
    rng = np.random.default_rng(0)
    for xi, ct in enumerate(TOP5):
        for off, pre in ((-w / 2, "r2_all_seed"), (w / 2, "r2_deg_seed")):
            pts = np.array([col(ct, f"{pre}{s}") for s in range(5)], dtype=float)
            pts = pts[~np.isnan(pts)]
            jit = (rng.random(len(pts)) - 0.5) * (w * 0.55)
            ax.scatter(np.full(len(pts), x[xi] + off) + jit, pts, s=24,
                       facecolor="white", edgecolor="black", linewidth=1.0,
                       zorder=5, alpha=0.95)
    ax.set_title(f"{age}  |  {'Male' if sex=='M' else 'Female'}", pad=10)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right", rotation_mode="anchor")
    ax.set_ylim(0, 1.05)
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.tick_params(axis="both", width=1.5, length=6)
    ax.grid(axis="y", ls=":", alpha=0.45, linewidth=1.0)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)

for ax in axes[:, 0]:
    ax.set_ylabel("scVIDR prediction R$^{2}$")
fig.suptitle("HF-Diet Gene Expression Predictions (mean ± SD, 5 seeds)",
             fontsize=21, fontweight="bold", y=0.99)
# shared color key centered beneath the title (not overlapping the panels)
handles, hlabels = axes[0, 0].get_legend_handles_labels()
handles.append(Line2D([0], [0], marker="o", linestyle="None", markerfacecolor="white",
                      markeredgecolor="black", markersize=8, label="Individual seeds"))
hlabels.append("Individual seeds")
fig.legend(handles, hlabels, loc="upper center", ncol=3,
           bbox_to_anchor=(0.5, 0.95), frameon=True, edgecolor="black",
           framealpha=0.95, columnspacing=2.2, handlelength=1.8, borderpad=0.6)
fig.tight_layout(rect=[0, 0, 1, 0.91], h_pad=3.5)

png = OUTDIR / "r2_top5_celltypes_bar.png"
pdf = OUTDIR / "r2_top5_celltypes_bar.pdf"
fig.savefig(png, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"\nWrote (300 dpi): {png}\n                 {pdf}")
