"""
Supplementary table figure (journal quality, 300 DPI): scVIDR HF-diet prediction
R2 (mean +/- SD over the 5-seed run) for every taPVAT cell type, per timeframe
(8W/24W) x sex (Male / Female / Both), for both metrics (all HVGs and top-100
diet DEGs). Includes the total held-out HF nucleus count per cell type.

Reads:  results/r2_results.csv, results/target_manifest.csv
Writes: results/figures/r2_supplementary_table.{png,pdf}
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

plt.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42})

RESULTS = Path(__file__).resolve().parents[3] / "code/scVIDR/results"
OUTDIR = RESULTS / "figures"
OUTDIR.mkdir(parents=True, exist_ok=True)

d = pd.read_csv(RESULTS / "r2_results.csv")
man = pd.read_csv(RESULTS / "target_manifest.csv")
man["total"] = man["control_count"] + man["treated_count"]
both = man[man["sex_subset"] == "Both"]
src = both if len(both) else man[man["sex_subset"].isin(["M", "F"])]
abund = src.groupby("target")["total"].sum().sort_values(ascending=False)
cell_order = list(abund.index)

# total held-out HF nuclei per cell type (summed over the single-sex strata)
nhf = (d[d["sex"].isin(["M", "F"])].groupby("cell_type")["n_actual_hf_cells"].sum())

LABELS = {
    "Adipocytes_Brown": "Brown adipocytes",
    "ECs_Cap": "Capillary endothelial cells",
    "Fibroblasts_Bmper+_Nrxn1+": "Bmper$^{+}$Nrxn1$^{+}$ fibroblasts",
    "Adipocytes_3": "White adipocytes (Adipocytes_3)",
    "Pericytes": "Pericytes",
}
def disp(ct):
    return LABELS.get(ct, ct.replace("_", " "))

# stratum groups: 8W [M,F,Both], 24W [M,F,Both]
AGES = [("8W", [("M", "Male"), ("F", "Female"), ("Both", "Both")]),
        ("24W", [("M", "Male"), ("F", "Female"), ("Both", "Both")])]
groups = [(age, sex, sl) for age, sexes in AGES for sex, sl in sexes]

def val(age, sex, ct, metric):
    r = d[(d["age"] == age) & (d["sex"] == sex) & (d["cell_type"] == ct)]
    if not len(r):
        return "—"
    r = r.iloc[0]
    m, s = r[f"r2_{metric}_mean"], r[f"r2_{metric}_sd"]
    if pd.isna(m):
        return "—"
    return f"{m:.2f}±{s:.2f}"

# ---- geometry ----
name_w, n_w, sub_w = 3.3, 1.25, 1.02
n_sub = len(groups) * 2
val_x0 = name_w + n_w
W = val_x0 + n_sub * sub_w
h_age, h_sex, h_met, rh = 0.55, 0.55, 0.50, 0.50
header_h = h_age + h_sex + h_met
nrow = len(cell_order)
H = header_h + nrow * rh

C_AGE, C_SEX = "#33414A", "#4A5B64"        # neutral slate structural headers
C_HEADTX = "white"
C_HVG, C_DEG = "#1F7A8C", "#D9774B"        # teal + terracotta (match the bar figure)
C_HVG_L, C_DEG_L = "#D9E9EC", "#F7E3D9"    # light teal / light terracotta row tints
C_NAME, C_NALT, C_GRID = "#EAEEF0", "#F1F4F5", "#C7CFD4"

fig, ax = plt.subplots(figsize=(W * 1.30, H * 0.60 + 1.3))
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.invert_yaxis()
ax.axis("off")

def rect(x, y, w, h, fc, ec=C_GRID, lw=0.8, z=1):
    ax.add_patch(Rectangle((x, y), w, h, facecolor=fc, edgecolor=ec, linewidth=lw, zorder=z))

def subx(j):
    return val_x0 + j * sub_w

# left header columns (span all 3 header rows)
rect(0, 0, name_w, header_h, C_SEX, z=2)
ax.text(0.14, header_h / 2, "Cell type", color=C_HEADTX, fontsize=12.5,
        fontweight="bold", va="center", ha="left", zorder=3)
rect(name_w, 0, n_w, header_h, C_SEX, z=2)
ax.text(name_w + n_w / 2, header_h / 2, "Held-out\nHF (n)", color=C_HEADTX,
        fontsize=11, fontweight="bold", va="center", ha="center", zorder=3)

# age super-header
for ai, (age, sexes) in enumerate(AGES):
    x = subx(ai * 6)
    rect(x, 0, 6 * sub_w, h_age, C_AGE, z=2)
    ax.text(x + 3 * sub_w, h_age / 2, age, color=C_HEADTX, fontsize=13.5,
            fontweight="bold", va="center", ha="center", zorder=3)
# sex + metric headers
for g, (age, sex, sl) in enumerate(groups):
    x = subx(2 * g)
    rect(x, h_age, 2 * sub_w, h_sex, C_SEX, z=2)
    ax.text(x + sub_w, h_age + h_sex / 2, sl, color=C_HEADTX, fontsize=12,
            fontweight="bold", va="center", ha="center", zorder=3)
    rect(subx(2 * g), h_age + h_sex, sub_w, h_met, C_HVG, z=2)
    rect(subx(2 * g + 1), h_age + h_sex, sub_w, h_met, C_DEG, z=2)
    ax.text(subx(2 * g) + sub_w / 2, h_age + h_sex + h_met / 2, "HVGs", color="white",
            fontsize=9.5, fontweight="bold", va="center", ha="center", zorder=3)
    ax.text(subx(2 * g + 1) + sub_w / 2, h_age + h_sex + h_met / 2, "DEGs", color="white",
            fontsize=9.5, fontweight="bold", va="center", ha="center", zorder=3)

# data rows
for i, ct in enumerate(cell_order):
    y = header_h + i * rh
    rect(0, y, name_w, rh, C_NAME, z=1)
    ax.text(0.14, y + rh / 2, disp(ct), fontsize=10, fontweight="bold",
            va="center", ha="left", zorder=3)
    n = int(nhf.get(ct, 0))
    rect(name_w, y, n_w, rh, C_NALT if i % 2 else "white", z=1)
    ax.text(name_w + n_w / 2, y + rh / 2, f"{n:,}" if n else "—", fontsize=9.3,
            va="center", ha="center", zorder=3)
    for g, (age, sex, sl) in enumerate(groups):
        for k, metric in enumerate(["all_hvgs", "top100_degs"]):
            x = subx(2 * g + k)
            tint = (C_HVG_L if k == 0 else C_DEG_L) if i % 2 else "white"
            rect(x, y, sub_w, rh, tint, z=1)
            ax.text(x + sub_w / 2, y + rh / 2, val(age, sex, ct, metric),
                    fontsize=8.7, va="center", ha="center", zorder=3)

fig.suptitle("scVIDR HF-diet prediction accuracy — R$^{2}$ (mean ± SD, 5 seeds)",
             fontsize=16, fontweight="bold", y=0.978)
fig.text(0.5, 0.930, "All highly variable genes (HVGs) and top-100 diet DEGs per "
         "stratum; cell types ordered by total abundance. Held-out HF (n) = total "
         "high-fat nuclei withheld per cell type.",
         fontsize=10, ha="center", style="italic", color="#333333")
fig.subplots_adjust(left=0.010, right=0.990, top=0.90, bottom=0.012)

png = OUTDIR / "r2_supplementary_table.png"
pdf = OUTDIR / "r2_supplementary_table.pdf"
fig.savefig(png, dpi=300, bbox_inches="tight")
fig.savefig(pdf, bbox_inches="tight")
print(f"wrote (300 dpi): {png}\n                 {pdf}\nrows: {nrow}  groups: {len(groups)}")
