"""
Compute per-seed R2 and seed-aggregated (mean +/- SD) performance for a
completed multi-seed scVIDR age/sex run.

Layout expected (written by run_tapvat_scvidr.py run --seed N):
  results/<age>/<sex>/<cell_type>/seed<N>/predictions/HF_PRED.h5ad

For each cell type:
  - the actual HF cells and the top-100 diet DEGs are fixed (real data)
  - each seed's prediction is scored on both gene sets (all HVGs, top-100 DEGs)
  - we report mean and SD across seeds, plus the per-seed values

Usage:
  python code/scVIDR/scripts/summarize_seeds.py --age 24W --sex M
  python code/scVIDR/scripts/summarize_seeds.py --age 24W --sex M --seeds 0,1,2,3,4

Output:
  results/<age>/<sex>/performance_summary_seeds.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy import sparse, stats

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "code/scVIDR/results"

parser = argparse.ArgumentParser()
parser.add_argument("--age", required=True)
parser.add_argument("--sex", required=True)
parser.add_argument("--seeds", default="0,1,2,3,4", help="comma-separated seeds")
parser.add_argument("--results-dir", default=None)
args = parser.parse_args()

seeds = [int(s) for s in args.seeds.split(",") if s.strip() != ""]
results_dir = Path(args.results_dir) if args.results_dir else RESULTS_ROOT / args.age / args.sex
input_h5ad = results_dir / "inputs" / f"{args.age}_{args.sex}_scvidr_targets.h5ad"
if not input_h5ad.exists():
    raise FileNotFoundError(f"Input h5ad not found: {input_h5ad}")

print(f"Loading {input_h5ad} ...")
adata = sc.read_h5ad(input_h5ad)
sc.pp.normalize_total(adata)
sc.pp.log1p(adata)
sc.pp.highly_variable_genes(adata, n_top_genes=5000)
adata_hvg = adata[:, adata.var.highly_variable].copy()
if sparse.issparse(adata_hvg.X):
    adata_hvg.X = adata_hvg.X.toarray()
print(f"Normalized adata: {adata_hvg.shape[0]} cells x {adata_hvg.shape[1]} HVGs")


def r2_between(actual_sub, pred_adata, genes):
    a = np.array(actual_sub[:, genes].X).mean(axis=0)
    p = np.array(pred_adata[:, genes].X).mean(axis=0)
    _, _, r, _, _ = stats.linregress(a, p)
    r2 = r ** 2
    return r2 if np.isfinite(r2) else np.nan


def top_degs(target, common_set):
    """Top-100 Control-vs-HF DEGs among the real cells (matches reg_mean_plot)."""
    sub = adata_hvg[
        (adata_hvg.obs["celltype"] == target)
        & (adata_hvg.obs["diet"].isin(["Control", "HF"]))
    ].copy()
    if sub.obs["diet"].nunique() != 2:
        return []
    try:
        sc.tl.rank_genes_groups(sub, groupby="diet", method="wilcoxon")
        names = list(sub.uns["rank_genes_groups"]["names"]["HF"])
        return [g for g in names[:100] if g in common_set]
    except Exception as exc:
        print(f"    note: DEG computation failed for {target}: {exc}")
        return []


SKIP_DIRS = {"inputs"}
rows = []
for cell_type_dir in sorted(results_dir.iterdir()):
    if not cell_type_dir.is_dir() or cell_type_dir.name in SKIP_DIRS:
        continue
    target = cell_type_dir.name

    actual_mask = (adata_hvg.obs["celltype"] == target) & (adata_hvg.obs["diet"] == "HF")
    n_actual = int(actual_mask.sum())
    if n_actual == 0:
        continue
    actual_sub = adata_hvg[actual_mask]

    per_all, per_deg = {}, {}
    top_genes = None
    for seed in seeds:
        pred_path = cell_type_dir / f"seed{seed}" / "predictions" / "HF_PRED.h5ad"
        if not pred_path.exists():
            continue
        pred = sc.read_h5ad(pred_path)
        if pred.shape[0] == 0:
            continue
        if sparse.issparse(pred.X):
            pred.X = pred.X.toarray()
        common = actual_sub.var_names.intersection(pred.var_names)
        if len(common) == 0:
            continue
        r2_all = r2_between(actual_sub, pred, common)
        if not np.isfinite(r2_all):
            continue
        per_all[seed] = round(float(r2_all), 4)

        if top_genes is None:
            top_genes = top_degs(target, set(common))
        if top_genes and len(top_genes) >= 2:
            r2d = r2_between(actual_sub, pred, top_genes)
            if np.isfinite(r2d):
                per_deg[seed] = round(float(r2d), 4)

    if not per_all:
        print(f"  SKIP {target}: no seed predictions found")
        continue

    all_vals = np.array(list(per_all.values()), dtype=float)
    deg_vals = np.array(list(per_deg.values()), dtype=float)

    def _mean(v):
        return round(float(np.mean(v)), 4) if v.size else np.nan

    def _sd(v):
        return round(float(np.std(v, ddof=1)), 4) if v.size > 1 else (0.0 if v.size == 1 else np.nan)

    row = {
        "age": args.age,
        "sex": args.sex,
        "cell_type": target,
        "r2_all_hvgs_mean": _mean(all_vals),
        "r2_all_hvgs_sd": _sd(all_vals),
        "r2_top100_degs_mean": _mean(deg_vals),
        "r2_top100_degs_sd": _sd(deg_vals),
        "n_seeds": int(all_vals.size),
        "n_actual_hf_cells": n_actual,
        "n_top100_degs": len(top_genes) if top_genes else 0,
    }
    for seed in seeds:
        row[f"r2_all_seed{seed}"] = per_all.get(seed, np.nan)
        row[f"r2_deg_seed{seed}"] = per_deg.get(seed, np.nan)
    rows.append(row)
    print(f"  {target}: all={row['r2_all_hvgs_mean']}+/-{row['r2_all_hvgs_sd']} "
          f"deg={row['r2_top100_degs_mean']}+/-{row['r2_top100_degs_sd']} (n={row['n_seeds']})")

if not rows:
    print("No completed seed predictions found.")
else:
    df = pd.DataFrame(rows).sort_values("r2_all_hvgs_mean", ascending=False).reset_index(drop=True)
    out_path = results_dir / "performance_summary_seeds.csv"
    df.to_csv(out_path, index=False)
    print(f"\nSaved -> {out_path}")
