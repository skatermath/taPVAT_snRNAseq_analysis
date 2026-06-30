"""Aggregate the per-stratum multi-seed scVIDR summaries into ONE results table.

Reads results/<age>/<sex>/performance_summary_seeds.csv for all six strata
(8W/24W x M/F/Both) and writes the single canonical results file:

  results/r2_results.csv   one row per cell type x stratum, with the held-out
                           HF nucleus count and R2 (all HVGs and top-100 diet
                           DEGs) as both "mean +/- SD" and numeric mean/SD
                           columns, plus the per-seed R2 values.

Usage:
  python code/scVIDR/scripts/aggregate_seeds.py
"""
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS = REPO_ROOT / "code/scVIDR/results"
AGES = ["8W", "24W"]
SEXES = ["M", "F", "Both"]

frames = []
for age in AGES:
    for sex in SEXES:
        f = RESULTS / age / sex / "performance_summary_seeds.csv"
        if not f.exists():
            print(f"  missing: {f.relative_to(REPO_ROOT)}")
            continue
        df = pd.read_csv(f)
        if not df.empty:
            frames.append(df)

if not frames:
    raise SystemExit("No performance_summary_seeds.csv files found yet.")

long = pd.concat(frames, ignore_index=True)
long["age"] = pd.Categorical(long["age"], AGES, ordered=True)
long["sex"] = pd.Categorical(long["sex"], SEXES, ordered=True)
long = long.sort_values(["age", "sex", "r2_all_hvgs_mean"], ascending=[True, True, False])


def fmt(mean, sd):
    if pd.isna(mean):
        return ""
    return f"{mean:.3f} ± {sd:.3f}" if not pd.isna(sd) else f"{mean:.3f}"


long["R2_all_HVGs"] = [fmt(m, s) for m, s in zip(long["r2_all_hvgs_mean"], long["r2_all_hvgs_sd"])]
long["R2_top100_DEGs"] = [fmt(m, s) for m, s in zip(long["r2_top100_degs_mean"], long["r2_top100_degs_sd"])]

seed_cols = [c for c in long.columns if c.startswith("r2_all_seed") or c.startswith("r2_deg_seed")]
cols = ["age", "sex", "cell_type", "n_actual_hf_cells", "n_top100_degs", "n_seeds",
        "R2_all_HVGs", "R2_top100_DEGs",
        "r2_all_hvgs_mean", "r2_all_hvgs_sd", "r2_top100_degs_mean", "r2_top100_degs_sd"] + seed_cols
cols = [c for c in cols if c in long.columns]
long[cols].to_csv(RESULTS / "r2_results.csv", index=False)

print(f"Strata: {long.groupby(['age', 'sex'], observed=True).ngroups}   rows: {len(long)}")
print("Wrote: results/r2_results.csv")
