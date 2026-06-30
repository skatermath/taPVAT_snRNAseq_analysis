# Running the scVIDR workflow on a SLURM HPC cluster

This pipeline trains and evaluates scVIDR per cell type across strata as a set of
SLURM jobs. It is written to run from a fast scratch filesystem on a cluster that
provides GPU nodes and a conda/Miniforge environment module. Adjust the paths,
host names, account, partitions, and module names below to match your cluster.

Set these to match your environment:

```bash
LOGIN=<user>@<login-host>     # SLURM submit/login host (file transfer + sbatch)
DEV=<user>@<build-node>       # a node with internet for building the conda env
ROOT=$SCRATCH/scvidr          # working dir on a large/scratch filesystem
LOCAL_REPO=/path/to/repo      # the cloned repository on your workstation
```

Layout created under `$ROOT`:

```text
$ROOT/
  code/scVIDR/          scripts, configs, scVIDR_code_from_github/, hpcc/ (this dir)
  data/                 the input annotated .h5ad (the only large file transferred)
  envs/scVIDR_env       conda prefix env (Linux + CUDA), built on a node with internet
  conda_pkgs/           conda package cache (kept off your home quota)
  code/scVIDR/results/  generated outputs (per-stratum inputs, models, predictions, R2)
```

Typical node roles: a **build node** with internet to create the conda env; a
**login/submit host** for file transfer and `sbatch`; **compute nodes** that only
*activate* the prebuilt env (no internet needed).

## 1. Transfer code + data

```bash
rsync -az --exclude 'results/' --exclude '__pycache__/' --exclude '.DS_Store' \
  "$LOCAL_REPO/code/scVIDR/" "$LOGIN:$ROOT/code/scVIDR/"
rsync -a "$LOCAL_REPO/data/"*.h5ad "$LOGIN:$ROOT/data/"
```

## 2. Build the conda env (once, on a node with internet)

```bash
ssh -J $LOGIN $DEV
module purge && module load Miniforge3       # or your cluster's conda/Miniforge module
export CONDA_PKGS_DIRS=$ROOT/conda_pkgs
conda env create -p $ROOT/envs/scVIDR_env -f $ROOT/code/scVIDR/hpcc/scVIDR_env_hpcc.yml
source "${EBROOTMINIFORGE3}/etc/profile.d/conda.sh"   # adjust to your module's conda.sh
conda activate $ROOT/envs/scVIDR_env
python -c "import scvi, torch, scanpy; print('scvi', scvi.__version__, '| torch', torch.__version__)"
```

## 3. Configure and submit

In `code/scVIDR/configs/tapvat_scvidr_targets.yml`, set `input_h5ad` and
`output_dir` and confirm the `obs_columns` mapping. Edit the `#SBATCH` headers in
`hpcc/*.sb` (account, partition, time, memory) and the `ROOT` path inside them for
your cluster, then submit:

```bash
bash $ROOT/code/scVIDR/hpcc/submit_seeds.sh
squeue -u $USER
```

The scripts chain via SLURM dependencies:
- `00_prepare_inputs` (CPU) — one AnnData per stratum
- `01_run_seeds` (GPU array, one task per stratum × seed) — train + predict every
  cell type; outputs under `results/<age>/<sex>/<cell_type>/seed<N>/`
- `02_summarize_seeds` (CPU) — `summarize_seeds.py` per stratum + `aggregate_seeds.py`

## 4. Results

Per stratum: `results/<age>/<sex>/performance_summary_seeds.csv` (mean/SD/per-seed).
Pull the combined table back to your workstation:

```bash
rsync -a "$LOGIN:$ROOT/code/scVIDR/results/r2_results.csv" "$LOCAL_REPO/code/scVIDR/results/"
```

- `r2_results.csv` — the single R² results table: every cell type × stratum,
  held-out HF n, and R² (all HVGs and top-100 DEGs) as mean ± SD plus numeric
  mean/SD and per-seed values.

## Notes

- The runner auto-selects CUDA on GPU nodes (falls back to CPU otherwise).
- Right-size the per-task `#SBATCH` resources to your data; targeting a
  short-queue partition lets the small array tasks backfill quickly.
- No cell-count threshold is applied; the only automatic exclusion is a cell type
  with an empty Control or treated group in a given stratum (recorded in
  `results/target_manifest.csv`).
- Scratch filesystems are often purged after a period of inactivity — copy the
  final CSVs (and any models you want to keep) to persistent storage.
