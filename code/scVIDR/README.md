# taPVAT scVIDR Workflow

Held-out high-fat-diet (HF) response prediction for taPVAT single-nucleus
RNA-seq, using scVIDR. For each age/sex stratum, every annotated cell type is
held out in turn and its HF transcriptional state is predicted from its Control
state; predictions are scored by R² (over all 5,000 highly variable genes and
over the top-100 diet DEGs).

## Layout

```text
configs/                  Workflow configuration
hpcc/                     MSU ICER SLURM scripts + conda environment
results/                  R2 result tables and the main figure
scripts/                  taPVAT workflow scripts
scVIDR_code_from_github/  Vendored scVIDR source (pinned snapshot from GitHub)
```

The vendored scVIDR source is pinned in
`scVIDR_code_from_github/UPSTREAM_COMMIT.md`.

## Method summary

- **Strata:** age (8W, 24W) × sex (M, F, and Both = M+F).
- **Targets:** every fine-grained `celltype` label is an independent held-out
  target. Doublets are excluded.
- **No abundance threshold:** a cell type is modeled whenever it has ≥1 cell in
  both Control and HF (the structural minimum for a Control→HF shift).
- **Per target:** an scVIDR VAE is trained on all cells except that cell type's
  HF cells; its HF state is then predicted via scVIDR's cross-cell-type latent
  regression.
- **Robustness:** 5 random seeds (0–4); results reported as mean ± SD.
- **Metrics:** R² between predicted and measured mean HF expression, over all
  HVGs and over the top-100 Control-vs-HF DEGs.

## Environment

```bash
conda env create -f code/scVIDR/hpcc/scVIDR_env_hpcc.yml   # creates scVIDR_env
conda activate scVIDR_env
```

## Running

The full workflow runs on MSU ICER via SLURM — see `hpcc/README_HPCC.md`. The
underlying steps (from the repository root, after `conda activate scVIDR_env`):

```bash
RUN=code/scVIDR/scripts/run_tapvat_scvidr.py
python $RUN list-targets                          # eligibility manifest
python $RUN prepare-inputs                         # per-stratum AnnData inputs
python $RUN run --seed 0 --skip-gene-scores        # train + predict (one seed)
```

Then summarize across seeds and build the tables and figure:

```bash
python code/scVIDR/scripts/summarize_seeds.py --age 24W --sex M --seeds 0,1,2,3,4
python code/scVIDR/scripts/aggregate_seeds.py        # single R2 results CSV (r2_results.csv)
python code/scVIDR/scripts/plot_r2_bar.py            # main figure (300 dpi)
python code/scVIDR/scripts/make_r2_table_figure.py   # supplementary table figure (300 dpi)
```

## Scripts

- `run_tapvat_scvidr.py` — list-targets / prepare-inputs / run (train + predict)
- `summarize_seeds.py` — per-seed R² (all HVGs + top-100 DEGs), mean ± SD
- `aggregate_seeds.py` — combine strata into the single R² results CSV (`r2_results.csv`)
- `plot_r2_bar.py` — main-figure bar plot, top-5 most numerous cell types
- `make_r2_table_figure.py` — supplementary table figure, all cell types × strata
