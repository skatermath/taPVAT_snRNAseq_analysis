#!/bin/bash
# Submit the full scVIDR workflow on ICER with SLURM dependencies:
#   prepare-inputs  ->  30-task GPU seed array (6 strata x seeds 0-4)  ->  summarize
# Run from anywhere on the cluster:
#   bash /mnt/scratch/bowmand8/taPVAT_scVIDR/code/scVIDR/hpcc/submit_seeds.sh
set -euo pipefail
HPCC=/mnt/scratch/bowmand8/taPVAT_scVIDR/code/scVIDR/hpcc
mkdir -p "$HPCC/logs"

PREP=$(sbatch --parsable "$HPCC/00_prepare_inputs.sb")
echo "prepare-inputs  : job $PREP"

ARR=$(sbatch --parsable --dependency=afterok:"$PREP" "$HPCC/01_run_seeds.sb")
echo "seed array (30) : job $ARR  (depends on $PREP)"

SUM=$(sbatch --parsable --dependency=afterok:"$ARR" "$HPCC/02_summarize_seeds.sb")
echo "seed summarize  : job $SUM  (depends on $ARR)"

echo
echo "Monitor:  squeue -u \$USER"
echo "Outputs:  /mnt/scratch/bowmand8/taPVAT_scVIDR/code/scVIDR/results/r2_seeds_*.csv"
