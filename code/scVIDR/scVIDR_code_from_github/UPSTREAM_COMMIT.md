# scVIDR Upstream Source

This directory contains a source snapshot from:

https://github.com/BhattacharyaLab/scVIDR.git

Pinned commit:

```text
bcb6a96c85cddc4eef052be739c122efb737fe0e
```

The upstream `.git` directory was removed so this repository can track the
source files directly without creating a nested Git repository.

Local compatibility fixes in this vendored snapshot:

- `bin/scvidr_train.py` exposes `--max_epochs`.
- `bin/scvidr_predict.py` always creates the output directory and writes
  predictions inside it.
- `bin/scvidr_genescores.py` parses `--training_size` as an integer and uses a
  valid CPU/CUDA/MPS device-selection expression.
- `bin/scvidr_train.py` and `bin/scvidr_predict.py` expose `--seed`, which sets
  `scvi.settings.seed` (and numpy/torch) for reproducibility. This matters in
  predict because scVIDR balances cell populations with unseeded
  `np.random.choice`, so the predicted cells (and R2) are otherwise
  non-deterministic even for a fixed model.
- `bin/scvidr_predict.py` predicts from a uniform, reportable number of the
  held-out cell type's control cells -- `min(n_HF, 1500)`, randomly sampled --
  reusing the fitted regression, instead of scVIDR's balanced subsample (whose
  size varies widely by cell type, e.g. 1.4k for brown vs 6.6k for others). The
  predicted mean (and thus R2) is unchanged -- verified empirically that 1.4k vs
  6.6k control cells give the same R2 -- so this only standardizes the
  predicted-cell count for clearer reporting.
