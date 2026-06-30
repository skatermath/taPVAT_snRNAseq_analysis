#!/usr/bin/env python
"""Run taPVAT scVIDR held-out target prediction.

The default command is intentionally inspectable: use `list-targets` first to
confirm the target counts before preparing inputs or launching model training.
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import pandas as pd
import scanpy as sc
import yaml


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
DEFAULT_CONFIG = REPO_ROOT / "code/scVIDR/configs/tapvat_scvidr_targets.yml"


@dataclass(frozen=True)
class Scope:
    age: str
    sex_label: str
    sexes: tuple[str, ...]


def load_config(path: Path) -> dict:
    with path.open() as handle:
        config = yaml.safe_load(handle)
    return config


def resolve_repo_path(path_value: str | Path) -> Path:
    path = Path(path_value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def obs_columns(config: dict) -> dict:
    return config["obs_columns"]


def read_obs(config: dict) -> pd.DataFrame:
    cols = obs_columns(config)
    input_h5ad = resolve_repo_path(config["input_h5ad"])
    adata = sc.read_h5ad(input_h5ad, backed="r")
    required = [cols["time"], cols["sex"], cols["diet"], cols["celltype"]]
    missing = [col for col in required if col not in adata.obs.columns]
    if missing:
        raise ValueError(f"Input AnnData is missing required obs columns: {missing}")
    obs = adata.obs[required].copy()
    obs = obs[~obs[cols["celltype"]].isin(config["exclude_celltypes"])].copy()
    return obs


def iter_scopes(config: dict, age_filter: str | None = None, sex_filter: str | None = None) -> Iterable[Scope]:
    for age in config["ages"]:
        if age_filter and age != age_filter:
            continue
        for sex_label, sexes in config["sexes"].items():
            if sex_filter and sex_label != sex_filter:
                continue
            yield Scope(age=age, sex_label=sex_label, sexes=tuple(sexes))


def target_counts_for_scope(obs: pd.DataFrame, config: dict, scope: Scope) -> pd.DataFrame:
    cols = obs_columns(config)
    subset = obs[
        (obs[cols["time"]] == scope.age)
        & (obs[cols["sex"]].isin(scope.sexes))
    ].copy()

    if subset.empty:
        return pd.DataFrame(columns=["target", "control_count", "treated_count", "eligible", "reason"])

    counts = pd.crosstab(subset[cols["celltype"]], subset[cols["diet"]])
    targets = sorted(subset[cols["celltype"]].unique())

    rows = []
    for target in targets:
        control_count = int(counts.at[target, config["control_diet"]]) if target in counts.index and config["control_diet"] in counts.columns else 0
        treated_count = int(counts.at[target, config["treated_diet"]]) if target in counts.index and config["treated_diet"] in counts.columns else 0
        reasons = []
        # Structural requirement (NOT a count threshold): scVIDR learns a
        # Control->HF latent shift per cell type and regresses across cell types.
        # A cell type with literally zero cells in one condition has an undefined
        # centroid (NaN) that breaks the whole scope's regression, and as a target
        # it has either no baseline to perturb or no ground truth to score. Such a
        # cell type is excluded only for the scope where a group is empty.
        if control_count == 0:
            reasons.append("0 Control cells (perturbation undefined)")
        if treated_count == 0:
            reasons.append("0 HF cells (no ground truth)")
        if control_count < int(config["min_control_cells"]):
            reasons.append(f"Control cells < {config['min_control_cells']}")
        if treated_count < int(config["min_treated_cells"]):
            reasons.append(f"HF cells < {config['min_treated_cells']}")
        rows.append(
            {
                "age": scope.age,
                "sex_subset": scope.sex_label,
                "target": target,
                "control_count": control_count,
                "treated_count": treated_count,
                "eligible": not reasons,
                "reason": "; ".join(reasons) if reasons else "ok",
            }
        )
    return pd.DataFrame(rows)


def build_manifest(config: dict, age_filter: str | None = None, sex_filter: str | None = None) -> pd.DataFrame:
    obs = read_obs(config)
    frames = [target_counts_for_scope(obs, config, scope) for scope in iter_scopes(config, age_filter, sex_filter)]
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def write_manifest(manifest: pd.DataFrame, config: dict) -> Path:
    output_dir = resolve_repo_path(config["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "target_manifest.csv"
    manifest.to_csv(path, index=False)
    return path


def print_manifest(manifest: pd.DataFrame) -> None:
    if manifest.empty:
        print("No scopes matched the requested filters.")
        return
    for (age, sex_subset), frame in manifest.groupby(["age", "sex_subset"], sort=False):
        print(f"\n## {age} {sex_subset}")
        print(frame[["target", "control_count", "treated_count", "eligible", "reason"]].to_string(index=False))


def scope_dir(config: dict, scope: Scope) -> Path:
    return resolve_repo_path(config["output_dir"]) / scope.age / scope.sex_label


def target_dir(config: dict, scope: Scope, target: str, seed: int | None = None) -> Path:
    base = scope_dir(config, scope) / target
    return base if seed is None else base / f"seed{seed}"


def subset_input_path(config: dict, scope: Scope) -> Path:
    return scope_dir(config, scope) / "inputs" / f"{scope.age}_{scope.sex_label}_scvidr_targets.h5ad"


def celltypes_keep_path(config: dict, scope: Scope) -> Path:
    return scope_dir(config, scope) / "inputs" / f"{scope.age}_{scope.sex_label}_celltypes_keep.txt"


def eligible_targets(manifest: pd.DataFrame, scope: Scope, target_filter: str | None = None) -> list[str]:
    frame = manifest[
        (manifest["age"] == scope.age)
        & (manifest["sex_subset"] == scope.sex_label)
        & (manifest["eligible"])
    ]
    targets = frame["target"].tolist()
    if target_filter:
        targets = [target for target in targets if target == target_filter]
    return targets


def prepare_inputs(config: dict, manifest: pd.DataFrame, age_filter: str | None = None, sex_filter: str | None = None) -> list[Path]:
    input_h5ad = resolve_repo_path(config["input_h5ad"])
    cols = obs_columns(config)
    adata = sc.read_h5ad(input_h5ad)

    # Exclude technical artifacts (e.g. Doublets) from the full dataset
    adata = adata[~adata.obs[cols["celltype"]].isin(config["exclude_celltypes"])].copy()

    written = []
    for scope in iter_scopes(config, age_filter, sex_filter):
        targets = eligible_targets(manifest, scope)
        if not targets:
            print(f"Skipping input preparation for {scope.age} {scope.sex_label}: no eligible targets.")
            continue

        subset = adata[
            (adata.obs[cols["time"]] == scope.age)
            & (adata.obs[cols["sex"]].isin(scope.sexes))
            & (adata.obs[cols["celltype"]].isin(targets))
        ].copy()

        input_path = subset_input_path(config, scope)
        input_path.parent.mkdir(parents=True, exist_ok=True)
        subset.write_h5ad(input_path)

        keep_path = celltypes_keep_path(config, scope)
        keep_path.write_text("\n".join(targets) + "\n")

        written.extend([input_path, keep_path])
        print(f"Wrote {input_path.relative_to(REPO_ROOT)} with {subset.n_obs} cells and {len(targets)} targets.")

    return written


def upstream_bin_dir(config: dict) -> Path:
    return resolve_repo_path(config["upstream_dir"]) / "bin"


def command_text(cmd: list[str]) -> str:
    return shlex.join(cmd)


def run_command(cmd: list[str], cwd: Path, log_path: Path, dry_run: bool) -> None:
    if dry_run:
        print(command_text(cmd))
        return

    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as log:
        log.write(f"$ {command_text(cmd)}\n\n")
        subprocess.run(cmd, cwd=cwd, check=True, stdout=log, stderr=subprocess.STDOUT)


def write_command_record(path: Path, commands: list[list[str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(command_text(cmd) for cmd in commands) + "\n")


def commands_for_target(config: dict, scope: Scope, target: str, skip_gene_scores: bool, seed: int | None = None) -> list[tuple[str, list[str]]]:
    cols = obs_columns(config)
    python = sys.executable
    upstream_bin = upstream_bin_dir(config)
    input_path = subset_input_path(config, scope).resolve()
    keep_path = celltypes_keep_path(config, scope).resolve()
    out_dir = target_dir(config, scope, target, seed).resolve()
    model_dir = out_dir / "model"
    prediction_dir = out_dir / "predictions"
    gene_score_dir = out_dir / "gene_scores"

    train_cmd = [
        python,
        "scvidr_train.py",
        "single_dose",
        str(input_path),
        str(model_dir),
        "--dose_column",
        cols["diet"],
        "--celltype_column",
        cols["celltype"],
        "--test_celltype",
        target,
        "--control_dose",
        config["control_diet"],
        "--treated_dose",
        config["treated_diet"],
        "--celltypes_keep",
        str(keep_path),
        "--max_epochs",
        str(config["training"]["max_epochs"]),
    ]
    predict_cmd = [
        python,
        "scvidr_predict.py",
        "single_dose",
        str(input_path),
        str(model_dir),
        str(prediction_dir),
        "--model",
        "scVIDR",
        "--dose_column",
        cols["diet"],
        "--celltype_column",
        cols["celltype"],
        "--test_celltype",
        target,
        "--control_dose",
        config["control_diet"],
        "--treated_dose",
        config["treated_diet"],
        "--celltypes_keep",
        str(keep_path),
    ]
    if seed is not None:
        train_cmd += ["--seed", str(seed)]
        predict_cmd += ["--seed", str(seed)]
    commands = [("train", train_cmd), ("predict", predict_cmd)]

    if not skip_gene_scores:
        gene_cmd = [
            python,
            "scvidr_genescores.py",
            str(input_path),
            str(model_dir),
            str(gene_score_dir),
            "--dose_column",
            cols["diet"],
            "--celltype_column",
            cols["celltype"],
            "--test_celltype",
            target,
            "--control_dose",
            config["control_diet"],
            "--treated_dose",
            config["treated_diet"],
            "--celltypes_keep",
            str(keep_path),
            "--training_size",
            str(config["training"]["gene_score_training_size"]),
        ]
        commands.append(("gene_scores", gene_cmd))

    # Make the static checker happy if this function changes later.
    assert upstream_bin.exists(), f"Missing upstream bin directory: {upstream_bin}"
    return commands


def run_workflow(
    config: dict,
    manifest: pd.DataFrame,
    age_filter: str | None,
    sex_filter: str | None,
    target_filter: str | None,
    dry_run: bool,
    skip_gene_scores: bool,
    seed: int | None = None,
    predict_only: bool = False,
) -> None:
    upstream_bin = upstream_bin_dir(config)
    failures: list[str] = []
    seed_tag = "" if seed is None else f" seed{seed}"
    for scope in iter_scopes(config, age_filter, sex_filter):
        targets = eligible_targets(manifest, scope, target_filter)
        if not targets:
            print(f"Skipping {scope.age} {scope.sex_label}: no eligible targets matched.")
            continue

        if not dry_run and not subset_input_path(config, scope).exists():
            print(f"Preparing input for {scope.age} {scope.sex_label}.")
            prepare_inputs(config, manifest, scope.age, scope.sex_label)

        for target in targets:
            out_dir = target_dir(config, scope, target, seed)
            commands = commands_for_target(config, scope, target, skip_gene_scores, seed)
            if predict_only:
                # Re-run only prediction (reuse the already-trained model).
                commands = [(n, c) for (n, c) in commands if n != "train"]
            write_command_record(out_dir / "logs/commands.txt", [cmd for _, cmd in commands])
            print(f"\n# {scope.age} {scope.sex_label} {target}{seed_tag}")
            try:
                for step_name, cmd in commands:
                    log_path = out_dir / "logs" / f"{step_name}.log"
                    run_command(cmd, cwd=upstream_bin, log_path=log_path, dry_run=dry_run)
            except subprocess.CalledProcessError as exc:
                # With no cell-count threshold some tiny cell types may fail to
                # train/predict. Record and continue instead of halting the batch.
                label = f"{scope.age} {scope.sex_label} {target}{seed_tag}"
                print(f"  FAILED {label}: {exc}. See logs in {out_dir / 'logs'}.")
                failures.append(label)

    if failures:
        print(f"\n{len(failures)} target(s) failed and were skipped:")
        for label in failures:
            print(f"  - {label}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="taPVAT scVIDR workflow runner")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Workflow YAML config")

    subparsers = parser.add_subparsers(dest="command", required=True)

    list_parser = subparsers.add_parser("list-targets", help="List eligible prediction targets")
    list_parser.add_argument("--age", choices=["24W", "8W"])
    list_parser.add_argument("--sex", choices=["M", "F", "Both"])

    prep_parser = subparsers.add_parser("prepare-inputs", help="Write age/sex subset h5ad files")
    prep_parser.add_argument("--age", choices=["24W", "8W"])
    prep_parser.add_argument("--sex", choices=["M", "F", "Both"])

    run_parser = subparsers.add_parser("run", help="Run or dry-run train/predict/gene-score commands")
    run_parser.add_argument("--age", choices=["24W", "8W"])
    run_parser.add_argument("--sex", choices=["M", "F", "Both"])
    run_parser.add_argument("--target", help="Run one target only")
    run_parser.add_argument("--dry-run", action="store_true", help="Print commands instead of executing them")
    run_parser.add_argument("--skip-gene-scores", action="store_true", help="Skip gene-score calculation")
    run_parser.add_argument("--seed", type=int, default=None, help="Random seed; outputs go under <target>/seed<N>/")
    run_parser.add_argument("--predict-only", action="store_true", help="Skip training; re-run prediction from the existing model")

    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_repo_path(args.config))

    age_filter = getattr(args, "age", None)
    sex_filter = getattr(args, "sex", None)
    manifest = build_manifest(config, age_filter=age_filter, sex_filter=sex_filter)
    manifest_path = resolve_repo_path(config["output_dir"]) / "target_manifest.csv"
    # Only the planning commands write the shared manifest file. The `run`
    # command builds the manifest in memory for eligibility but does not write it,
    # so many concurrent array tasks never race on the same file.
    if args.command in ("list-targets", "prepare-inputs"):
        manifest_path = write_manifest(manifest, config)

    if args.command == "list-targets":
        print_manifest(manifest)
        print(f"\nWrote {manifest_path.relative_to(REPO_ROOT)}")
        return

    if args.command == "prepare-inputs":
        prepare_inputs(config, manifest, age_filter=age_filter, sex_filter=sex_filter)
        print(f"\nWrote {manifest_path.relative_to(REPO_ROOT)}")
        return

    if args.command == "run":
        run_workflow(
            config=config,
            manifest=manifest,
            age_filter=age_filter,
            sex_filter=sex_filter,
            target_filter=args.target,
            dry_run=args.dry_run,
            skip_gene_scores=args.skip_gene_scores,
            seed=args.seed,
            predict_only=args.predict_only,
        )
        return

    raise ValueError(f"Unknown command: {args.command}")


if __name__ == "__main__":
    main()
