#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Set these before importing project modules that load Torch/OpenMP. Forking
# after OpenMP initializes can fail on managed macOS sandboxes.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
CG_RESULTS_DIR = EXPERIMENT_DIR / "results" / "disease_cg"
CG_PLOTS_DIR = CG_RESULTS_DIR / "plots"
CG_LABMATE_DIR = CG_RESULTS_DIR / "labmate"
DEFAULT_CG_RELATIONS_CSV = CG_RESULTS_DIR / "cg_relations_direct.csv"
DEFAULT_CG_METADATA_TSV = CG_RESULTS_DIR / "cg_nodes.tsv"
DEFAULT_CG_NODE_LIST = CG_RESULTS_DIR / "cg_nodes.txt"
DEFAULT_CG_SUMMARY_JSON = CG_RESULTS_DIR / "cg_dataset_summary.json"
RECOMMENDED_RUNTIME = "/opt/homebrew/Caskroom/miniforge/base/envs/logic-bank-benchmark/bin/python"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_step(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)


def train_command(
    checkpoint: Path,
    train_config: Path,
    train_log: Path,
    metadata_tsv: Path,
    relations_csv: Path,
    params: dict[str, object],
) -> list[str]:
    command = [
        sys.executable,
        str(SCRIPT_DIR / "train_disease90.py"),
        "--dataset",
        str(relations_csv),
        "--metadata-tsv",
        str(metadata_tsv),
        "--checkpoint",
        str(checkpoint),
        "--train-config",
        str(train_config),
        "--log",
        str(train_log),
        "--selection-metric",
        str(params["selection_metric"]),
        "--eval-each",
        str(params["eval_each"]),
        "--fresh",
        "--quiet",
        "--dim",
        str(params["dim"]),
        "--epochs",
        str(params["epochs"]),
        "--lr",
        str(params["lr"]),
        "--negs",
        str(params["negs"]),
        "--batchsize",
        str(params["batchsize"]),
        "--burnin",
        str(params["burnin"]),
        "--dampening",
        str(params["dampening"]),
        "--gpu",
        str(params["gpu"]),
        "--cpcc-weight",
        str(params["cpcc_weight"]),
        "--radial-weight",
        str(params["radial_weight"]),
        "--radial-margin",
        str(params["radial_margin"]),
        "--cpcc-min-group-size",
        str(params["cpcc_min_group_size"]),
        "--depth-band-weight",
        str(params["depth_band_weight"]),
        "--depth-quantile-weight",
        str(params["depth_quantile_weight"]),
        "--depth-quantile-margin",
        str(params["depth_quantile_margin"]),
        "--branch-weight",
        str(params["branch_weight"]),
        "--branch-cos-margin",
        str(params["branch_cos_margin"]),
        "--branch-teacher-weight",
        str(params["branch_teacher_weight"]),
        "--branch-contrastive-weight",
        str(params["branch_contrastive_weight"]),
        "--branch-contrastive-margin",
        str(params["branch_contrastive_margin"]),
        "--branch-contrastive-hard-k",
        str(params["branch_contrastive_hard_k"]),
        "--geometry-schedule",
        str(params["geometry_schedule"]),
        "--init-source",
        str(params["init_source"]),
    ]
    if params.get("init_checkpoint"):
        command.extend(["--init-checkpoint", str(params["init_checkpoint"])])
    if params.get("branch_teacher_checkpoint"):
        command.extend(["--branch-teacher-checkpoint", str(params["branch_teacher_checkpoint"])])
    return command


def baseline_params(args: argparse.Namespace) -> dict[str, object]:
    return {
        "dim": 10,
        "epochs": args.baseline_epochs,
        "lr": 0.1,
        "negs": 50,
        "batchsize": 64,
        "burnin": 20,
        "dampening": 0.75,
        "gpu": args.gpu,
        "eval_each": args.eval_each,
        "cpcc_weight": 0.05,
        "cpcc_min_group_size": 2,
        "radial_weight": 0.01,
        "radial_margin": 0.02,
        "depth_band_weight": 0.0,
        "depth_quantile_weight": 0.0,
        "depth_quantile_margin": 0.001,
        "branch_weight": 0.0,
        "branch_cos_margin": 0.2,
        "branch_teacher_weight": 0.0,
        "branch_teacher_checkpoint": None,
        "branch_contrastive_weight": 0.0,
        "branch_contrastive_margin": 0.02,
        "branch_contrastive_hard_k": 0,
        "geometry_schedule": "ramp",
        "selection_metric": "combined",
        "init_source": "none",
        "init_checkpoint": None,
    }


def repair_params(args: argparse.Namespace, baseline_best: Path) -> dict[str, object]:
    return {
        "dim": 10,
        "epochs": args.repair_epochs,
        "lr": 0.03,
        "negs": 50,
        "batchsize": 64,
        "burnin": 20,
        "dampening": 0.75,
        "gpu": args.gpu,
        "eval_each": args.eval_each,
        "cpcc_weight": 0.05,
        "cpcc_min_group_size": 2,
        "radial_weight": 0.10,
        "radial_margin": 0.0005,
        "depth_band_weight": 0.0,
        "depth_quantile_weight": 0.1,
        "depth_quantile_margin": 0.0005,
        "branch_weight": 0.0,
        "branch_cos_margin": 0.2,
        "branch_teacher_weight": 0.1,
        "branch_teacher_checkpoint": baseline_best,
        "branch_contrastive_weight": 0.3,
        "branch_contrastive_margin": 0.02,
        "branch_contrastive_hard_k": 0,
        "geometry_schedule": "constant",
        "selection_metric": "combined",
        "init_source": "none",
        "init_checkpoint": baseline_best,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the C/G Poincare+HypStructure export pipeline")
    parser.add_argument("--results-dir", type=Path, default=CG_RESULTS_DIR)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_CG_METADATA_TSV)
    parser.add_argument("--relations-csv", type=Path, default=DEFAULT_CG_RELATIONS_CSV)
    parser.add_argument("--nodes-txt", type=Path, default=DEFAULT_CG_NODE_LIST)
    parser.add_argument("--dataset-summary-json", type=Path, default=DEFAULT_CG_SUMMARY_JSON)
    parser.add_argument("--baseline-epochs", type=int, default=300)
    parser.add_argument("--repair-epochs", type=int, default=500)
    parser.add_argument("--eval-each", type=int, default=5)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-baseline", action="store_true")
    parser.add_argument("--skip-repair", action="store_true")
    parser.add_argument("--skip-rescore", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--skip-labmate", action="store_true")
    args = parser.parse_args()

    if args.smoke:
        args.baseline_epochs = min(args.baseline_epochs, 2)
        args.repair_epochs = min(args.repair_epochs, 2)
        args.eval_each = 1

    args.results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = args.results_dir / "plots"
    labmate_dir = args.results_dir / "labmate"
    mpl_dir = args.results_dir / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env = os.environ.copy()
    env["MPLCONFIGDIR"] = str(mpl_dir)
    env["XDG_CACHE_HOME"] = str(mpl_dir)
    env["OMP_NUM_THREADS"] = "1"
    env["MKL_NUM_THREADS"] = "1"
    env["OPENBLAS_NUM_THREADS"] = "1"

    baseline_checkpoint = args.results_dir / "cg_current_hybrid.pth"
    baseline_best = Path(f"{baseline_checkpoint}.best")
    baseline_train_config = args.results_dir / "baseline_train_config.json"
    baseline_train_log = args.results_dir / "baseline_train.log"
    repair_checkpoint = args.results_dir / "cg_stage_d_repair.pth"
    repair_train_config = args.results_dir / "repair_train_config.json"
    repair_train_log = args.results_dir / "repair_train.log"
    offline_rescore_json = args.results_dir / "offline_rescore.json"
    offline_rescore_md = args.results_dir / "offline_rescore.md"
    offline_best = args.results_dir / "cg_stage_d_repair.offline_best.pth"
    final_checkpoint = offline_best if offline_best.exists() or not args.skip_rescore else Path(f"{repair_checkpoint}.best")
    export_tsv = args.results_dir / "embeddings.tsv"
    embedding_stats = args.results_dir / "embedding_stats.json"
    eval_json = args.results_dir / "eval_metrics.json"
    eval_md = args.results_dir / "eval_summary.md"

    if not args.skip_build:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "build_cg_subforest.py"),
                "--out-csv",
                str(args.relations_csv),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--nodes-txt",
                str(args.nodes_txt),
                "--summary-json",
                str(args.dataset_summary_json),
                "--mode",
                "direct",
            ],
            env,
        )

    if not args.skip_baseline:
        run_step(
            train_command(
                baseline_checkpoint,
                baseline_train_config,
                baseline_train_log,
                args.metadata_tsv,
                args.relations_csv,
                baseline_params(args),
            ),
            env,
        )
    if not baseline_best.exists():
        raise FileNotFoundError(f"Missing C/G baseline teacher checkpoint: {baseline_best}")

    if not args.skip_repair:
        run_step(
            train_command(
                repair_checkpoint,
                repair_train_config,
                repair_train_log,
                args.metadata_tsv,
                args.relations_csv,
                repair_params(args, baseline_best),
            ),
            env,
        )

    if not args.skip_rescore:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "rescore_disease90_run.py"),
                "--checkpoint-prefix",
                str(repair_checkpoint),
                "--train-log",
                str(repair_train_log),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--relations-csv",
                str(args.relations_csv),
                "--out-json",
                str(offline_rescore_json),
                "--out-md",
                str(offline_rescore_md),
                "--best-checkpoint",
                str(offline_best),
                "--top-k-full-eval",
                "10",
            ],
            env,
        )
        final_checkpoint = offline_best
    if not final_checkpoint.exists():
        raise FileNotFoundError(f"Missing final C/G checkpoint: {final_checkpoint}")

    if not args.skip_export:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "export_disease90_embeddings.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--out-tsv",
                str(export_tsv),
                "--stats-json",
                str(embedding_stats),
            ],
            env,
        )

    if not args.skip_visualize:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_disease90.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--plots-dir",
                str(plots_dir),
                "--title-label",
                "C/G Poincare+HypStructure",
                "--root-node-id",
                "CG_ROOT",
                "--root-label",
                "C/G synthetic root",
                "--label-route-node-ids",
                "CG_ROOT",
                "20",
                "60",
            ],
            env,
        )

    if not args.skip_evaluate:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "evaluate_disease90.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--relations-csv",
                str(args.relations_csv),
                "--out-json",
                str(eval_json),
                "--out-md",
                str(eval_md),
                "--title-label",
                "C/G embedding",
            ],
            env,
        )

    command_text = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if not args.skip_labmate:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "export_cg_labmate_embeddings.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(args.metadata_tsv),
                "--dataset-summary-json",
                str(args.dataset_summary_json),
                "--eval-json",
                str(eval_json),
                "--offline-rescore-json",
                str(offline_rescore_json),
                "--baseline-train-config",
                str(baseline_train_config),
                "--repair-train-config",
                str(repair_train_config),
                "--out-dir",
                str(labmate_dir),
                "--runtime",
                RECOMMENDED_RUNTIME,
                "--command",
                command_text,
            ],
            env,
        )

    run_summary = {
        "results_dir": str(args.results_dir),
        "runtime": sys.executable,
        "recommended_runtime": RECOMMENDED_RUNTIME,
        "metadata_tsv": str(args.metadata_tsv),
        "relations_csv": str(args.relations_csv),
        "dataset_summary_json": str(args.dataset_summary_json),
        "baseline_checkpoint": str(baseline_best),
        "repair_checkpoint_prefix": str(repair_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "offline_rescore_json": str(offline_rescore_json),
        "eval_json": str(eval_json),
        "labmate_code_only": str(labmate_dir / "cg_embeddings_codes_only.tsv"),
        "labmate_all_nodes": str(labmate_dir / "cg_embeddings_all_nodes.tsv"),
        "baseline_params": baseline_params(args),
        "repair_params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in repair_params(args, baseline_best).items()
        },
    }
    write_json(args.results_dir / "run_summary.json", run_summary)
    print(f"Wrote C/G run summary to {args.results_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
