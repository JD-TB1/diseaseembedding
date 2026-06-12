#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from pathlib import Path


os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"


SCRIPT_DIR = Path(__file__).resolve().parent
EXPERIMENT_DIR = SCRIPT_DIR.parent
RESULTS_DIR = EXPERIMENT_DIR / "results"
RECOMMENDED_RUNTIME = "/opt/homebrew/Caskroom/miniforge/base/envs/logic-bank-benchmark/bin/python"


def normalize_prefixes(values: list[str] | tuple[str, ...]) -> list[str]:
    prefixes: list[str] = []
    for value in values:
        for item in str(value).split(","):
            cleaned = item.strip().upper()
            if cleaned:
                prefixes.append(cleaned)
    prefixes = list(dict.fromkeys(prefixes))
    if not prefixes:
        raise ValueError("At least one ICD code prefix is required")
    invalid = [prefix for prefix in prefixes if not re.fullmatch(r"[A-Z]", prefix)]
    if invalid:
        raise ValueError(f"Prefixes must be one ICD letter each, got: {invalid}")
    return prefixes


def default_file_prefix(prefixes: list[str]) -> str:
    return "".join(prefixes).lower()


def default_results_dir(prefixes: list[str]) -> Path:
    return RESULTS_DIR / f"disease_{default_file_prefix(prefixes)}"


def default_synthetic_root_id(prefixes: list[str]) -> str:
    return f"{''.join(prefixes)}_ROOT"


def default_synthetic_root_meaning(prefixes: list[str]) -> str:
    return f"Synthetic root for ICD {'/'.join(prefixes)} target subforest"


def write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def run_step(command: list[str], env: dict[str, str]) -> None:
    print(" ".join(command))
    subprocess.run(command, check=True, env=env)


def load_json(path: Path) -> dict[str, object]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return payload


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


def derive_label_route(summary: dict[str, object], metadata_tsv: Path) -> tuple[str, str, list[str]]:
    natural_roots = summary.get("natural_roots")
    if not isinstance(natural_roots, list) or not natural_roots:
        raise ValueError("Dataset summary is missing natural roots")
    root_mode = summary.get("root_mode")
    if root_mode == "combined":
        root_id = str(summary["root_id"])
        root_label = str(summary.get("synthetic_root_id") or root_id)
        route = [root_id]
        for item in natural_roots:
            if isinstance(item, dict):
                route.append(str(item["node_id"]))
        return root_id, root_label, route

    root = natural_roots[0]
    if not isinstance(root, dict):
        raise ValueError("Dataset summary natural root entry is malformed")
    root_id = str(root["node_id"])
    root_label = str(root["coding"])

    rows = []
    with metadata_tsv.open(encoding="utf-8") as handle:
        header = handle.readline().rstrip("\n").split("\t")
        for line in handle:
            values = line.rstrip("\n").split("\t")
            rows.append(dict(zip(header, values)))
    route = [root_id]
    current = root_id
    for _ in range(4):
        children = [row for row in rows if row["parent_id"] == current]
        if not children:
            break
        current = children[0]["node_id"]
        route.append(current)
    return root_id, root_label, route


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a generic ICD prefix Poincare+HypStructure export pipeline")
    parser.add_argument("--prefixes", nargs="+", required=True, help="ICD prefixes, e.g. C, G, I, or C G I")
    parser.add_argument("--root-mode", choices=["auto", "individual", "combined"], default="auto")
    parser.add_argument("--results-dir", type=Path, default=None)
    parser.add_argument("--file-prefix", default=None)
    parser.add_argument("--dataset-label", default=None)
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

    prefixes = normalize_prefixes(args.prefixes)
    args.file_prefix = args.file_prefix or default_file_prefix(prefixes)
    args.results_dir = args.results_dir or default_results_dir(prefixes)
    args.dataset_label = args.dataset_label or f"{'/'.join(prefixes)} embedding"
    root_mode = args.root_mode
    if root_mode == "auto":
        root_mode = "combined" if len(prefixes) > 1 else "individual"

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

    metadata_tsv = args.results_dir / f"{args.file_prefix}_nodes.tsv"
    relations_csv = args.results_dir / f"{args.file_prefix}_relations_direct.csv"
    nodes_txt = args.results_dir / f"{args.file_prefix}_nodes.txt"
    dataset_summary_json = args.results_dir / f"{args.file_prefix}_dataset_summary.json"
    baseline_checkpoint = args.results_dir / f"{args.file_prefix}_current_hybrid.pth"
    baseline_best = Path(f"{baseline_checkpoint}.best")
    baseline_train_config = args.results_dir / "baseline_train_config.json"
    baseline_train_log = args.results_dir / "baseline_train.log"
    repair_checkpoint = args.results_dir / f"{args.file_prefix}_stage_d_repair.pth"
    repair_train_config = args.results_dir / "repair_train_config.json"
    repair_train_log = args.results_dir / "repair_train.log"
    offline_rescore_json = args.results_dir / "offline_rescore.json"
    offline_rescore_md = args.results_dir / "offline_rescore.md"
    offline_best = args.results_dir / f"{args.file_prefix}_stage_d_repair.offline_best.pth"
    final_checkpoint = offline_best if offline_best.exists() or not args.skip_rescore else Path(f"{repair_checkpoint}.best")
    export_tsv = args.results_dir / "embeddings.tsv"
    embedding_stats = args.results_dir / "embedding_stats.json"
    eval_json = args.results_dir / "eval_metrics.json"
    eval_md = args.results_dir / "eval_summary.md"

    if not args.skip_build:
        build_command = [
            sys.executable,
            str(SCRIPT_DIR / "build_disease_subforest.py"),
            "--prefixes",
            *prefixes,
            "--root-mode",
            root_mode,
            "--out-csv",
            str(relations_csv),
            "--metadata-tsv",
            str(metadata_tsv),
            "--nodes-txt",
            str(nodes_txt),
            "--summary-json",
            str(dataset_summary_json),
            "--mode",
            "direct",
        ]
        if root_mode == "combined":
            synthetic_root_id = default_synthetic_root_id(prefixes)
            build_command.extend(
                [
                    "--synthetic-root-id",
                    synthetic_root_id,
                    "--synthetic-root-code",
                    synthetic_root_id,
                    "--synthetic-root-meaning",
                    default_synthetic_root_meaning(prefixes),
                ]
            )
        run_step(build_command, env)
    summary = load_json(dataset_summary_json)

    if not args.skip_baseline:
        run_step(
            train_command(
                baseline_checkpoint,
                baseline_train_config,
                baseline_train_log,
                metadata_tsv,
                relations_csv,
                baseline_params(args),
            ),
            env,
        )
    if not baseline_best.exists():
        raise FileNotFoundError(f"Missing baseline teacher checkpoint: {baseline_best}")

    if not args.skip_repair:
        run_step(
            train_command(
                repair_checkpoint,
                repair_train_config,
                repair_train_log,
                metadata_tsv,
                relations_csv,
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
                str(metadata_tsv),
                "--relations-csv",
                str(relations_csv),
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
        raise FileNotFoundError(f"Missing final checkpoint: {final_checkpoint}")

    if not args.skip_export:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "export_disease90_embeddings.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(metadata_tsv),
                "--out-tsv",
                str(export_tsv),
                "--stats-json",
                str(embedding_stats),
            ],
            env,
        )

    root_node_id, root_label, label_route = derive_label_route(summary, metadata_tsv)
    if not args.skip_visualize:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "visualize_disease90.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(metadata_tsv),
                "--plots-dir",
                str(plots_dir),
                "--title-label",
                args.dataset_label,
                "--root-node-id",
                root_node_id,
                "--root-label",
                root_label,
                "--label-route-node-ids",
                *label_route,
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
                str(metadata_tsv),
                "--relations-csv",
                str(relations_csv),
                "--out-json",
                str(eval_json),
                "--out-md",
                str(eval_md),
                "--title-label",
                args.dataset_label,
            ],
            env,
        )

    command_text = " ".join([sys.executable, str(Path(__file__).resolve()), *sys.argv[1:]])
    if not args.skip_labmate:
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "export_disease_labmate_embeddings.py"),
                "--checkpoint",
                str(final_checkpoint),
                "--metadata-tsv",
                str(metadata_tsv),
                "--dataset-summary-json",
                str(dataset_summary_json),
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
                "--file-prefix",
                args.file_prefix,
                "--dataset-label",
                args.dataset_label,
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
        "prefixes": prefixes,
        "root_mode": root_mode,
        "file_prefix": args.file_prefix,
        "dataset_label": args.dataset_label,
        "metadata_tsv": str(metadata_tsv),
        "relations_csv": str(relations_csv),
        "dataset_summary_json": str(dataset_summary_json),
        "baseline_checkpoint": str(baseline_best),
        "repair_checkpoint_prefix": str(repair_checkpoint),
        "final_checkpoint": str(final_checkpoint),
        "offline_rescore_json": str(offline_rescore_json),
        "eval_json": str(eval_json),
        "plots_dir": str(plots_dir),
        "labmate_code_only": str(labmate_dir / f"{args.file_prefix}_embeddings_codes_only.tsv"),
        "labmate_all_nodes": str(labmate_dir / f"{args.file_prefix}_embeddings_all_nodes.tsv"),
        "baseline_params": baseline_params(args),
        "repair_params": {
            key: str(value) if isinstance(value, Path) else value
            for key, value in repair_params(args, baseline_best).items()
        },
        "dataset_summary": summary,
    }
    write_json(args.results_dir / "run_summary.json", run_summary)
    print(f"Wrote run summary to {args.results_dir / 'run_summary.json'}")


if __name__ == "__main__":
    main()
