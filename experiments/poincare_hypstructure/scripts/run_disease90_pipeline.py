#!/usr/bin/env python3

import argparse
import os
import subprocess
import sys
from pathlib import Path

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    LOGS_DIR,
    ROOT_ID,
    parse_args_with_defaults,
)


def run_step(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the full disease-90 Poincare+HypStructure pipeline")
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument("--relation-mode", choices=["closure", "direct", "hybrid"], default="direct")
    parser.add_argument("--long-edge-stride", type=int, default=2)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--skip-build", action="store_true")
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--skip-export", action="store_true")
    parser.add_argument("--skip-visualize", action="store_true")
    parser.add_argument("--skip-evaluate", action="store_true")
    parser.add_argument("--dim", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--negs", type=int, default=50)
    parser.add_argument("--batchsize", type=int, default=64)
    parser.add_argument("--burnin", type=int, default=20)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--fresh", action="store_true")
    parser.add_argument("--cpcc-weight", type=float, default=0.05)
    parser.add_argument("--radial-weight", type=float, default=0.01)
    parser.add_argument("--radial-margin", type=float, default=0.02)
    parser.add_argument("--cpcc-min-group-size", type=int, default=2)
    parser.add_argument("--depth-band-weight", type=float, default=0.0)
    parser.add_argument("--depth-quantile-weight", type=float, default=0.0)
    parser.add_argument("--depth-quantile-margin", type=float, default=0.001)
    parser.add_argument("--branch-weight", type=float, default=0.0)
    parser.add_argument("--branch-cos-margin", type=float, default=0.2)
    parser.add_argument("--branch-teacher-weight", type=float, default=0.0)
    parser.add_argument("--branch-teacher-checkpoint", type=Path, default=None)
    parser.add_argument("--branch-contrastive-weight", type=float, default=0.0)
    parser.add_argument("--branch-contrastive-margin", type=float, default=0.02)
    parser.add_argument("--branch-contrastive-hard-k", type=int, default=0)
    parser.add_argument("--init-checkpoint", type=Path, default=None)
    parser.add_argument(
        "--init-source",
        choices=["current-hybrid", "direct-poincare", "none"],
        default="current-hybrid",
    )
    parser.add_argument("--geometry-schedule", choices=["ramp", "constant", "off"], default="ramp")
    parser.add_argument(
        "--selection-metric",
        choices=["combined", "reconstruction_map", "depth_spearman", "negative_loss"],
        default="combined",
    )
    args = parse_args_with_defaults(parser)

    env = os.environ.copy()
    mpl_dir = LOGS_DIR / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    script_dir = Path(__file__).resolve().parent
    if not args.skip_build:
        run_step(
            [
                sys.executable,
                str(script_dir / "build_disease90_relations.py"),
                "--root-id",
                args.root_id,
                "--mode",
                args.relation_mode,
                "--long-edge-stride",
                str(args.long_edge_stride),
                "--out-csv",
                str(args.dataset),
            ],
            env,
        )
    if not args.skip_train:
        command = [
            sys.executable,
            str(script_dir / "train_disease90.py"),
            "--dataset",
            str(args.dataset),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
            "--checkpoint",
            str(args.checkpoint),
            "--dim",
            str(args.dim),
            "--epochs",
            str(args.epochs),
            "--lr",
            str(args.lr),
            "--negs",
            str(args.negs),
            "--batchsize",
            str(args.batchsize),
            "--burnin",
            str(args.burnin),
            "--gpu",
            str(args.gpu),
            "--cpcc-weight",
            str(args.cpcc_weight),
            "--radial-weight",
            str(args.radial_weight),
            "--radial-margin",
            str(args.radial_margin),
            "--cpcc-min-group-size",
            str(args.cpcc_min_group_size),
            "--depth-band-weight",
            str(args.depth_band_weight),
            "--depth-quantile-weight",
            str(args.depth_quantile_weight),
            "--depth-quantile-margin",
            str(args.depth_quantile_margin),
            "--branch-weight",
            str(args.branch_weight),
            "--branch-cos-margin",
            str(args.branch_cos_margin),
            "--branch-teacher-weight",
            str(args.branch_teacher_weight),
            "--branch-contrastive-weight",
            str(args.branch_contrastive_weight),
            "--branch-contrastive-margin",
            str(args.branch_contrastive_margin),
            "--branch-contrastive-hard-k",
            str(args.branch_contrastive_hard_k),
            "--init-source",
            args.init_source,
            "--geometry-schedule",
            args.geometry_schedule,
            "--selection-metric",
            args.selection_metric,
        ]
        if args.branch_teacher_checkpoint is not None:
            command.extend(["--branch-teacher-checkpoint", str(args.branch_teacher_checkpoint)])
        if args.init_checkpoint is not None:
            command.extend(["--init-checkpoint", str(args.init_checkpoint)])
        if args.fresh:
            command.append("--fresh")
        run_step(command, env)
    if not args.skip_export:
        run_step(
            [
                sys.executable,
                str(script_dir / "export_disease90_embeddings.py"),
                "--checkpoint",
                f"{args.checkpoint}.best",
                "--metadata-tsv",
                str(DEFAULT_METADATA_TSV),
            ],
            env,
        )
    if not args.skip_visualize:
        run_step(
            [
                sys.executable,
                str(script_dir / "visualize_disease90.py"),
                "--checkpoint",
                f"{args.checkpoint}.best",
                "--metadata-tsv",
                str(DEFAULT_METADATA_TSV),
            ],
            env,
        )
    if not args.skip_evaluate:
        run_step(
            [
                sys.executable,
                str(script_dir / "evaluate_disease90.py"),
                "--checkpoint",
                f"{args.checkpoint}.best",
                "--metadata-tsv",
                str(DEFAULT_METADATA_TSV),
                "--relations-csv",
                str(args.dataset),
            ],
            env,
        )


if __name__ == "__main__":
    main()
