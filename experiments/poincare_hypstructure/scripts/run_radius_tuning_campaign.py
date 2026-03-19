#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

from disease90_common import (
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    EXPERIMENT_DIR,
    LOGS_DIR,
    PROJECT_DIR,
    ROOT_ID,
    parse_args_with_defaults,
    write_json,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EXPERIMENT_DIR / "tuning" / "radius_separation"
STAGE_NAMES = ["stage0", "stage1", "stage2", "stage3", "stage4", "stage5"]


def format_value(value: float | int) -> str:
    if isinstance(value, int):
        return str(value)
    text = f"{value:.4g}"
    return text.replace("-", "m").replace(".", "p")


def run_step(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def ensure_direct_dataset(env: dict[str, str]) -> None:
    run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_disease90_relations.py"),
            "--root-id",
            ROOT_ID,
            "--mode",
            "direct",
            "--out-csv",
            str(DEFAULT_RELATIONS_CSV),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
        ],
        env,
    )


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def rank_summary(summary: dict[str, object]) -> tuple[float, ...]:
    rescore = summary["offline_rescore"]
    metrics = rescore["selected_candidate"]["metrics"]
    return tuple(metrics["rank_key"])


def load_stage_summaries(campaign_dir: Path, stage_name: str) -> list[dict[str, object]]:
    stage_dir = campaign_dir / "runs" / stage_name
    if not stage_dir.exists():
        return []
    summaries = []
    for run_summary in sorted(stage_dir.glob("*/run_summary.json")):
        summaries.append(load_json(run_summary))
    return summaries


def best_stage_summaries(campaign_dir: Path, stage_name: str, top_k: int = 1) -> list[dict[str, object]]:
    summaries = load_stage_summaries(campaign_dir, stage_name)
    ranked = sorted(
        summaries,
        key=lambda item: (
            1 if item["offline_rescore"]["selected_candidate"]["acceptance_pass"] else 0,
            *rank_summary(item),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def make_run_name(stage_name: str, params: dict[str, object]) -> str:
    ordered_keys = [
        "cpcc_weight",
        "radial_weight",
        "radial_margin",
        "cpcc_min_group_size",
        "lr",
        "burnin",
        "epochs",
        "negs",
        "dampening",
        "dim",
        "batchsize",
    ]
    parts = [stage_name]
    for key in ordered_keys:
        if key in params:
            parts.append(f"{key.replace('_', '')}-{format_value(params[key])}")
    return "__".join(parts)


def run_single_training(
    stage_name: str,
    params: dict[str, object],
    campaign_dir: Path,
    env: dict[str, str],
    skip_existing: bool,
) -> dict[str, object]:
    run_name = make_run_name(stage_name, params)
    run_dir = campaign_dir / "runs" / stage_name / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_prefix = run_dir / f"{run_name}.pth"
    train_log = run_dir / "train.log"
    train_config = run_dir / "train_config.json"
    rescore_json = run_dir / "offline_rescore.json"
    rescore_md = run_dir / "offline_rescore.md"
    offline_best_checkpoint = run_dir / f"{run_name}.offline_best.pth"
    export_tsv = run_dir / "embeddings.tsv"
    embed_stats = run_dir / "embedding_stats.json"
    eval_json = run_dir / "eval_metrics.json"
    eval_md = run_dir / "eval_summary.md"
    plots_dir = run_dir / "plots"
    run_summary_json = run_dir / "run_summary.json"
    manifest_json = run_dir / "run_manifest.json"

    if skip_existing and run_summary_json.exists() and offline_best_checkpoint.exists():
        return load_json(run_summary_json)

    manifest = {"stage": stage_name, "run_name": run_name, "params": params}
    write_json(manifest_json, manifest)

    train_command = [
        sys.executable,
        str(SCRIPT_DIR / "train_disease90.py"),
        "--dataset",
        str(DEFAULT_RELATIONS_CSV),
        "--metadata-tsv",
        str(DEFAULT_METADATA_TSV),
        "--checkpoint",
        str(checkpoint_prefix),
        "--train-config",
        str(train_config),
        "--log",
        str(train_log),
        "--selection-metric",
        "depth_spearman",
        "--eval-each",
        "1",
        "--fresh",
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
    ]
    run_step(train_command, env)

    run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "rescore_disease90_run.py"),
            "--checkpoint-prefix",
            str(checkpoint_prefix),
            "--train-log",
            str(train_log),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
            "--relations-csv",
            str(DEFAULT_RELATIONS_CSV),
            "--out-json",
            str(rescore_json),
            "--out-md",
            str(rescore_md),
            "--best-checkpoint",
            str(offline_best_checkpoint),
        ],
        env,
    )

    run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "export_disease90_embeddings.py"),
            "--checkpoint",
            str(offline_best_checkpoint),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
            "--out-tsv",
            str(export_tsv),
            "--stats-json",
            str(embed_stats),
        ],
        env,
    )
    run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "visualize_disease90.py"),
            "--checkpoint",
            str(offline_best_checkpoint),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
            "--plots-dir",
            str(plots_dir),
        ],
        env,
    )
    run_step(
        [
            sys.executable,
            str(SCRIPT_DIR / "evaluate_disease90.py"),
            "--checkpoint",
            str(offline_best_checkpoint),
            "--metadata-tsv",
            str(DEFAULT_METADATA_TSV),
            "--relations-csv",
            str(DEFAULT_RELATIONS_CSV),
            "--out-json",
            str(eval_json),
            "--out-md",
            str(eval_md),
        ],
        env,
    )

    summary = {
        "stage": stage_name,
        "run_name": run_name,
        "run_dir": str(run_dir),
        "params": params,
        "offline_rescore": load_json(rescore_json),
        "evaluation": load_json(eval_json),
    }
    write_json(run_summary_json, summary)
    return summary


def write_stage_summary(campaign_dir: Path, stage_name: str, summaries: list[dict[str, object]]) -> None:
    stage_dir = campaign_dir / "summaries"
    stage_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        summaries,
        key=lambda item: (
            1 if item["offline_rescore"]["selected_candidate"]["acceptance_pass"] else 0,
            *rank_summary(item),
        ),
        reverse=True,
    )
    payload = {
        "stage": stage_name,
        "run_count": len(ranked),
        "runs": ranked,
    }
    write_json(stage_dir / f"{stage_name}_summary.json", payload)

    lines = [f"# {stage_name} summary", ""]
    for summary in ranked:
        metrics = summary["offline_rescore"]["selected_candidate"]["metrics"]
        lines.extend(
            [
                f"## {summary['run_name']}",
                f"- acceptance_pass: {summary['offline_rescore']['selected_candidate']['acceptance_pass']}",
                f"- depth_spearman: {metrics['depth_radius']['spearman']:.4f}",
                f"- min_adjacent_gap: {metrics['radius_structure']['minimum_adjacent_gap']:.6f}",
                f"- parent_child_radial_violation_rate: {metrics['radius_structure']['parent_child_radial_violation_rate']:.4f}",
                f"- leaf_mean_radius: {metrics['radius_structure']['leaf_mean_radius']:.6f}",
                f"- reconstruction_map: {metrics['reconstruction']['map_rank']:.4f}",
                "",
            ]
        )
    write_markdown(stage_dir / f"{stage_name}_summary.md", lines)


def stage1_configs(gpu: int) -> list[dict[str, object]]:
    configs = []
    for radial_weight in [0.05, 0.10, 0.20, 0.40]:
        for radial_margin in [0.01, 0.02]:
            configs.append(
                {
                    "cpcc_weight": 0.05,
                    "cpcc_min_group_size": 2,
                    "radial_weight": radial_weight,
                    "radial_margin": radial_margin,
                    "lr": 0.10,
                    "burnin": 20,
                    "epochs": 300,
                    "negs": 50,
                    "dampening": 0.75,
                    "dim": 10,
                    "batchsize": 64,
                    "gpu": gpu,
                }
            )
    return configs


def stage2_configs(base: dict[str, object]) -> list[dict[str, object]]:
    configs = []
    for cpcc_weight in [0.01, 0.03, 0.05]:
        for min_group_size in [2, 4, 8]:
            params = dict(base)
            params["cpcc_weight"] = cpcc_weight
            params["cpcc_min_group_size"] = min_group_size
            configs.append(params)
    return configs


def stage3_configs(bases: list[dict[str, object]]) -> list[dict[str, object]]:
    schedules = [
        {"lr": 0.05, "burnin": 5, "epochs": 400},
        {"lr": 0.05, "burnin": 10, "epochs": 400},
        {"lr": 0.03, "burnin": 0, "epochs": 600},
    ]
    configs = []
    for base in bases:
        for schedule in schedules:
            params = dict(base)
            params.update(schedule)
            configs.append(params)
    return configs


def stage4_configs(base: dict[str, object]) -> list[dict[str, object]]:
    configs = []
    for negs in [25, 50, 100]:
        for dampening in [0.50, 0.75]:
            params = dict(base)
            params["negs"] = negs
            params["dampening"] = dampening
            params["batchsize"] = 64
            configs.append(params)
    return configs


def stage5_configs(base: dict[str, object]) -> list[dict[str, object]]:
    configs = []
    for dim in [5, 8, 10]:
        for batchsize in [32, 64]:
            params = dict(base)
            scale = batchsize / 64.0
            params["dim"] = dim
            params["batchsize"] = batchsize
            params["cpcc_weight"] = round(float(base["cpcc_weight"]) * scale, 10)
            params["radial_weight"] = round(float(base["radial_weight"]) * scale, 10)
            configs.append(params)
    return configs


def maybe_limit(configs: list[dict[str, object]], limit_runs: int) -> list[dict[str, object]]:
    if limit_runs > 0:
        return configs[:limit_runs]
    return configs


def run_stage_configs(
    stage_name: str,
    configs: list[dict[str, object]],
    campaign_dir: Path,
    env: dict[str, str],
    skip_existing: bool,
) -> list[dict[str, object]]:
    summaries = []
    for params in configs:
        summaries.append(run_single_training(stage_name, params, campaign_dir, env, skip_existing))
    write_stage_summary(campaign_dir, stage_name, summaries)
    return summaries


def run_stage0(campaign_dir: Path, env: dict[str, str]) -> None:
    stage0_dir = campaign_dir / "stage0"
    stage0_dir.mkdir(parents=True, exist_ok=True)

    baseline_cases = [
        {
            "name": "poincare_only_direct",
            "checkpoint_prefix": PROJECT_DIR / "experiments" / "poincare_only" / "results" / "disease90" / "disease90_embeddings_direct.pth",
            "train_log": PROJECT_DIR / "experiments" / "poincare_only" / "logs" / "train_disease90_direct.log",
        },
        {
            "name": "current_hybrid",
            "checkpoint_prefix": EXPERIMENT_DIR / "results" / "disease90" / "disease90_embeddings_poincare_hypstructure.pth",
            "train_log": EXPERIMENT_DIR / "logs" / "train_disease90_poincare_hypstructure.log",
        },
    ]

    comparisons = []
    for baseline in baseline_cases:
        out_dir = stage0_dir / baseline["name"]
        out_dir.mkdir(parents=True, exist_ok=True)
        offline_best = out_dir / f"{baseline['name']}.offline_best.pth"
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "rescore_disease90_run.py"),
                "--checkpoint-prefix",
                str(baseline["checkpoint_prefix"]),
                "--train-log",
                str(baseline["train_log"]),
                "--metadata-tsv",
                str(DEFAULT_METADATA_TSV),
                "--relations-csv",
                str(DEFAULT_RELATIONS_CSV),
                "--out-json",
                str(out_dir / "offline_rescore.json"),
                "--out-md",
                str(out_dir / "offline_rescore.md"),
                "--best-checkpoint",
                str(offline_best),
            ],
            env,
        )
        run_step(
            [
                sys.executable,
                str(SCRIPT_DIR / "evaluate_disease90.py"),
                "--checkpoint",
                str(offline_best),
                "--metadata-tsv",
                str(DEFAULT_METADATA_TSV),
                "--relations-csv",
                str(DEFAULT_RELATIONS_CSV),
                "--out-json",
                str(out_dir / "eval_metrics.json"),
                "--out-md",
                str(out_dir / "eval_summary.md"),
            ],
            env,
        )
        comparisons.append(
            {
                "name": baseline["name"],
                "offline_rescore": load_json(out_dir / "offline_rescore.json"),
                "evaluation": load_json(out_dir / "eval_metrics.json"),
            }
        )

    write_json(stage0_dir / "baseline_calibration.json", {"cases": comparisons})
    lines = ["# Stage 0 baseline calibration", ""]
    for item in comparisons:
        metrics = item["evaluation"]
        lines.extend(
            [
                f"## {item['name']}",
                f"- depth_spearman: {metrics['depth_radius']['spearman']:.4f}",
                f"- min_adjacent_gap: {metrics['radius_structure']['minimum_adjacent_gap']:.6f}",
                f"- parent_child_radial_violation_rate: {metrics['radius_structure']['parent_child_radial_violation_rate']:.4f}",
                f"- leaf_mean_radius: {metrics['radius_structure']['leaf_mean_radius']:.6f}",
                f"- leaf_internal_radius_ratio: {metrics['radius_structure']['leaf_internal_radius_ratio']:.4f}",
                f"- reconstruction_map: {metrics['reconstruction']['map_rank']:.4f}",
                f"- parent_mean_rank: {metrics['parent_ranking']['mean_rank']:.4f}",
                "",
            ]
        )
    write_markdown(stage0_dir / "baseline_calibration.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the disease-90 radius-separation tuning campaign")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--stages", nargs="+", choices=STAGE_NAMES, default=STAGE_NAMES)
    parser.add_argument("--gpu", type=int, default=-1)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    args = parse_args_with_defaults(parser)

    env = os.environ.copy()
    mpl_dir = LOGS_DIR / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    ensure_direct_dataset(env)
    args.campaign_dir.mkdir(parents=True, exist_ok=True)

    if "stage0" in args.stages:
        run_stage0(args.campaign_dir, env)

    if "stage1" in args.stages:
        run_stage_configs(
            "stage1",
            maybe_limit(stage1_configs(args.gpu), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )

    if "stage2" in args.stages:
        stage1_best = best_stage_summaries(args.campaign_dir, "stage1", top_k=1)
        if not stage1_best:
            raise ValueError("Stage 2 requires completed Stage 1 results")
        run_stage_configs(
            "stage2",
            maybe_limit(stage2_configs(stage1_best[0]["params"]), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )

    if "stage3" in args.stages:
        stage2_best = best_stage_summaries(args.campaign_dir, "stage2", top_k=2)
        if len(stage2_best) < 2:
            raise ValueError("Stage 3 requires two completed Stage 2 results")
        run_stage_configs(
            "stage3",
            maybe_limit(stage3_configs([summary["params"] for summary in stage2_best]), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )

    if "stage4" in args.stages:
        stage3_best = best_stage_summaries(args.campaign_dir, "stage3", top_k=1)
        if not stage3_best:
            raise ValueError("Stage 4 requires completed Stage 3 results")
        run_stage_configs(
            "stage4",
            maybe_limit(stage4_configs(stage3_best[0]["params"]), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )

    if "stage5" in args.stages:
        stage4_best = best_stage_summaries(args.campaign_dir, "stage4", top_k=1)
        if not stage4_best:
            raise ValueError("Stage 5 requires completed Stage 4 results")
        run_stage_configs(
            "stage5",
            maybe_limit(stage5_configs(stage4_best[0]["params"]), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )


if __name__ == "__main__":
    main()
