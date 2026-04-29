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
CAMPAIGN_DIR = EXPERIMENT_DIR / "tuning" / "geometry_separation"
STAGE_NAMES = ["stage0", "stage1", "stage2", "stage3"]


def format_value(value: float | int | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}".replace("-", "m").replace(".", "p")
    return value.replace("_", "")


def run_step(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.write_text("\n".join(lines), encoding="utf-8")


def checkpoint_epoch(path: Path) -> int:
    suffix = path.name.rsplit(".", 1)[-1]
    return int(suffix) if suffix.isdigit() else -1


def latest_numbered_checkpoint(prefix: Path) -> Path | None:
    snapshots = sorted(
        [path for path in prefix.parent.glob(f"{prefix.name}.*") if checkpoint_epoch(path) >= 0],
        key=checkpoint_epoch,
    )
    return snapshots[-1] if snapshots else None


def default_direct_init_checkpoint() -> Path | None:
    prefix = PROJECT_DIR / "experiments" / "poincare_only" / "results" / "disease90" / "disease90_embeddings_direct.pth"
    candidates = [Path(f"{prefix}.best"), prefix]
    latest = latest_numbered_checkpoint(prefix)
    if latest is not None:
        candidates.append(latest)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


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


def rank_summary(summary: dict[str, object]) -> tuple[float, ...]:
    metrics = summary["offline_rescore"]["selected_candidate"]["metrics"]
    return tuple(metrics["rank_key"])


def load_stage_summaries(campaign_dir: Path, stage_name: str) -> list[dict[str, object]]:
    stage_dir = campaign_dir / "runs" / stage_name
    if not stage_dir.exists():
        return []
    return [load_json(path) for path in sorted(stage_dir.glob("*/run_summary.json"))]


def best_stage_summaries(campaign_dir: Path, stage_name: str, top_k: int) -> list[dict[str, object]]:
    ranked = sorted(
        load_stage_summaries(campaign_dir, stage_name),
        key=lambda item: (
            1 if item["offline_rescore"]["selected_candidate"]["acceptance_pass"] else 0,
            *rank_summary(item),
        ),
        reverse=True,
    )
    return ranked[:top_k]


def make_run_name(stage_name: str, params: dict[str, object]) -> str:
    ordered_keys = [
        "depth_band_weight",
        "branch_weight",
        "branch_cos_margin",
        "radial_weight",
        "radial_margin",
        "cpcc_weight",
        "lr",
        "burnin",
        "epochs",
        "negs",
        "dampening",
        "dim",
        "batchsize",
        "geometry_schedule",
    ]
    parts = [stage_name]
    for key in ordered_keys:
        if key in params:
            parts.append(f"{key.replace('_', '')}-{format_value(params[key])}")
    return "__".join(parts)


def rescore_floor_args(params: dict[str, object]) -> list[str]:
    return [
        "--floor-reconstruction-map",
        "0.30",
        "--floor-parent-mean-rank",
        "1.20",
        "--floor-ancestor-map",
        "0.95",
        "--floor-sibling-ratio",
        "0.30",
        "--floor-branch-ratio",
        "0.35",
        "--floor-min-adjacent-gap",
        str(params.get("floor_min_adjacent_gap", 0.0)),
        "--floor-positive-quantile-gaps",
        "3",
        "--ceiling-radial-violation",
        "0.10",
    ]


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

    write_json(manifest_json, {"stage": stage_name, "run_name": run_name, "params": params})

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
        "--depth-band-weight",
        str(params["depth_band_weight"]),
        "--branch-weight",
        str(params["branch_weight"]),
        "--branch-cos-margin",
        str(params["branch_cos_margin"]),
        "--geometry-schedule",
        str(params["geometry_schedule"]),
    ]
    if params.get("init_checkpoint"):
        train_command.extend(["--init-checkpoint", str(params["init_checkpoint"])])
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
            *rescore_floor_args(params),
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
    summary_dir = campaign_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    ranked = sorted(
        summaries,
        key=lambda item: (
            1 if item["offline_rescore"]["selected_candidate"]["acceptance_pass"] else 0,
            *rank_summary(item),
        ),
        reverse=True,
    )
    write_json(summary_dir / f"{stage_name}_summary.json", {"stage": stage_name, "run_count": len(ranked), "runs": ranked})

    lines = [f"# {stage_name} geometry summary", ""]
    for summary in ranked:
        metrics = summary["offline_rescore"]["selected_candidate"]["metrics"]
        radius = metrics["radius_structure"]
        ratios = metrics["ratios"]
        lines.extend(
            [
                f"## {summary['run_name']}",
                f"- acceptance_pass: {summary['offline_rescore']['selected_candidate']['acceptance_pass']}",
                f"- reconstruction_map: {metrics['reconstruction']['map_rank']:.4f}",
                f"- parent_mean_rank: {metrics['parent_ranking']['mean_rank']:.4f}",
                f"- ancestor_map: {metrics['ancestor_ranking']['mean_average_precision']:.4f}",
                f"- branch_ratio: {ratios['within_branch_to_across_branch_mean']:.4f}",
                f"- sibling_ratio: {ratios['sibling_to_non_sibling_mean']:.4f}",
                f"- positive_quantile_gaps: {radius['positive_adjacent_depth_quantile_gap_count']}",
                f"- min_quantile_gap: {radius['minimum_adjacent_quantile_gap']:.6f}",
                f"- radial_violation: {radius['parent_child_radial_violation_rate']:.4f}",
                "",
            ]
        )
    write_markdown(summary_dir / f"{stage_name}_summary.md", lines)


def base_params(gpu: int) -> dict[str, object]:
    return {
        "cpcc_weight": 0.05,
        "cpcc_min_group_size": 2,
        "radial_weight": 0.05,
        "radial_margin": 0.01,
        "depth_band_weight": 0.05,
        "branch_weight": 0.05,
        "branch_cos_margin": 0.2,
        "geometry_schedule": "ramp",
        "lr": 0.10,
        "burnin": 20,
        "epochs": 300,
        "negs": 50,
        "dampening": 0.75,
        "dim": 10,
        "batchsize": 64,
        "gpu": gpu,
    }


def stage1_configs(gpu: int) -> list[dict[str, object]]:
    base = base_params(gpu)
    configs = []
    for depth_weight, branch_weight, label in [
        (0.05, 0.0, "depth_only"),
        (0.0, 0.05, "branch_only"),
        (0.05, 0.05, "combined"),
    ]:
        params = dict(base)
        params["depth_band_weight"] = depth_weight
        params["branch_weight"] = branch_weight
        params["ablation"] = label
        configs.append(params)
    return configs


def stage2_configs(gpu: int) -> list[dict[str, object]]:
    base = base_params(gpu)
    configs = []
    for depth_weight in [0.02, 0.05, 0.10]:
        for branch_weight in [0.02, 0.05, 0.10]:
            for radial_weight in [0.02, 0.05]:
                params = dict(base)
                params["depth_band_weight"] = depth_weight
                params["branch_weight"] = branch_weight
                params["radial_weight"] = radial_weight
                configs.append(params)
    return configs


def stage3_configs(campaign_dir: Path, gpu: int) -> list[dict[str, object]]:
    init_checkpoint = default_direct_init_checkpoint()
    configs = []
    for summary in best_stage_summaries(campaign_dir, "stage2", top_k=3):
        params = dict(summary["params"])
        params["epochs"] = 500
        params["geometry_schedule"] = "ramp"
        params["gpu"] = gpu
        if init_checkpoint is not None:
            params["init_checkpoint"] = str(init_checkpoint)
        configs.append(params)
    if not configs:
        raise ValueError("Stage 3 requires completed Stage 2 results")
    return configs


def maybe_limit(configs: list[dict[str, object]], limit_runs: int) -> list[dict[str, object]]:
    return configs[:limit_runs] if limit_runs > 0 else configs


def run_stage_configs(
    stage_name: str,
    configs: list[dict[str, object]],
    campaign_dir: Path,
    env: dict[str, str],
    skip_existing: bool,
) -> list[dict[str, object]]:
    summaries = [
        run_single_training(stage_name, params, campaign_dir, env, skip_existing)
        for params in configs
    ]
    write_stage_summary(campaign_dir, stage_name, summaries)
    return summaries


def run_stage0(campaign_dir: Path, env: dict[str, str]) -> None:
    stage0_dir = campaign_dir / "stage0"
    stage0_dir.mkdir(parents=True, exist_ok=True)
    baselines = [
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
    for baseline in baselines:
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
    lines = ["# Stage 0 geometry baseline calibration", ""]
    for comparison in comparisons:
        metrics = comparison["evaluation"]
        radius = metrics["radius_structure"]
        ratios = metrics["ratios"]
        lines.extend(
            [
                f"## {comparison['name']}",
                f"- reconstruction_map: {metrics['reconstruction']['map_rank']:.4f}",
                f"- parent_mean_rank: {metrics['parent_ranking']['mean_rank']:.4f}",
                f"- ancestor_map: {metrics['ancestor_ranking']['mean_average_precision']:.4f}",
                f"- branch_ratio: {ratios['within_branch_to_across_branch_mean']:.4f}",
                f"- sibling_ratio: {ratios['sibling_to_non_sibling_mean']:.4f}",
                f"- min_adjacent_gap: {radius['minimum_adjacent_gap']:.6f}",
                f"- positive_quantile_gaps: {radius['positive_adjacent_depth_quantile_gap_count']}",
                f"- radial_violation: {radius['parent_child_radial_violation_rate']:.4f}",
                f"- branch_silhouette: {metrics['branch_geometry']['branch_silhouette']:.4f}",
                "",
            ]
        )
    write_markdown(stage0_dir / "baseline_calibration.md", lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run disease-90 geometry-separation tuning")
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
        run_stage_configs("stage1", maybe_limit(stage1_configs(args.gpu), args.limit_runs), args.campaign_dir, env, args.skip_existing)
    if "stage2" in args.stages:
        run_stage_configs("stage2", maybe_limit(stage2_configs(args.gpu), args.limit_runs), args.campaign_dir, env, args.skip_existing)
    if "stage3" in args.stages:
        run_stage_configs("stage3", maybe_limit(stage3_configs(args.campaign_dir, args.gpu), args.limit_runs), args.campaign_dir, env, args.skip_existing)


if __name__ == "__main__":
    main()
