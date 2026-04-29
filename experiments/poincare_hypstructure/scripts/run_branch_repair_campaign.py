#!/usr/bin/env python3

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    EXPERIMENT_DIR,
    LOGS_DIR,
    PROJECT_DIR,
    parse_args_with_defaults,
    write_json,
)
from disease90_metrics import (
    compute_disease90_metrics,
    gate_deficit_rank_key,
    load_metadata_and_relations,
    passes_acceptance_floors,
)


SCRIPT_DIR = Path(__file__).resolve().parent
CAMPAIGN_DIR = EXPERIMENT_DIR / "tuning" / "branch_repair"
STAGE_NAMES = ["stageA", "stageB", "stageC", "stageD"]


def run_step(command: list[str], env: dict[str, str]) -> None:
    subprocess.run(command, check=True, env=env)


def load_json(path: Path) -> dict[str, object]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_markdown(path: Path, lines: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def current_hybrid_checkpoint() -> Path | None:
    stage0_best = (
        EXPERIMENT_DIR
        / "tuning"
        / "geometry_separation"
        / "stage0"
        / "current_hybrid"
        / "current_hybrid.offline_best.pth"
    )
    candidates = [stage0_best, Path(f"{DEFAULT_CHECKPOINT}.offline_best"), Path(f"{DEFAULT_CHECKPOINT}.best"), DEFAULT_CHECKPOINT]
    latest = latest_numbered_checkpoint(DEFAULT_CHECKPOINT)
    if latest is not None:
        candidates.append(latest)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def direct_poincare_checkpoint() -> Path | None:
    prefix = PROJECT_DIR / "experiments" / "poincare_only" / "results" / "disease90" / "disease90_embeddings_direct.pth"
    candidates = [Path(f"{prefix}.best"), prefix]
    latest = latest_numbered_checkpoint(prefix)
    if latest is not None:
        candidates.append(latest)
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def default_repair_init_checkpoint() -> Path:
    checkpoint = current_hybrid_checkpoint() or direct_poincare_checkpoint()
    if checkpoint is None:
        raise FileNotFoundError("No current-hybrid or direct-Poincare initialization checkpoint was found")
    return checkpoint


def format_value(value: float | int | str) -> str:
    if isinstance(value, int):
        return str(value)
    if isinstance(value, float):
        return f"{value:.4g}".replace("-", "m").replace(".", "p")
    return value.replace("_", "")


def make_run_name(stage_name: str, params: dict[str, object]) -> str:
    key_aliases = {
        "depth_quantile_weight": "dqw",
        "depth_quantile_margin": "dqm",
        "branch_teacher_weight": "btw",
        "branch_contrastive_weight": "bcw",
        "branch_contrastive_margin": "bcm",
        "branch_contrastive_hard_k": "bck",
        "radial_weight": "rw",
        "radial_margin": "rm",
        "cpcc_weight": "cw",
        "lr": "lr",
        "epochs": "ep",
        "negs": "neg",
        "dampening": "damp",
        "dim": "dim",
        "batchsize": "bs",
    }
    digest_payload = json.dumps(params, sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.sha1(digest_payload).hexdigest()[:8]
    parts = [stage_name]
    for key, alias in key_aliases.items():
        if key in params:
            parts.append(f"{alias}{format_value(params[key])}")
    parts.append(digest)
    return "_".join(parts)


def rescore_floor_args() -> list[str]:
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
        "0.0",
        "--floor-positive-quantile-gaps",
        "3",
        "--ceiling-radial-violation",
        "0.10",
    ]


def candidate_record(label: str, checkpoint: Path) -> dict[str, object]:
    metadata_rows, relations = load_metadata_and_relations(DEFAULT_METADATA_TSV, DEFAULT_RELATIONS_CSV)
    metrics = compute_disease90_metrics(checkpoint, metadata_rows, relations)
    metrics["checkpoint"] = str(checkpoint)
    metrics["acceptance_pass"] = passes_acceptance_floors(metrics)
    metrics["rank_key"] = gate_deficit_rank_key(metrics)
    return {"label": label, "checkpoint": str(checkpoint), "metrics": metrics}


def baseline_record() -> dict[str, object] | None:
    checkpoint = current_hybrid_checkpoint()
    if checkpoint is None:
        return None
    return candidate_record("current_hybrid", checkpoint)


def stage_summary_paths(campaign_dir: Path, stage_name: str) -> tuple[Path, Path]:
    summary_dir = campaign_dir / "summaries"
    summary_dir.mkdir(parents=True, exist_ok=True)
    return summary_dir / f"{stage_name}_summary.json", summary_dir / f"{stage_name}_summary.md"


def summarize_candidate_lines(record: dict[str, object]) -> list[str]:
    metrics = record["metrics"]
    radius = metrics["radius_structure"]
    ratios = metrics["ratios"]
    return [
        f"## {record['label']}",
        f"- acceptance_pass: {metrics['acceptance_pass']}",
        f"- gate_deficit_score: {metrics['gate_deficit_score']:.4f}",
        f"- reconstruction_map: {metrics['reconstruction']['map_rank']:.4f}",
        f"- parent_mean_rank: {metrics['parent_ranking']['mean_rank']:.4f}",
        f"- ancestor_map: {metrics['ancestor_ranking']['mean_average_precision']:.4f}",
        f"- branch_ratio: {ratios['within_branch_to_across_branch_mean']:.4f}",
        f"- same_depth_branch_ratio: {ratios['same_depth_within_branch_to_across_branch_mean']:.4f}",
        f"- angular_branch_ratio: {ratios['angular_within_branch_to_across_branch_mean']:.4f}",
        f"- sibling_ratio: {ratios['sibling_to_non_sibling_mean']:.4f}",
        f"- positive_quantile_gaps: {radius['positive_adjacent_depth_quantile_gap_count']}",
        f"- min_quantile_gap: {radius['minimum_adjacent_quantile_gap']:.6f}",
        f"- radial_violation: {radius['parent_child_radial_violation_rate']:.4f}",
        f"- checkpoint: {record['checkpoint']}",
        "",
    ]


def write_candidate_summary(
    campaign_dir: Path,
    stage_name: str,
    records: list[dict[str, object]],
    skipped_reason: str | None = None,
) -> None:
    baseline = baseline_record()
    ranked = sorted(records, key=lambda item: gate_deficit_rank_key(item["metrics"]), reverse=True)
    out_json, out_md = stage_summary_paths(campaign_dir, stage_name)
    write_json(
        out_json,
        {
            "stage": stage_name,
            "skipped_reason": skipped_reason,
            "baseline": baseline,
            "run_count": len(ranked),
            "runs": ranked,
        },
    )
    lines = [f"# {stage_name} branch repair summary", ""]
    if skipped_reason:
        lines.extend([f"- skipped_reason: {skipped_reason}", ""])
    if baseline is not None:
        lines.extend(["# baseline", ""])
        lines.extend(summarize_candidate_lines(baseline))
    if ranked:
        lines.extend(["# ranked runs", ""])
        for record in ranked:
            lines.extend(summarize_candidate_lines(record))
    write_markdown(out_md, lines)


def base_params(gpu: int, init_checkpoint: Path, teacher_checkpoint: Path) -> dict[str, object]:
    return {
        "cpcc_weight": 0.05,
        "cpcc_min_group_size": 2,
        "radial_weight": 0.10,
        "radial_margin": 0.0005,
        "depth_band_weight": 0.0,
        "depth_quantile_weight": 0.3,
        "depth_quantile_margin": 0.001,
        "branch_weight": 0.0,
        "branch_cos_margin": 0.2,
        "branch_teacher_weight": 0.1,
        "branch_teacher_checkpoint": str(teacher_checkpoint),
        "branch_contrastive_weight": 0.1,
        "branch_contrastive_margin": 0.02,
        "branch_contrastive_hard_k": 0,
        "geometry_schedule": "constant",
        "selection_metric": "combined",
        "init_checkpoint": str(init_checkpoint),
        "lr": 0.01,
        "burnin": 20,
        "epochs": 150,
        "negs": 50,
        "dampening": 0.75,
        "dim": 10,
        "batchsize": 64,
        "gpu": gpu,
    }


def stage_b_configs(gpu: int) -> list[dict[str, object]]:
    init_checkpoint = default_repair_init_checkpoint()
    teacher_checkpoint = current_hybrid_checkpoint() or init_checkpoint
    base = base_params(gpu, init_checkpoint, teacher_checkpoint)
    configs = []
    for depth_margin in [0.0005, 0.0010, 0.0015]:
        for depth_weight in [0.1, 0.3, 0.6]:
            for teacher_weight in [0.1, 0.3]:
                for contrastive_weight in [0.0, 0.1, 0.3]:
                    for lr in [0.01, 0.03]:
                        params = dict(base)
                        params["depth_quantile_margin"] = depth_margin
                        params["depth_quantile_weight"] = depth_weight
                        params["branch_teacher_weight"] = teacher_weight
                        params["branch_contrastive_weight"] = contrastive_weight
                        params["lr"] = lr
                        configs.append(params)
    return configs


def load_stage_run_summaries(campaign_dir: Path, stage_name: str) -> list[dict[str, object]]:
    stage_dir = campaign_dir / "runs" / stage_name
    if not stage_dir.exists():
        return []
    return [load_json(path) for path in sorted(stage_dir.glob("*/run_summary.json"))]


def run_summary_record(summary: dict[str, object]) -> dict[str, object]:
    selected = summary["offline_rescore"]["selected_candidate"]
    return {
        "label": summary["run_name"],
        "checkpoint": selected["checkpoint"],
        "metrics": selected["metrics"],
        "run_dir": summary["run_dir"],
        "params": summary["params"],
    }


def best_records(campaign_dir: Path, stage_names: list[str], top_k: int) -> list[dict[str, object]]:
    records = []
    for stage_name in stage_names:
        for summary in load_stage_run_summaries(campaign_dir, stage_name):
            records.append(run_summary_record(summary))
    ranked = sorted(records, key=lambda item: gate_deficit_rank_key(item["metrics"]), reverse=True)
    return ranked[:top_k]


def stage_c_configs(campaign_dir: Path, gpu: int) -> tuple[list[dict[str, object]], str | None]:
    best_stage_b = best_records(campaign_dir, ["stageB"], top_k=1)
    if not best_stage_b:
        raise ValueError("Stage C requires completed Stage B results")
    best = best_stage_b[0]
    branch_ratio = best["metrics"]["ratios"]["within_branch_to_across_branch_mean"]
    if branch_ratio <= 0.35:
        return [], f"Stage B branch ratio already <= 0.35 ({branch_ratio:.4f})"

    teacher_checkpoint = current_hybrid_checkpoint() or Path(best["checkpoint"])
    configs = []
    for contrastive_weight in [0.5, 1.0]:
        params = dict(best["params"])
        params["epochs"] = 150
        params["gpu"] = gpu
        params["init_checkpoint"] = best["checkpoint"]
        params["branch_teacher_checkpoint"] = str(teacher_checkpoint)
        params["branch_contrastive_weight"] = contrastive_weight
        params["branch_contrastive_hard_k"] = 5
        configs.append(params)
    return configs, None


def stage_d_configs(campaign_dir: Path, gpu: int) -> list[dict[str, object]]:
    top_records = best_records(campaign_dir, ["stageB", "stageC"], top_k=5)
    if not top_records:
        raise ValueError("Stage D requires completed Stage B or Stage C results")
    teacher_checkpoint = current_hybrid_checkpoint() or Path(top_records[0]["checkpoint"])
    configs = []
    for record in top_records:
        params = dict(record["params"])
        params["epochs"] = 500
        params["gpu"] = gpu
        params["init_checkpoint"] = record["checkpoint"]
        params["branch_teacher_checkpoint"] = str(teacher_checkpoint)
        configs.append(params)
    return configs


def maybe_limit(configs: list[dict[str, object]], limit_runs: int) -> list[dict[str, object]]:
    return configs[:limit_runs] if limit_runs > 0 else configs


def maybe_override_epochs(configs: list[dict[str, object]], epochs_override: int) -> list[dict[str, object]]:
    if epochs_override <= 0:
        return configs
    overridden = []
    for params in configs:
        updated = dict(params)
        updated["epochs"] = epochs_override
        overridden.append(updated)
    return overridden


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
        str(params["selection_metric"]),
        "--eval-each",
        "5",
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
        "--branch-teacher-checkpoint",
        str(params["branch_teacher_checkpoint"]),
        "--branch-contrastive-weight",
        str(params["branch_contrastive_weight"]),
        "--branch-contrastive-margin",
        str(params["branch_contrastive_margin"]),
        "--branch-contrastive-hard-k",
        str(params["branch_contrastive_hard_k"]),
        "--geometry-schedule",
        str(params["geometry_schedule"]),
        "--init-checkpoint",
        str(params["init_checkpoint"]),
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
            "--full-eval-all",
            *rescore_floor_args(),
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


def run_stage_configs(
    stage_name: str,
    configs: list[dict[str, object]],
    campaign_dir: Path,
    env: dict[str, str],
    skip_existing: bool,
) -> list[dict[str, object]]:
    summaries = [run_single_training(stage_name, params, campaign_dir, env, skip_existing) for params in configs]
    records = [run_summary_record(summary) for summary in summaries]
    write_candidate_summary(campaign_dir, stage_name, records)
    return summaries


def run_stage_a(campaign_dir: Path) -> None:
    records = []
    geometry_summary_dir = EXPERIMENT_DIR / "tuning" / "geometry_separation" / "summaries"
    for stage_name in ["stage1", "stage2", "stage3"]:
        summary_path = geometry_summary_dir / f"{stage_name}_summary.json"
        if not summary_path.exists():
            continue
        summary = load_json(summary_path)
        if not summary.get("runs"):
            continue
        selected = summary["runs"][0]["offline_rescore"]["selected_candidate"]
        records.append(candidate_record(f"geometry_{stage_name}_best", Path(selected["checkpoint"])))
    write_candidate_summary(campaign_dir, "stageA", records)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run current-hybrid branch repair tuning for Disease-90 embeddings")
    parser.add_argument("--campaign-dir", type=Path, default=CAMPAIGN_DIR)
    parser.add_argument("--stages", nargs="+", choices=STAGE_NAMES, default=STAGE_NAMES)
    parser.add_argument("--limit-runs", type=int, default=0)
    parser.add_argument("--epochs-override", type=int, default=0)
    parser.add_argument("--skip-existing", action="store_true")
    parser.add_argument("--gpu", type=int, default=-1)
    args = parse_args_with_defaults(parser)

    env = os.environ.copy()
    mpl_dir = LOGS_DIR / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)

    if "stageA" in args.stages:
        run_stage_a(args.campaign_dir)
    if "stageB" in args.stages:
        run_stage_configs(
            "stageB",
            maybe_limit(maybe_override_epochs(stage_b_configs(args.gpu), args.epochs_override), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )
    if "stageC" in args.stages:
        configs, skipped_reason = stage_c_configs(args.campaign_dir, args.gpu)
        configs = maybe_override_epochs(configs, args.epochs_override)
        if skipped_reason:
            write_candidate_summary(args.campaign_dir, "stageC", [], skipped_reason=skipped_reason)
        else:
            run_stage_configs(
                "stageC",
                maybe_limit(configs, args.limit_runs),
                args.campaign_dir,
                env,
                args.skip_existing,
            )
    if "stageD" in args.stages:
        run_stage_configs(
            "stageD",
            maybe_limit(maybe_override_epochs(stage_d_configs(args.campaign_dir, args.gpu), args.epochs_override), args.limit_runs),
            args.campaign_dir,
            env,
            args.skip_existing,
        )


if __name__ == "__main__":
    main()
