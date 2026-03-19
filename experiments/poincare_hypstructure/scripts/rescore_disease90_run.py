#!/usr/bin/env python3

from __future__ import annotations

import argparse
import ast
import json
import shutil
from pathlib import Path

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVAL_MD,
    DEFAULT_EVAL_JSON,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    write_json,
)
from disease90_metrics import (
    compute_disease90_metrics,
    compute_radius_structure_metrics,
    depth_first_rank_key,
    load_metadata_and_relations,
    passes_acceptance_floors,
)


def parse_training_log(log_path: Path) -> dict[int, dict[str, object]]:
    epoch_to_stats = {}
    if not log_path.exists():
        return epoch_to_stats
    for line in log_path.read_text(encoding="utf-8").splitlines():
        if not line.startswith("json_stats: "):
            continue
        raw_payload = line[len("json_stats: ") :].replace(", }", " }").replace(", ]", " ]")
        try:
            payload = json.loads(raw_payload)
        except json.JSONDecodeError:
            payload = ast.literal_eval(raw_payload)
        if "map_rank" in payload:
            epoch_to_stats[int(payload["epoch"])] = payload
    return epoch_to_stats


def checkpoint_epoch(path: Path) -> int | None:
    suffix = path.name.rsplit(".", 1)[-1]
    if suffix.isdigit():
        return int(suffix)
    return None


def preliminary_rank_key(record: dict[str, object]) -> tuple[float, ...]:
    reconstruction_map = float(record["reconstruction_map"])
    if reconstruction_map != reconstruction_map:
        reconstruction_map = float("-inf")
    return (
        1.0 if record["monotonic_depth_means"] else 0.0,
        float(record["minimum_adjacent_gap"]),
        float(record["depth_radius_spearman"]),
        -float(record["parent_child_radial_violation_rate"]),
        float(record["leaf_mean_radius"]),
        reconstruction_map,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline rescore disease-90 checkpoints using depth-first rules")
    parser.add_argument("--checkpoint-prefix", type=Path, default=DEFAULT_CHECKPOINT)
    parser.add_argument("--train-log", type=Path, required=True)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--relations-csv", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_EVAL_MD)
    parser.add_argument("--best-checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.offline_best"))
    parser.add_argument("--top-k-full-eval", type=int, default=10)
    parser.add_argument("--floor-reconstruction-map", type=float, default=0.30)
    parser.add_argument("--floor-parent-mean-rank", type=float, default=1.30)
    parser.add_argument("--floor-sibling-ratio", type=float, default=0.35)
    parser.add_argument("--floor-branch-ratio", type=float, default=0.50)
    args = parse_args_with_defaults(parser)

    metadata_rows, relations = load_metadata_and_relations(args.metadata_tsv, args.relations_csv)
    logged_stats = parse_training_log(args.train_log)
    numbered_snapshots = sorted(
        [path for path in args.checkpoint_prefix.parent.glob(f"{args.checkpoint_prefix.name}.*") if checkpoint_epoch(path) is not None],
        key=lambda path: int(checkpoint_epoch(path)),
    )
    snapshots = list(numbered_snapshots)
    if args.checkpoint_prefix.exists():
        snapshots.append(args.checkpoint_prefix)
    best_path = Path(f"{args.checkpoint_prefix}.best")
    if best_path.exists():
        snapshots.append(best_path)
    snapshots = list(dict.fromkeys(snapshots))
    if not snapshots:
        raise FileNotFoundError(f"No checkpoint snapshots found for prefix {args.checkpoint_prefix}")

    preliminary_records = []
    for snapshot in snapshots:
        epoch = checkpoint_epoch(snapshot)
        embeddings, objects, _ = checkpoint_embeddings_and_objects(snapshot)
        radius_structure = compute_radius_structure_metrics(embeddings, objects, metadata_rows)
        log_payload = logged_stats.get(epoch, {})
        preliminary_records.append(
            {
                "epoch": epoch if epoch is not None else -1,
                "checkpoint": str(snapshot),
                "reconstruction_map": float(log_payload.get("map_rank", float("nan"))),
                "reconstruction_mean_rank": float(log_payload.get("mean_rank", float("nan"))),
                "depth_radius_spearman": float(radius_structure["depth_radius"]["spearman"]),
                "depth_radius_pearson": float(radius_structure["depth_radius"]["pearson"]),
                "minimum_adjacent_gap": float(radius_structure["minimum_adjacent_gap"]),
                "monotonic_depth_means": bool(radius_structure["monotonic_depth_means"]),
                "parent_child_radial_violation_rate": float(radius_structure["parent_child_radial_violation_rate"]),
                "leaf_mean_radius": float(radius_structure["leaf_mean_radius"]),
                "internal_mean_radius": float(radius_structure["internal_mean_radius"]),
                "leaf_internal_radius_ratio": float(radius_structure["leaf_internal_radius_ratio"]),
                "mean_radius_by_depth": radius_structure["mean_radius_by_depth"],
                "adjacent_depth_gaps": radius_structure["adjacent_depth_gaps"],
            }
        )

    preliminary_records.sort(key=preliminary_rank_key, reverse=True)
    top_candidates = preliminary_records[: max(args.top_k_full_eval, 1)]
    required_candidates = {
        str(args.checkpoint_prefix),
        str(Path(f"{args.checkpoint_prefix}.best")),
    }
    indexed_candidates = {candidate["checkpoint"]: candidate for candidate in preliminary_records}
    for checkpoint_path in required_candidates:
        if checkpoint_path in indexed_candidates and indexed_candidates[checkpoint_path] not in top_candidates:
            top_candidates.append(indexed_candidates[checkpoint_path])

    full_candidates = []
    for candidate in top_candidates:
        metrics = compute_disease90_metrics(Path(candidate["checkpoint"]), metadata_rows, relations)
        metrics["checkpoint"] = candidate["checkpoint"]
        metrics["acceptance_pass"] = passes_acceptance_floors(
            metrics,
            floor_reconstruction_map=args.floor_reconstruction_map,
            floor_parent_mean_rank=args.floor_parent_mean_rank,
            floor_sibling_ratio=args.floor_sibling_ratio,
            floor_branch_ratio=args.floor_branch_ratio,
        )
        metrics["rank_key"] = depth_first_rank_key(metrics)
        full_candidates.append(metrics)

    feasible = [candidate for candidate in full_candidates if candidate["acceptance_pass"]]
    selected_pool = feasible if feasible else full_candidates
    selected = sorted(selected_pool, key=depth_first_rank_key, reverse=True)[0]
    selected_checkpoint = Path(selected["checkpoint"])

    args.best_checkpoint.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(selected_checkpoint, args.best_checkpoint)

    summary = {
        "checkpoint_prefix": str(args.checkpoint_prefix),
        "train_log": str(args.train_log),
        "candidate_count": len(preliminary_records),
        "preliminary_candidates": preliminary_records,
        "full_eval_candidates": full_candidates,
        "selected_candidate": {
            "checkpoint": str(selected_checkpoint),
            "epoch": selected["checkpoint_epoch"],
            "acceptance_pass": bool(selected["acceptance_pass"]),
            "rank_key": list(selected["rank_key"]),
            "metrics": selected,
        },
        "acceptance_floors": {
            "reconstruction_map": args.floor_reconstruction_map,
            "parent_mean_rank": args.floor_parent_mean_rank,
            "sibling_ratio": args.floor_sibling_ratio,
            "branch_ratio": args.floor_branch_ratio,
        },
    }
    write_json(args.out_json, summary)

    radius_structure = selected["radius_structure"]
    lines = [
        "# Offline Rescore Summary",
        "",
        f"- Selected checkpoint: {selected_checkpoint.name}",
        f"- Selected epoch: {selected['checkpoint_epoch']}",
        f"- Acceptance floors satisfied: {selected['acceptance_pass']}",
        f"- Reconstruction MAP: {selected['reconstruction']['map_rank']:.4f}",
        f"- Parent mean rank: {selected['parent_ranking']['mean_rank']:.4f}",
        f"- Depth/radius Spearman: {selected['depth_radius']['spearman']:.4f}",
        f"- Minimum adjacent depth gap: {radius_structure['minimum_adjacent_gap']:.6f}",
        f"- Parent-child radial violation rate: {radius_structure['parent_child_radial_violation_rate']:.4f}",
        f"- Leaf mean radius: {radius_structure['leaf_mean_radius']:.6f}",
        f"- Leaf/internal radius ratio: {radius_structure['leaf_internal_radius_ratio']:.4f}",
        "",
        "Mean radius by depth:",
    ]
    for depth, value in radius_structure["mean_radius_by_depth"].items():
        lines.append(f"- depth {depth}: {value:.6f}")
    args.out_md.write_text("\n".join(lines), encoding="utf-8")

    print(f"Wrote rescore summary to {args.out_json}")
    print(f"Wrote rescore report to {args.out_md}")
    print(f"Copied selected checkpoint to {args.best_checkpoint}")


if __name__ == "__main__":
    main()
