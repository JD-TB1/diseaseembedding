#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from disease90_common import (
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    read_metadata_tsv,
    write_json,
    write_tsv,
)


RECOMMENDED_RUNTIME = "/opt/homebrew/Caskroom/miniforge/base/envs/logic-bank-benchmark/bin/python"


def load_json_if_present(path: Path | None) -> object | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def summary_prefix_label(summary: dict[str, object]) -> str:
    prefixes = summary.get("target_prefixes")
    if isinstance(prefixes, list) and prefixes:
        return "/".join(str(prefix) for prefix in prefixes)
    return "disease"


def write_labmate_readme(
    path: Path,
    dataset_label: str,
    file_prefix: str,
    code_only_path: Path,
    all_nodes_path: Path,
    manifest_path: Path,
    summary: dict[str, object],
    checkpoint_path: Path,
    combined: bool,
) -> None:
    prefix_label = summary_prefix_label(summary)
    usage_line = (
        f"Use `{code_only_path.name}` when downstream training needs C, G, and I disease codes in one shared coordinate system."
        if combined
        else f"Use `{code_only_path.name}` for downstream training limited to the {prefix_label} disease tree."
    )
    context_line = (
        f"`{all_nodes_path.name}` keeps the synthetic root, chapter nodes, block ancestors, and target codes for hierarchy context."
        if combined
        else f"`{all_nodes_path.name}` keeps the chapter root, block ancestors, and target codes for hierarchy context."
    )
    counts = summary.get("target_code_counts_by_prefix")
    count_lines = []
    if isinstance(counts, dict):
        for prefix, count in counts.items():
            count_lines.append(f"- {prefix} target codes: `{count}`")

    lines = [
        f"# {dataset_label} Disease Embeddings",
        "",
        usage_line,
        context_line,
        "",
        "## Files",
        "",
        f"- Primary training table: `{code_only_path.name}`",
        f"- Hierarchy/context table: `{all_nodes_path.name}`",
        f"- Machine-readable manifest: `{manifest_path.name}`",
        "",
        "## Source Data",
        "",
        f"- Raw hierarchy: `{summary.get('source_data')}`",
        f"- Target code regex: `{summary.get('target_code_pattern')}`",
        f"- Target prefixes: `{prefix_label}`",
        *count_lines,
        f"- Target disease-code rows: `{summary.get('target_code_count')}`",
        f"- Training nodes: `{summary.get('training_node_count')}`",
        f"- Direct training edges: `{summary.get('direct_edge_count')}`",
        f"- Root mode: `{summary.get('root_mode')}`",
        "",
        "## Embedding Source",
        "",
        f"- Final checkpoint: `{checkpoint_path}`",
        f"- File prefix: `{file_prefix}`",
        "- Recipe: current-hybrid baseline followed by Stage D branch-repair geometry",
        "  using the same-prefix baseline checkpoint as both initialization and branch teacher.",
        "",
        "## Columns",
        "",
        "`node_id`, `coding`, `meaning`, `parent_id`, `depth`, `top_branch_id`,",
        "`top_branch_code`, `selectable`, `is_target_code`, then `dim_1` through `dim_10`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labmate-ready disease embedding TSV files")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--metadata-tsv", type=Path, required=True)
    parser.add_argument("--dataset-summary-json", type=Path, required=True)
    parser.add_argument("--eval-json", type=Path, default=None)
    parser.add_argument("--offline-rescore-json", type=Path, default=None)
    parser.add_argument("--baseline-train-config", type=Path, default=None)
    parser.add_argument("--repair-train-config", type=Path, default=None)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--file-prefix", required=True)
    parser.add_argument("--dataset-label", required=True)
    parser.add_argument("--runtime", default=RECOMMENDED_RUNTIME)
    parser.add_argument("--command", default="")
    parser.add_argument("--expected-code-rows", type=int, default=None)
    parser.add_argument("--expected-all-node-rows", type=int, default=None)
    args = parse_args_with_defaults(parser)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, checkpoint = checkpoint_embeddings_and_objects(args.checkpoint)
    if embeddings.shape[0] != len(objects):
        raise ValueError("Embedding row count does not match checkpoint object count")

    missing = [node_id for node_id in objects if node_id not in metadata_map]
    if missing:
        raise ValueError(f"{len(missing)} checkpoint nodes are missing from metadata, example: {missing[:5]}")

    dim = embeddings.shape[1]
    dim_fields = [f"dim_{index + 1}" for index in range(dim)]
    fieldnames = [
        "node_id",
        "coding",
        "meaning",
        "parent_id",
        "depth",
        "top_branch_id",
        "top_branch_code",
        "selectable",
        "is_target_code",
        *dim_fields,
    ]

    rows = []
    for node_id, vector in zip(objects, embeddings):
        meta = metadata_map[node_id]
        row = {field: meta.get(field, "") for field in fieldnames if field in meta or field.startswith("dim_")}
        for index, value in enumerate(vector, start=1):
            row[f"dim_{index}"] = f"{float(value):.12g}"
        rows.append(row)

    code_rows = [row for row in rows if row["is_target_code"] == "Y"]
    code_only_path = args.out_dir / f"{args.file_prefix}_embeddings_codes_only.tsv"
    all_nodes_path = args.out_dir / f"{args.file_prefix}_embeddings_all_nodes.tsv"
    manifest_path = args.out_dir / "manifest.json"
    readme_path = args.out_dir / "README.md"

    write_tsv(code_only_path, code_rows, fieldnames)
    write_tsv(all_nodes_path, rows, fieldnames)

    numeric = embeddings.astype(np.float64)
    summary = load_json_if_present(args.dataset_summary_json) or {}
    if not isinstance(summary, dict):
        summary = {}
    expected_code_rows = args.expected_code_rows
    if expected_code_rows is None and summary.get("target_code_count") is not None:
        expected_code_rows = int(summary["target_code_count"])
    expected_all_node_rows = args.expected_all_node_rows
    if expected_all_node_rows is None and summary.get("training_node_count") is not None:
        expected_all_node_rows = int(summary["training_node_count"])

    coding_counts: dict[str, int] = {}
    for row in code_rows:
        coding_counts[row["coding"]] = coding_counts.get(row["coding"], 0) + 1
    duplicate_codings = sorted(coding for coding, count in coding_counts.items() if count > 1)
    off_prefix_targets = []
    prefixes = summary.get("target_prefixes")
    if isinstance(prefixes, list):
        allowed = tuple(str(prefix) for prefix in prefixes)
        off_prefix_targets = [row["coding"] for row in code_rows if not row["coding"].startswith(allowed)]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "dataset_label": args.dataset_label,
        "file_prefix": args.file_prefix,
        "runtime": args.runtime,
        "command": args.command,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metadata_tsv": str(args.metadata_tsv),
        "dataset_summary_json": str(args.dataset_summary_json),
        "eval_metrics_json": str(args.eval_json) if args.eval_json else None,
        "offline_rescore_json": str(args.offline_rescore_json) if args.offline_rescore_json else None,
        "primary_file": str(code_only_path),
        "secondary_file": str(all_nodes_path),
        "recommended_downstream_file": str(code_only_path),
        "context_file": str(all_nodes_path),
        "row_counts": {
            "code_only": len(code_rows),
            "all_nodes": len(rows),
            "expected_code_only": expected_code_rows,
            "expected_all_nodes": expected_all_node_rows,
            "dataset_target_code_count": summary.get("target_code_count"),
            "dataset_training_node_count": summary.get("training_node_count"),
        },
        "embedding": {
            "dim": dim,
            "finite": bool(np.isfinite(numeric).all()),
            "radius_min": float(np.linalg.norm(numeric, axis=1).min()),
            "radius_mean": float(np.linalg.norm(numeric, axis=1).mean()),
            "radius_max": float(np.linalg.norm(numeric, axis=1).max()),
        },
        "columns": fieldnames,
        "duplicate_code_only_codings": duplicate_codings,
        "off_prefix_target_codings": off_prefix_targets,
        "dataset_summary": summary,
        "eval_metrics": load_json_if_present(args.eval_json),
        "offline_rescore": load_json_if_present(args.offline_rescore_json),
        "baseline_train_config": load_json_if_present(args.baseline_train_config),
        "repair_train_config": load_json_if_present(args.repair_train_config),
    }
    write_json(manifest_path, manifest)
    write_labmate_readme(
        readme_path,
        args.dataset_label,
        args.file_prefix,
        code_only_path,
        all_nodes_path,
        manifest_path,
        summary,
        args.checkpoint,
        combined=summary.get("root_mode") == "combined",
    )

    if expected_code_rows is not None and len(code_rows) != expected_code_rows:
        raise ValueError(f"Expected {expected_code_rows} code rows, found {len(code_rows)}")
    if expected_all_node_rows is not None and len(rows) != expected_all_node_rows:
        raise ValueError(f"Expected {expected_all_node_rows} all-node rows, found {len(rows)}")
    if off_prefix_targets:
        raise ValueError(f"Found off-prefix target codings, example: {off_prefix_targets[:5]}")
    if not np.isfinite(numeric).all():
        raise ValueError("Embedding table contains non-finite values")

    print(f"Wrote code-only embeddings to {code_only_path}")
    print(f"Wrote all-node embeddings to {all_nodes_path}")
    print(f"Wrote labmate manifest to {manifest_path}")
    print(f"Wrote labmate README to {readme_path}")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"error: {exc}", file=sys.stderr)
        raise
