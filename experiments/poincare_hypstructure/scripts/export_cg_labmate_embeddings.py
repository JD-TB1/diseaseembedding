#!/usr/bin/env python3

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from build_cg_subforest import DEFAULT_CG_METADATA_TSV, DEFAULT_CG_SUMMARY_JSON
from disease90_common import (
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    read_metadata_tsv,
    write_json,
    write_tsv,
)


CG_RESULTS_DIR = Path(__file__).resolve().parents[1] / "results" / "disease_cg"
DEFAULT_LABMATE_DIR = CG_RESULTS_DIR / "labmate"
DEFAULT_CG_CHECKPOINT = CG_RESULTS_DIR / "cg_stage_d_repair.offline_best.pth"
RECOMMENDED_RUNTIME = "/opt/homebrew/Caskroom/miniforge/base/envs/logic-bank-benchmark/bin/python"


def load_json_if_present(path: Path | None) -> object | None:
    if path is None or not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_labmate_readme(
    path: Path,
    code_only_path: Path,
    all_nodes_path: Path,
    manifest_path: Path,
    summary: dict[str, object],
    checkpoint_path: Path,
) -> None:
    lines = [
        "# C/G Disease Embeddings",
        "",
        "Use `cg_embeddings_codes_only.tsv` for downstream feature-selection training.",
        "`cg_embeddings_all_nodes.tsv` is provided only when hierarchy context or plotting needs",
        "the synthetic root, chapter nodes, and block ancestors.",
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
        f"- C target codes: `{summary.get('C_count')}`",
        f"- G target codes: `{summary.get('G_count')}`",
        f"- Target disease-code rows: `{summary.get('target_code_count')}`",
        f"- Training nodes including synthetic root: `{summary.get('training_node_count')}`",
        f"- Direct training edges: `{summary.get('direct_edge_count')}`",
        "",
        "## Embedding Source",
        "",
        f"- Final checkpoint: `{checkpoint_path}`",
        "- Recipe: C/G current-hybrid baseline followed by Stage D branch-repair geometry",
        "  using the C/G baseline checkpoint as both initialization and branch teacher.",
        "",
        "## Columns",
        "",
        "`node_id`, `coding`, `meaning`, `parent_id`, `depth`, `top_branch_id`,",
        "`top_branch_code`, `selectable`, `is_target_code`, then `dim_1` through `dim_10`.",
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Export labmate-ready C/G embedding TSV files")
    parser.add_argument("--checkpoint", type=Path, default=DEFAULT_CG_CHECKPOINT)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_CG_METADATA_TSV)
    parser.add_argument("--dataset-summary-json", type=Path, default=DEFAULT_CG_SUMMARY_JSON)
    parser.add_argument("--eval-json", type=Path, default=CG_RESULTS_DIR / "eval_metrics.json")
    parser.add_argument("--offline-rescore-json", type=Path, default=CG_RESULTS_DIR / "offline_rescore.json")
    parser.add_argument("--baseline-train-config", type=Path, default=CG_RESULTS_DIR / "baseline_train_config.json")
    parser.add_argument("--repair-train-config", type=Path, default=CG_RESULTS_DIR / "repair_train_config.json")
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_LABMATE_DIR)
    parser.add_argument("--runtime", default=RECOMMENDED_RUNTIME)
    parser.add_argument("--command", default="")
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
        row = {field: meta[field] for field in fieldnames if field in meta}
        for index, value in enumerate(vector, start=1):
            row[f"dim_{index}"] = f"{float(value):.12g}"
        rows.append(row)

    code_rows = [row for row in rows if row["is_target_code"] == "Y"]
    code_only_path = args.out_dir / "cg_embeddings_codes_only.tsv"
    all_nodes_path = args.out_dir / "cg_embeddings_all_nodes.tsv"
    manifest_path = args.out_dir / "manifest.json"
    readme_path = args.out_dir / "README.md"

    write_tsv(code_only_path, code_rows, fieldnames)
    write_tsv(all_nodes_path, rows, fieldnames)

    numeric = embeddings.astype(np.float64)
    coding_counts: dict[str, int] = {}
    for row in code_rows:
        coding_counts[row["coding"]] = coding_counts.get(row["coding"], 0) + 1
    duplicate_codings = sorted(coding for coding, count in coding_counts.items() if count > 1)
    summary = load_json_if_present(args.dataset_summary_json) or {}
    if not isinstance(summary, dict):
        summary = {}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).replace(microsecond=0).isoformat(),
        "runtime": args.runtime,
        "command": args.command,
        "checkpoint": str(args.checkpoint),
        "checkpoint_epoch": checkpoint.get("epoch"),
        "metadata_tsv": str(args.metadata_tsv),
        "dataset_summary_json": str(args.dataset_summary_json),
        "eval_metrics_json": str(args.eval_json),
        "offline_rescore_json": str(args.offline_rescore_json),
        "primary_file": str(code_only_path),
        "secondary_file": str(all_nodes_path),
        "recommended_downstream_file": str(code_only_path),
        "context_file": str(all_nodes_path),
        "row_counts": {
            "code_only": len(code_rows),
            "all_nodes": len(rows),
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
        "dataset_summary": summary,
        "eval_metrics": load_json_if_present(args.eval_json),
        "offline_rescore": load_json_if_present(args.offline_rescore_json),
        "baseline_train_config": load_json_if_present(args.baseline_train_config),
        "repair_train_config": load_json_if_present(args.repair_train_config),
    }
    write_json(manifest_path, manifest)
    write_labmate_readme(readme_path, code_only_path, all_nodes_path, manifest_path, summary, args.checkpoint)

    if len(code_rows) != 955:
        raise ValueError(f"Expected 955 C/G code rows, found {len(code_rows)}")
    if len(rows) != 984:
        raise ValueError(f"Expected 984 all-node rows, found {len(rows)}")
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
