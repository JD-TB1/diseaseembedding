#!/usr/bin/env python3

from __future__ import annotations

import argparse
from pathlib import Path

from build_disease_subforest import build_prefix_subforest_metadata
from disease90_common import (
    DATA_PATH,
    EXPERIMENT_DIR,
    build_relation_rows,
    parse_args_with_defaults,
    read_datacode_tsv,
    write_json,
    write_tsv,
)


CG_RESULTS_DIR = EXPERIMENT_DIR / "results" / "disease_cg"
DEFAULT_CG_RELATIONS_CSV = CG_RESULTS_DIR / "cg_relations_direct.csv"
DEFAULT_CG_METADATA_TSV = CG_RESULTS_DIR / "cg_nodes.tsv"
DEFAULT_CG_NODE_LIST = CG_RESULTS_DIR / "cg_nodes.txt"
DEFAULT_CG_SUMMARY_JSON = CG_RESULTS_DIR / "cg_dataset_summary.json"
CG_ROOT_ID = "CG_ROOT"
CG_ROOT_CODE = "CG_ROOT"
CG_ROOT_MEANING = "Synthetic root for ICD C/G target subforest"
METADATA_FIELDS = [
    "node_id",
    "coding",
    "meaning",
    "parent_id",
    "depth",
    "top_branch_id",
    "top_branch_code",
    "selectable",
    "is_target_code",
]


def build_cg_subforest_metadata(
    records: list[dict[str, str]],
    synthetic_root_id: str = CG_ROOT_ID,
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, object]]:
    return build_prefix_subforest_metadata(
        records,
        ["C", "G"],
        root_mode="combined",
        synthetic_root_id=synthetic_root_id,
        synthetic_root_code=CG_ROOT_CODE,
        synthetic_root_meaning=CG_ROOT_MEANING,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the ICD C/G target subforest for Poincare training")
    parser.add_argument("--tsv", type=Path, default=DATA_PATH)
    parser.add_argument("--mode", choices=["closure", "direct", "hybrid"], default="direct")
    parser.add_argument("--long-edge-stride", type=int, default=2)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_CG_RELATIONS_CSV)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_CG_METADATA_TSV)
    parser.add_argument("--nodes-txt", type=Path, default=DEFAULT_CG_NODE_LIST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_CG_SUMMARY_JSON)
    args = parse_args_with_defaults(parser)

    records = read_datacode_tsv(args.tsv)
    metadata_rows, ancestors, summary = build_cg_subforest_metadata(records)
    relation_rows = build_relation_rows(
        metadata_rows,
        ancestors,
        mode=args.mode,
        long_edge_stride=args.long_edge_stride,
    )

    for path in (args.out_csv, args.metadata_tsv, args.nodes_txt, args.summary_json):
        path.parent.mkdir(parents=True, exist_ok=True)

    write_tsv(args.metadata_tsv, metadata_rows, METADATA_FIELDS)
    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        handle.write("id1,id2,weight\n")
        for row in relation_rows:
            handle.write(f"{row['id1']},{row['id2']},{row['weight']}\n")
    with args.nodes_txt.open("w", encoding="utf-8") as handle:
        for row in metadata_rows:
            handle.write(f"{row['node_id']}\n")

    summary["relation_mode"] = args.mode
    summary["training_edge_count"] = len(relation_rows)
    summary["long_edge_stride"] = args.long_edge_stride if args.mode == "hybrid" else None
    write_json(args.summary_json, summary)
    print(f"Wrote {len(metadata_rows)} nodes to {args.metadata_tsv}")
    print(f"Wrote {len(relation_rows)} {args.mode} edges to {args.out_csv}")
    print(f"Wrote dataset summary to {args.summary_json}")


if __name__ == "__main__":
    main()
