#!/usr/bin/env python3

import argparse
from pathlib import Path

from disease90_common import (
    DATA_PATH,
    DEFAULT_METADATA_TSV,
    DEFAULT_NODE_LIST,
    DEFAULT_RELATIONS_CSV,
    DEFAULT_SUMMARY_JSON,
    ROOT_ID,
    build_relation_rows,
    build_subtree_metadata,
    parse_args_with_defaults,
    read_datacode_tsv,
    write_json,
    write_tsv,
)


def main() -> None:
    parser = argparse.ArgumentParser(description="Build disease-90 transitive-closure relations")
    parser.add_argument("--tsv", type=Path, default=DATA_PATH)
    parser.add_argument("--root-id", default=ROOT_ID)
    parser.add_argument("--mode", choices=["closure", "direct", "hybrid"], default="closure")
    parser.add_argument("--long-edge-stride", type=int, default=2)
    parser.add_argument("--out-csv", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--nodes-txt", type=Path, default=DEFAULT_NODE_LIST)
    parser.add_argument("--summary-json", type=Path, default=DEFAULT_SUMMARY_JSON)
    args = parse_args_with_defaults(parser)

    records = read_datacode_tsv(args.tsv)
    metadata_rows, ancestors, summary = build_subtree_metadata(records, args.root_id)
    relation_rows = build_relation_rows(
        metadata_rows,
        ancestors,
        mode=args.mode,
        long_edge_stride=args.long_edge_stride,
    )
    write_tsv(
        args.metadata_tsv,
        metadata_rows,
        [
            "node_id",
            "coding",
            "meaning",
            "parent_id",
            "depth",
            "top_branch_id",
            "top_branch_code",
            "selectable",
        ],
    )

    with args.out_csv.open("w", newline="", encoding="utf-8") as handle:
        handle.write("id1,id2,weight\n")
        for edge in relation_rows:
            handle.write(f"{edge['id1']},{edge['id2']},{edge['weight']}\n")

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
