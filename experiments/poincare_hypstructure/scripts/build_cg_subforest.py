#!/usr/bin/env python3

from __future__ import annotations

import argparse
import re
from collections import Counter, defaultdict, deque
from pathlib import Path

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
TARGET_CODE_PATTERN = re.compile(r"^[CG][0-9]")


def is_cg_target_code(record: dict[str, str]) -> bool:
    return bool(TARGET_CODE_PATTERN.match(record["coding"]))


def record_sort_key(node_id: str, order_index: dict[str, int]) -> tuple[int, str]:
    return order_index.get(node_id, 10**9), node_id


def build_cg_subforest_metadata(
    records: list[dict[str, str]],
    synthetic_root_id: str = CG_ROOT_ID,
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, object]]:
    node_map = {record["node_id"]: record for record in records}
    order_index = {record["node_id"]: index for index, record in enumerate(records)}
    if synthetic_root_id in node_map:
        raise ValueError(f"Synthetic root id conflicts with source node_id: {synthetic_root_id}")

    target_ids = {record["node_id"] for record in records if is_cg_target_code(record)}
    if not target_ids:
        raise ValueError("No C/G ICD target codes were found")

    included_ids = set(target_ids)
    for node_id in sorted(target_ids, key=lambda item: record_sort_key(item, order_index)):
        current = node_map[node_id]["parent_id"]
        while current and current != "0" and current in node_map:
            included_ids.add(current)
            current = node_map[current]["parent_id"]

    children: dict[str, list[str]] = defaultdict(list)
    for record in records:
        parent_id = record["parent_id"]
        if parent_id:
            children[parent_id].append(record["node_id"])

    natural_roots = sorted(
        [node_id for node_id in included_ids if node_map[node_id]["parent_id"] not in included_ids],
        key=lambda item: record_sort_key(item, order_index),
    )
    if not natural_roots:
        raise ValueError("C/G subforest has no natural roots")

    ordered_nodes = [synthetic_root_id]
    seen = {synthetic_root_id}
    queue = deque(natural_roots)
    while queue:
        node_id = queue.popleft()
        if node_id in seen or node_id not in included_ids:
            continue
        seen.add(node_id)
        ordered_nodes.append(node_id)
        for child_id in children.get(node_id, []):
            if child_id in included_ids:
                queue.append(child_id)

    missing = included_ids - set(ordered_nodes)
    if missing:
        raise ValueError(f"{len(missing)} included nodes were not reachable, example: {sorted(missing)[:5]}")

    parent_map = {synthetic_root_id: "0"}
    for node_id in ordered_nodes[1:]:
        parent_id = node_map[node_id]["parent_id"]
        parent_map[node_id] = parent_id if parent_id in included_ids else synthetic_root_id

    depth_map = {synthetic_root_id: 0}
    queue = deque([synthetic_root_id])
    child_map: dict[str, list[str]] = defaultdict(list)
    for node_id, parent_id in parent_map.items():
        if node_id != synthetic_root_id:
            child_map[parent_id].append(node_id)
    while queue:
        node_id = queue.popleft()
        for child_id in child_map.get(node_id, []):
            depth_map[child_id] = depth_map[node_id] + 1
            queue.append(child_id)

    top_branch_id = {synthetic_root_id: synthetic_root_id}
    for root_id in natural_roots:
        top_branch_id[root_id] = root_id
        queue = deque(child_map.get(root_id, []))
        while queue:
            node_id = queue.popleft()
            top_branch_id[node_id] = root_id
            for child_id in child_map.get(node_id, []):
                queue.append(child_id)

    ancestors: dict[str, set[str]] = {}
    for node_id in ordered_nodes:
        chain = set()
        current = parent_map[node_id]
        while current and current != "0":
            chain.add(current)
            current = parent_map[current]
        ancestors[node_id] = chain

    metadata_rows = [
        {
            "node_id": synthetic_root_id,
            "coding": CG_ROOT_CODE,
            "meaning": CG_ROOT_MEANING,
            "parent_id": "0",
            "depth": "0",
            "top_branch_id": synthetic_root_id,
            "top_branch_code": CG_ROOT_CODE,
            "selectable": "N",
            "is_target_code": "N",
        }
    ]
    for node_id in ordered_nodes[1:]:
        record = node_map[node_id]
        branch_id = top_branch_id[node_id]
        metadata_rows.append(
            {
                "node_id": node_id,
                "coding": record["coding"],
                "meaning": record["meaning"],
                "parent_id": parent_map[node_id],
                "depth": str(depth_map[node_id]),
                "top_branch_id": branch_id,
                "top_branch_code": node_map[branch_id]["coding"],
                "selectable": record["selectable"],
                "is_target_code": "Y" if node_id in target_ids else "N",
            }
        )

    direct_edge_count = len(metadata_rows) - 1
    closure_edge_count = sum(len(values) for node_id, values in ancestors.items() if node_id != synthetic_root_id)
    depth_counts = Counter(int(row["depth"]) for row in metadata_rows)
    target_records = [node_map[node_id] for node_id in target_ids]
    summary = {
        "source_data": str(DATA_PATH),
        "target_code_pattern": TARGET_CODE_PATTERN.pattern,
        "synthetic_root_id": synthetic_root_id,
        "target_code_count": len(target_ids),
        "C_count": sum(1 for record in target_records if record["coding"].startswith("C")),
        "G_count": sum(1 for record in target_records if record["coding"].startswith("G")),
        "real_node_count": len(included_ids),
        "synthetic_node_count": 1,
        "training_node_count": len(metadata_rows),
        "direct_edge_count": direct_edge_count,
        "closure_edge_count": closure_edge_count,
        "block_ancestor_count": sum(1 for node_id in included_ids if node_map[node_id]["coding"].startswith("Block ")),
        "chapter_ancestor_count": sum(
            1 for node_id in included_ids if node_map[node_id]["coding"].startswith("Chapter ")
        ),
        "natural_roots": [
            {
                "node_id": node_id,
                "coding": node_map[node_id]["coding"],
                "meaning": node_map[node_id]["meaning"],
            }
            for node_id in natural_roots
        ],
        "depth_counts": dict(sorted(depth_counts.items())),
        "top_branch_count": len(natural_roots),
    }
    return metadata_rows, ancestors, summary


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
            "is_target_code",
        ],
    )
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
