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


RESULTS_DIR = EXPERIMENT_DIR / "results"
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


def normalize_prefixes(values: list[str] | tuple[str, ...]) -> list[str]:
    prefixes: list[str] = []
    for value in values:
        for item in str(value).split(","):
            cleaned = item.strip().upper()
            if cleaned:
                prefixes.append(cleaned)
    prefixes = list(dict.fromkeys(prefixes))
    if not prefixes:
        raise ValueError("At least one ICD code prefix is required")
    invalid = [prefix for prefix in prefixes if not re.fullmatch(r"[A-Z]", prefix)]
    if invalid:
        raise ValueError(f"Prefixes must be one ICD letter each, got: {invalid}")
    return prefixes


def target_code_pattern(prefixes: list[str]) -> re.Pattern[str]:
    escaped = "".join(re.escape(prefix) for prefix in prefixes)
    return re.compile(rf"^[{escaped}][0-9]")


def default_file_prefix(prefixes: list[str]) -> str:
    return "".join(prefixes).lower()


def default_results_dir(prefixes: list[str]) -> Path:
    return RESULTS_DIR / f"disease_{default_file_prefix(prefixes)}"


def default_synthetic_root_id(prefixes: list[str]) -> str:
    return f"{''.join(prefixes)}_ROOT"


def default_synthetic_root_meaning(prefixes: list[str]) -> str:
    return f"Synthetic root for ICD {'/'.join(prefixes)} target subforest"


def record_sort_key(node_id: str, order_index: dict[str, int]) -> tuple[int, str]:
    return order_index.get(node_id, 10**9), node_id


def _target_prefix(record: dict[str, str], prefixes: list[str], pattern: re.Pattern[str]) -> str | None:
    coding = record["coding"]
    if not pattern.match(coding):
        return None
    for prefix in prefixes:
        if coding.startswith(prefix):
            return prefix
    return None


def _build_children(records: list[dict[str, str]]) -> dict[str, list[str]]:
    children: dict[str, list[str]] = defaultdict(list)
    for record in records:
        parent_id = record["parent_id"]
        if parent_id:
            children[parent_id].append(record["node_id"])
    return children


def _ordered_included_nodes(
    natural_roots: list[str],
    included_ids: set[str],
    children: dict[str, list[str]],
) -> list[str]:
    ordered_nodes: list[str] = []
    seen: set[str] = set()
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
    return ordered_nodes


def _assign_top_branches(
    root_ids: list[str],
    parent_map: dict[str, str],
    child_map: dict[str, list[str]],
    synthetic_root_id: str | None,
) -> dict[str, str]:
    top_branch_id: dict[str, str] = {}
    if synthetic_root_id is not None:
        top_branch_id[synthetic_root_id] = synthetic_root_id
        branch_roots = root_ids
        for root_id in root_ids:
            top_branch_id[root_id] = root_id
    else:
        if len(root_ids) != 1:
            raise ValueError(f"Individual subforest mode expects one natural root, found {root_ids}")
        root_id = root_ids[0]
        top_branch_id[root_id] = root_id
        branch_roots = child_map.get(root_id, [])
        for child_id in branch_roots:
            top_branch_id[child_id] = child_id

    queue = deque(branch_roots)
    while queue:
        node_id = queue.popleft()
        branch_id = top_branch_id[node_id]
        for child_id in child_map.get(node_id, []):
            top_branch_id[child_id] = branch_id
            queue.append(child_id)

    for node_id in parent_map:
        top_branch_id.setdefault(node_id, node_id)
    return top_branch_id


def build_prefix_subforest_metadata(
    records: list[dict[str, str]],
    prefixes: list[str] | tuple[str, ...],
    root_mode: str = "auto",
    synthetic_root_id: str | None = None,
    synthetic_root_code: str | None = None,
    synthetic_root_meaning: str | None = None,
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, object]]:
    prefixes = normalize_prefixes(list(prefixes))
    if root_mode == "auto":
        root_mode = "combined" if len(prefixes) > 1 else "individual"
    if root_mode not in {"individual", "combined"}:
        raise ValueError(f"Unsupported root_mode: {root_mode}")
    if root_mode == "individual" and len(prefixes) != 1:
        raise ValueError("Individual root mode requires exactly one target prefix")

    pattern = target_code_pattern(prefixes)
    node_map = {record["node_id"]: record for record in records}
    order_index = {record["node_id"]: index for index, record in enumerate(records)}

    target_ids: set[str] = set()
    target_counts_by_prefix = {prefix: 0 for prefix in prefixes}
    for record in records:
        prefix = _target_prefix(record, prefixes, pattern)
        if prefix is None:
            continue
        target_ids.add(record["node_id"])
        target_counts_by_prefix[prefix] += 1
    if not target_ids:
        raise ValueError(f"No ICD target codes were found for prefixes {prefixes}")

    included_ids = set(target_ids)
    for node_id in sorted(target_ids, key=lambda item: record_sort_key(item, order_index)):
        current = node_map[node_id]["parent_id"]
        while current and current != "0" and current in node_map:
            included_ids.add(current)
            current = node_map[current]["parent_id"]

    children = _build_children(records)
    natural_roots = sorted(
        [node_id for node_id in included_ids if node_map[node_id]["parent_id"] not in included_ids],
        key=lambda item: record_sort_key(item, order_index),
    )
    if not natural_roots:
        raise ValueError("Subforest has no natural roots")
    if root_mode == "individual" and len(natural_roots) != 1:
        roots = [
            {"node_id": node_id, "coding": node_map[node_id]["coding"], "meaning": node_map[node_id]["meaning"]}
            for node_id in natural_roots
        ]
        raise ValueError(f"Individual subforest expected one chapter root, found {roots}")

    synthetic_id = synthetic_root_id if root_mode == "combined" else None
    if synthetic_id is None and root_mode == "combined":
        synthetic_id = default_synthetic_root_id(prefixes)
    if synthetic_id is not None and synthetic_id in node_map:
        raise ValueError(f"Synthetic root id conflicts with source node_id: {synthetic_id}")
    synthetic_code = synthetic_root_code or synthetic_id
    synthetic_meaning = synthetic_root_meaning or default_synthetic_root_meaning(prefixes)

    ordered_real_nodes = _ordered_included_nodes(natural_roots, included_ids, children)
    ordered_nodes = ([synthetic_id] if synthetic_id is not None else []) + ordered_real_nodes

    parent_map: dict[str, str] = {}
    if synthetic_id is not None:
        parent_map[synthetic_id] = "0"
    for node_id in ordered_real_nodes:
        parent_id = node_map[node_id]["parent_id"]
        if parent_id in included_ids:
            parent_map[node_id] = parent_id
        elif synthetic_id is not None:
            parent_map[node_id] = synthetic_id
        else:
            parent_map[node_id] = "0"

    child_map: dict[str, list[str]] = defaultdict(list)
    for node_id in ordered_nodes:
        parent_id = parent_map[node_id]
        if parent_id and parent_id != "0":
            child_map[parent_id].append(node_id)

    depth_map: dict[str, int] = {}
    root_queue = [synthetic_id] if synthetic_id is not None else natural_roots
    for root_id in root_queue:
        if root_id is not None:
            depth_map[root_id] = 0
    queue = deque(root_queue)
    while queue:
        node_id = queue.popleft()
        if node_id is None:
            continue
        for child_id in child_map.get(node_id, []):
            depth_map[child_id] = depth_map[node_id] + 1
            queue.append(child_id)

    top_branch_id = _assign_top_branches(natural_roots, parent_map, child_map, synthetic_id)

    ancestors: dict[str, set[str]] = {}
    for node_id in ordered_nodes:
        chain = set()
        current = parent_map[node_id]
        while current and current != "0":
            chain.add(current)
            current = parent_map[current]
        ancestors[node_id] = chain

    metadata_rows: list[dict[str, str]] = []
    if synthetic_id is not None:
        metadata_rows.append(
            {
                "node_id": synthetic_id,
                "coding": synthetic_code or synthetic_id,
                "meaning": synthetic_meaning,
                "parent_id": "0",
                "depth": "0",
                "top_branch_id": synthetic_id,
                "top_branch_code": synthetic_code or synthetic_id,
                "selectable": "N",
                "is_target_code": "N",
            }
        )

    for node_id in ordered_real_nodes:
        record = node_map[node_id]
        branch_id = top_branch_id[node_id]
        branch_code = synthetic_code if branch_id == synthetic_id else node_map[branch_id]["coding"]
        metadata_rows.append(
            {
                "node_id": node_id,
                "coding": record["coding"],
                "meaning": record["meaning"],
                "parent_id": parent_map[node_id],
                "depth": str(depth_map[node_id]),
                "top_branch_id": branch_id,
                "top_branch_code": branch_code,
                "selectable": record["selectable"],
                "is_target_code": "Y" if node_id in target_ids else "N",
            }
        )

    direct_edge_count = len(metadata_rows) - 1
    closure_edge_count = sum(len(values) for node_id, values in ancestors.items() if parent_map[node_id] != "0")
    depth_counts = Counter(int(row["depth"]) for row in metadata_rows)
    if synthetic_id is None:
        root_id = natural_roots[0]
        top_branch_ids = {row["node_id"] for row in metadata_rows if row["parent_id"] == root_id}
    else:
        top_branch_ids = {row["node_id"] for row in metadata_rows if row["parent_id"] == synthetic_id}

    summary = {
        "source_data": str(DATA_PATH),
        "target_prefixes": prefixes,
        "target_code_pattern": pattern.pattern,
        "root_mode": root_mode,
        "root_id": synthetic_id if synthetic_id is not None else natural_roots[0],
        "synthetic_root_id": synthetic_id,
        "synthetic_node_count": 1 if synthetic_id is not None else 0,
        "target_code_count": len(target_ids),
        "target_code_counts_by_prefix": target_counts_by_prefix,
        **{f"{prefix}_count": count for prefix, count in target_counts_by_prefix.items()},
        "real_node_count": len(included_ids),
        "training_node_count": len(metadata_rows),
        "node_count": len(metadata_rows),
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
        "top_branch_count": len(top_branch_ids),
    }
    return metadata_rows, ancestors, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an ICD prefix target subforest for Poincare training")
    parser.add_argument("--tsv", type=Path, default=DATA_PATH)
    parser.add_argument("--prefixes", nargs="+", required=True, help="ICD letter prefixes, e.g. C G I or C,G,I")
    parser.add_argument("--root-mode", choices=["auto", "individual", "combined"], default="auto")
    parser.add_argument("--synthetic-root-id", default=None)
    parser.add_argument("--synthetic-root-code", default=None)
    parser.add_argument("--synthetic-root-meaning", default=None)
    parser.add_argument("--mode", choices=["closure", "direct", "hybrid"], default="direct")
    parser.add_argument("--long-edge-stride", type=int, default=2)
    parser.add_argument("--out-csv", type=Path, default=None)
    parser.add_argument("--metadata-tsv", type=Path, default=None)
    parser.add_argument("--nodes-txt", type=Path, default=None)
    parser.add_argument("--summary-json", type=Path, default=None)
    args = parse_args_with_defaults(parser)

    prefixes = normalize_prefixes(args.prefixes)
    file_prefix = default_file_prefix(prefixes)
    results_dir = default_results_dir(prefixes)
    args.out_csv = args.out_csv or results_dir / f"{file_prefix}_relations_direct.csv"
    args.metadata_tsv = args.metadata_tsv or results_dir / f"{file_prefix}_nodes.tsv"
    args.nodes_txt = args.nodes_txt or results_dir / f"{file_prefix}_nodes.txt"
    args.summary_json = args.summary_json or results_dir / f"{file_prefix}_dataset_summary.json"

    records = read_datacode_tsv(args.tsv)
    metadata_rows, ancestors, summary = build_prefix_subforest_metadata(
        records,
        prefixes,
        root_mode=args.root_mode,
        synthetic_root_id=args.synthetic_root_id,
        synthetic_root_code=args.synthetic_root_code,
        synthetic_root_meaning=args.synthetic_root_meaning,
    )
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
