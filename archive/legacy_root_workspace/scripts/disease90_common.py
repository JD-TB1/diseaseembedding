#!/usr/bin/env python3

import argparse
import csv
import json
import math
import os
import subprocess
import sys
import sysconfig
from collections import Counter, defaultdict, deque
from pathlib import Path

import numpy as np
import torch


PROJECT_DIR = Path(__file__).resolve().parents[3]
WORKSPACE_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "datacode-19.tsv"
POINCARE_REF_DIR = PROJECT_DIR / "references" / "poincare-embeddings"
RESULTS_DIR = WORKSPACE_DIR / "results" / "disease90"
PLOTS_DIR = RESULTS_DIR / "plots"
METADATA_DIR = WORKSPACE_DIR / "metadata"
LOGS_DIR = WORKSPACE_DIR / "logs"
ROOT_ID = "90"
DEFAULT_CHECKPOINT = RESULTS_DIR / "disease90_embeddings.pth"
DEFAULT_RELATIONS_CSV = RESULTS_DIR / "disease90_relations_closure.csv"
DEFAULT_METADATA_TSV = METADATA_DIR / "disease90_nodes.tsv"
DEFAULT_NODE_LIST = METADATA_DIR / "disease90_nodes.txt"
DEFAULT_SUMMARY_JSON = METADATA_DIR / "disease90_dataset_summary.json"
DEFAULT_EXPORT_TSV = RESULTS_DIR / "embeddings.tsv"
DEFAULT_EMBED_STATS_JSON = RESULTS_DIR / "embedding_stats.json"
DEFAULT_EVAL_JSON = RESULTS_DIR / "eval_metrics.json"
DEFAULT_EVAL_MD = RESULTS_DIR / "eval_summary.md"
DEFAULT_TRAIN_CONFIG = RESULTS_DIR / "train_config.json"
DEFAULT_TRAIN_LOG = LOGS_DIR / "train_disease90.log"


def ensure_dirs() -> None:
    for path in (RESULTS_DIR, PLOTS_DIR, METADATA_DIR, LOGS_DIR):
        path.mkdir(parents=True, exist_ok=True)


def parse_args_with_defaults(parser: argparse.ArgumentParser) -> argparse.Namespace:
    ensure_dirs()
    return parser.parse_args()


def read_datacode_tsv(tsv_path: Path) -> list[dict[str, str]]:
    records = []
    with tsv_path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        required = {"coding", "meaning", "node_id", "parent_id", "selectable"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in {tsv_path}: {sorted(missing)}")
        for row in reader:
            node_id = str(row["node_id"]).strip()
            parent_id = str(row["parent_id"]).strip()
            if not node_id:
                continue
            records.append(
                {
                    "coding": str(row["coding"]).strip(),
                    "meaning": str(row["meaning"]).strip(),
                    "node_id": node_id,
                    "parent_id": parent_id,
                    "selectable": str(row["selectable"]).strip(),
                }
            )
    return records


def build_subtree_metadata(
    records: list[dict[str, str]], root_id: str
) -> tuple[list[dict[str, str]], dict[str, set[str]], dict[str, object]]:
    node_map = {record["node_id"]: record for record in records}
    if root_id not in node_map:
        raise ValueError(f"Root node_id={root_id} not found in dataset")

    children: dict[str, list[str]] = defaultdict(list)
    for record in records:
        parent_id = record["parent_id"]
        if parent_id:
            children[parent_id].append(record["node_id"])

    queue = deque([root_id])
    seen: set[str] = set()
    ordered_nodes: list[str] = []
    while queue:
        node_id = queue.popleft()
        if node_id in seen:
            continue
        seen.add(node_id)
        ordered_nodes.append(node_id)
        for child_id in children.get(node_id, []):
            queue.append(child_id)

    subtree_set = set(ordered_nodes)
    depths = {root_id: 0}
    queue = deque([root_id])
    while queue:
        node_id = queue.popleft()
        for child_id in children.get(node_id, []):
            if child_id in subtree_set and child_id not in depths:
                depths[child_id] = depths[node_id] + 1
                queue.append(child_id)

    top_branch_id = {root_id: root_id}
    queue = deque(children.get(root_id, []))
    for child_id in children.get(root_id, []):
        if child_id in subtree_set:
            top_branch_id[child_id] = child_id
    while queue:
        node_id = queue.popleft()
        for child_id in children.get(node_id, []):
            if child_id not in subtree_set:
                continue
            top_branch_id[child_id] = top_branch_id[node_id]
            queue.append(child_id)

    ancestors: dict[str, set[str]] = {}
    for node_id in ordered_nodes:
        chain: set[str] = set()
        current = node_map[node_id]["parent_id"]
        while current and current != "0" and current in subtree_set:
            chain.add(current)
            current = node_map[current]["parent_id"]
        ancestors[node_id] = chain

    direct_edges = set()
    closure_edges = set()
    for node_id in ordered_nodes:
        if node_id == root_id:
            continue
        parent_id = node_map[node_id]["parent_id"]
        if parent_id and parent_id in subtree_set:
            direct_edges.add((node_id, parent_id))
        for ancestor_id in ancestors[node_id]:
            closure_edges.add((node_id, ancestor_id))

    metadata_rows = []
    for node_id in ordered_nodes:
        record = node_map[node_id]
        branch_id = top_branch_id[node_id]
        branch_code = node_map[branch_id]["coding"] if branch_id in node_map else ""
        metadata_rows.append(
            {
                "node_id": node_id,
                "coding": record["coding"],
                "meaning": record["meaning"],
                "parent_id": record["parent_id"],
                "depth": str(depths[node_id]),
                "top_branch_id": branch_id,
                "top_branch_code": branch_code,
                "selectable": record["selectable"],
            }
        )

    child_counts = Counter()
    for node_id in ordered_nodes:
        for child_id in children.get(node_id, []):
            if child_id in subtree_set:
                child_counts[node_id] += 1

    depth_counts = Counter(int(row["depth"]) for row in metadata_rows)
    summary = {
        "root_id": root_id,
        "node_count": len(metadata_rows),
        "direct_edge_count": len(direct_edges),
        "closure_edge_count": len(closure_edges),
        "leaf_count": sum(1 for row in metadata_rows if child_counts[row["node_id"]] == 0),
        "internal_node_count": sum(1 for row in metadata_rows if child_counts[row["node_id"]] > 0),
        "max_depth": max(int(row["depth"]) for row in metadata_rows),
        "depth_counts": dict(sorted(depth_counts.items())),
        "top_branch_count": len({row["top_branch_id"] for row in metadata_rows if row["node_id"] != root_id}),
    }
    return metadata_rows, ancestors, summary


def build_relation_rows(
    metadata_rows: list[dict[str, str]],
    ancestors: dict[str, set[str]],
    mode: str,
    long_edge_stride: int = 2,
) -> list[dict[str, object]]:
    if mode not in {"closure", "direct", "hybrid"}:
        raise ValueError(f"Unsupported relation mode: {mode}")

    parent_map = {row["node_id"]: row["parent_id"] for row in metadata_rows}
    direct_edges: set[tuple[str, str]] = set()
    closure_edges: set[tuple[str, str]] = set()
    for node_id, ancestor_ids in ancestors.items():
        parent_id = parent_map[node_id]
        if parent_id and parent_id != "0":
            direct_edges.add((node_id, parent_id))
        for ancestor_id in ancestor_ids:
            closure_edges.add((node_id, ancestor_id))

    if mode == "direct":
        selected_edges = direct_edges
    elif mode == "closure":
        selected_edges = closure_edges
    else:
        selected_edges = set(direct_edges)
        for node_id, ancestor_ids in ancestors.items():
            extra_ancestors = sorted(
                ancestor_id for ancestor_id in ancestor_ids if ancestor_id != parent_map[node_id]
            )
            for index, ancestor_id in enumerate(extra_ancestors):
                if index % max(long_edge_stride, 1) == 0:
                    selected_edges.add((node_id, ancestor_id))

    rows = [{"id1": left, "id2": right, "weight": 1.0} for left, right in sorted(selected_edges)]
    return rows


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_metadata_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle, delimiter="\t")
        return [{key: value for key, value in row.items()} for row in reader]


def read_relations_csv(path: Path) -> list[tuple[str, str, float]]:
    rows = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"id1", "id2", "weight"}
        missing = required - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing columns in relations csv: {sorted(missing)}")
        for row in reader:
            rows.append((str(row["id1"]), str(row["id2"]), float(row["weight"])))
    return rows


def load_checkpoint(checkpoint_path: Path, map_location: str = "cpu") -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=map_location)
    if "embeddings" not in checkpoint:
        raise ValueError(f"Checkpoint {checkpoint_path} does not contain embeddings")
    return checkpoint


def checkpoint_embeddings_and_objects(
    checkpoint_path: Path,
) -> tuple[np.ndarray, list[str], dict[str, object]]:
    checkpoint = load_checkpoint(checkpoint_path)
    embeddings = checkpoint["embeddings"]
    if not isinstance(embeddings, torch.Tensor):
        embeddings = torch.tensor(embeddings)
    objects = checkpoint.get("objects")
    if objects is None:
        raise ValueError("Checkpoint is missing objects list")
    cleaned_objects = []
    for item in objects:
        cleaned_objects.append(item.decode("utf-8") if isinstance(item, bytes) else str(item))
    if embeddings.size(0) != len(cleaned_objects):
        raise ValueError(
            f"Embedding rows ({embeddings.size(0)}) do not match object count ({len(cleaned_objects)})"
        )
    return embeddings.detach().cpu().double().numpy(), cleaned_objects, checkpoint


def poincare_distance_matrix(source: np.ndarray, targets: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    source = np.asarray(source, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    sq_source = np.clip(np.sum(source * source, axis=-1, keepdims=True), 0.0, 1.0 - eps)
    sq_targets = np.clip(np.sum(targets * targets, axis=-1), 0.0, 1.0 - eps)
    diff = targets - source
    sq_dist = np.sum(diff * diff, axis=-1)
    x = 1.0 + 2.0 * sq_dist / np.maximum((1.0 - sq_source.squeeze(-1)) * (1.0 - sq_targets), eps)
    z = np.sqrt(np.maximum(x * x - 1.0, eps))
    return np.log(x + z)


def pearson_corr(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = math.sqrt(float(np.sum(x0 * x0) * np.sum(y0 * y0)))
    if denom == 0.0:
        return float("nan")
    return float(np.sum(x0 * y0) / denom)


def rank_array(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    i = 0
    while i < values.size:
        j = i + 1
        while j < values.size and values[order[j]] == values[order[i]]:
            j += 1
        rank = (i + j - 1) / 2.0 + 1.0
        ranks[order[i:j]] = rank
        i = j
    return ranks


def spearman_corr(x: np.ndarray, y: np.ndarray) -> float:
    return pearson_corr(rank_array(x), rank_array(y))


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = int(sorted_labels.sum())
    if positives == 0:
        return float("nan")
    precision_sum = 0.0
    hits = 0
    for idx, label in enumerate(sorted_labels, start=1):
        if label:
            hits += 1
            precision_sum += hits / idx
    return precision_sum / positives


def sample_pairwise_distances(
    embeddings: np.ndarray,
    index_pairs: list[tuple[int, int]],
    max_pairs: int = 5000,
    seed: int = 42,
) -> np.ndarray:
    if not index_pairs:
        return np.empty(0, dtype=np.float64)
    rng = np.random.default_rng(seed)
    if len(index_pairs) > max_pairs:
        chosen = rng.choice(len(index_pairs), size=max_pairs, replace=False)
        pairs = [index_pairs[index] for index in chosen]
    else:
        pairs = index_pairs
    distances = [
        float(poincare_distance_matrix(embeddings[i : i + 1], embeddings[j : j + 1])[0]) for i, j in pairs
    ]
    return np.asarray(distances, dtype=np.float64)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


def build_train_command(
    dataset_path: Path,
    checkpoint_path: Path,
    dim: int,
    epochs: int,
    lr: float,
    negs: int,
    batchsize: int,
    burnin: int,
    dampening: float,
    ndproc: int,
    eval_each: int,
    train_threads: int,
    gpu: int,
    sparse: bool,
    fresh: bool,
) -> list[str]:
    cmd = [
        sys.executable,
        str(POINCARE_REF_DIR / "embed.py"),
        "-dset",
        str(dataset_path),
        "-checkpoint",
        str(checkpoint_path),
        "-manifold",
        "poincare",
        "-model",
        "distance",
        "-eval",
        "reconstruction",
        "-dim",
        str(dim),
        "-epochs",
        str(epochs),
        "-lr",
        str(lr),
        "-negs",
        str(negs),
        "-batchsize",
        str(batchsize),
        "-burnin",
        str(burnin),
        "-dampening",
        str(dampening),
        "-ndproc",
        str(ndproc),
        "-eval_each",
        str(eval_each),
        "-train_threads",
        str(train_threads),
        "-gpu",
        str(gpu),
    ]
    if sparse:
        cmd.append("-sparse")
    if fresh:
        cmd.append("-fresh")
    return cmd


def compiled_extension_paths() -> list[Path]:
    suffix = sysconfig.get_config_var("EXT_SUFFIX") or ".so"
    return [
        POINCARE_REF_DIR / "hype" / f"graph_dataset{suffix}",
        POINCARE_REF_DIR / "hype" / f"adjacency_matrix_dataset{suffix}",
    ]


def ensure_poincare_reference_built() -> None:
    missing = [path for path in compiled_extension_paths() if not path.exists()]
    if not missing:
        return
    env = os.environ.copy()
    mpl_dir = LOGS_DIR / "mplconfig"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    env["MPLCONFIGDIR"] = str(mpl_dir)
    cmd = [sys.executable, "setup.py", "build_ext", "--inplace"]
    subprocess.run(cmd, cwd=POINCARE_REF_DIR, env=env, check=True)


def import_hype_modules():
    sys.path.insert(0, str(POINCARE_REF_DIR))
    from hype import build_model  # noqa: WPS433
    from hype.graph import eval_reconstruction  # noqa: WPS433

    return build_model, eval_reconstruction
