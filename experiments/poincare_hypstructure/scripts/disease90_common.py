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
EXPERIMENT_DIR = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_DIR / "data" / "datacode-19.tsv"
POINCARE_REF_DIR = PROJECT_DIR / "references" / "poincare-embeddings"
RESULTS_DIR = EXPERIMENT_DIR / "results" / "disease90"
PLOTS_DIR = RESULTS_DIR / "plots"
METADATA_DIR = EXPERIMENT_DIR / "metadata"
LOGS_DIR = EXPERIMENT_DIR / "logs"
ROOT_ID = "90"
DEFAULT_RELATIONS_CSV = RESULTS_DIR / "disease90_relations_direct.csv"
DEFAULT_METADATA_TSV = METADATA_DIR / "disease90_nodes.tsv"
DEFAULT_NODE_LIST = METADATA_DIR / "disease90_nodes.txt"
DEFAULT_SUMMARY_JSON = METADATA_DIR / "disease90_dataset_summary.json"
DEFAULT_CHECKPOINT = RESULTS_DIR / "disease90_embeddings_poincare_hypstructure.pth"
DEFAULT_TRAIN_CONFIG = RESULTS_DIR / "train_config.json"
DEFAULT_TRAIN_LOG = LOGS_DIR / "train_disease90_poincare_hypstructure.log"
DEFAULT_EXPORT_TSV = RESULTS_DIR / "embeddings.tsv"
DEFAULT_EMBED_STATS_JSON = RESULTS_DIR / "embedding_stats.json"
DEFAULT_EVAL_JSON = RESULTS_DIR / "eval_metrics.json"
DEFAULT_EVAL_MD = RESULTS_DIR / "eval_summary.md"


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
            if not node_id:
                continue
            records.append(
                {
                    "coding": str(row["coding"]).strip(),
                    "meaning": str(row["meaning"]).strip(),
                    "node_id": node_id,
                    "parent_id": str(row["parent_id"]).strip(),
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
    ordered_nodes: list[str] = []
    seen: set[str] = set()
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
        current = node_map[node_id]["parent_id"]
        chain: set[str] = set()
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
        metadata_rows.append(
            {
                "node_id": node_id,
                "coding": record["coding"],
                "meaning": record["meaning"],
                "parent_id": record["parent_id"],
                "depth": str(depths[node_id]),
                "top_branch_id": branch_id,
                "top_branch_code": node_map[branch_id]["coding"],
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
            parent_id = parent_map[node_id]
            extras = sorted(ancestor for ancestor in ancestor_ids if ancestor != parent_id)
            for index, ancestor_id in enumerate(extras):
                if index % max(long_edge_stride, 1) == 0:
                    selected_edges.add((node_id, ancestor_id))

    return [{"id1": left, "id2": right, "weight": 1.0} for left, right in sorted(selected_edges)]


def write_tsv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_json(path: Path, payload: object) -> None:
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)


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
    checkpoint = torch.load(checkpoint_path, map_location=map_location, weights_only=False)
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
    cleaned_objects = [item.decode("utf-8") if isinstance(item, bytes) else str(item) for item in objects]
    if embeddings.size(0) != len(cleaned_objects):
        raise ValueError(
            f"Embedding rows ({embeddings.size(0)}) do not match object count ({len(cleaned_objects)})"
        )
    return embeddings.detach().cpu().double().numpy(), cleaned_objects, checkpoint


def normalize_object_id(value: object) -> str:
    if isinstance(value, bytes):
        return value.decode("utf-8")
    return str(value)


def normalize_object_ids(values: list[object]) -> list[str]:
    return [normalize_object_id(value) for value in values]


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


def project_to_ball_torch(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    max_norm = 1.0 - eps
    norms = torch.linalg.vector_norm(values, dim=-1, keepdim=True).clamp_min(eps)
    scales = torch.clamp(max_norm / norms, max=1.0)
    return values * scales


def poincare_distance_torch(left: torch.Tensor, right: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    sq_left = torch.sum(left * left, dim=-1).clamp(min=0.0, max=1.0 - eps)
    sq_right = torch.sum(right * right, dim=-1).clamp(min=0.0, max=1.0 - eps)
    sq_dist = torch.sum((left - right) ** 2, dim=-1)
    x = 1.0 + 2.0 * sq_dist / (((1.0 - sq_left) * (1.0 - sq_right)).clamp_min(eps))
    z = torch.sqrt((x * x - 1.0).clamp_min(eps))
    return torch.log(x + z)


def condensed_poincare_distance_torch(values: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    if values.size(0) < 2:
        return values.new_empty((0,))
    sq_norms = torch.sum(values * values, dim=-1).clamp(min=0.0, max=1.0 - eps)
    sq_dist = torch.sum((values[:, None, :] - values[None, :, :]) ** 2, dim=-1)
    denom = ((1.0 - sq_norms)[:, None] * (1.0 - sq_norms)[None, :]).clamp_min(eps)
    x = 1.0 + 2.0 * sq_dist / denom
    z = torch.sqrt((x * x - 1.0).clamp_min(eps))
    distances = torch.log(x + z)
    tri = torch.triu_indices(values.size(0), values.size(0), offset=1, device=values.device)
    return distances[tri[0], tri[1]]


def poincare_mean_torch(values: torch.Tensor, dim: int = 0, c: float = 1.0, eps: float = 1e-5) -> torch.Tensor:
    denom = 1.0 + c * torch.sum(values * values, dim=-1, keepdim=True)
    klein = 2.0 * values / denom.clamp_min(eps)
    lorentz = 1.0 / torch.sqrt((1.0 - c * torch.sum(klein * klein, dim=-1, keepdim=True)).clamp_min(eps))
    mean = torch.sum(lorentz * klein, dim=dim, keepdim=True) / torch.sum(lorentz, dim=dim, keepdim=True).clamp_min(
        eps
    )
    poincare = mean / (1.0 + torch.sqrt((1.0 - c * torch.sum(mean * mean, dim=-1, keepdim=True)).clamp_min(eps)))
    return poincare.squeeze(dim)


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


def torch_pearson_corr(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    x0 = x - x.mean()
    y0 = y - y.mean()
    denom = torch.sqrt(torch.sum(x0 * x0) * torch.sum(y0 * y0)).clamp_min(eps)
    return torch.sum(x0 * y0) / denom


def average_precision(labels: np.ndarray, scores: np.ndarray) -> float:
    order = np.argsort(scores)[::-1]
    sorted_labels = labels[order]
    positives = int(sorted_labels.sum())
    if positives == 0:
        return float("nan")
    hits = 0
    precision_sum = 0.0
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
        selected = rng.choice(len(index_pairs), size=max_pairs, replace=False)
        pairs = [index_pairs[index] for index in selected]
    else:
        pairs = index_pairs
    distances = [
        float(poincare_distance_matrix(embeddings[left : left + 1], embeddings[right : right + 1])[0])
        for left, right in pairs
    ]
    return np.asarray(distances, dtype=np.float64)


def build_tree_helpers(
    metadata_rows: list[dict[str, str]],
) -> tuple[dict[str, str], dict[str, list[str]], dict[str, int], dict[str, list[str]], dict[str, list[str]]]:
    parent_map = {row["node_id"]: row["parent_id"] for row in metadata_rows}
    children = defaultdict(list)
    depth_map = {row["node_id"]: int(row["depth"]) for row in metadata_rows}
    for row in metadata_rows:
        if row["parent_id"] and row["parent_id"] != "0":
            children[row["parent_id"]].append(row["node_id"])

    path_to_root = {}
    for row in metadata_rows:
        node_id = row["node_id"]
        path = [node_id]
        current = parent_map[node_id]
        while current and current != "0":
            path.append(current)
            current = parent_map[current]
        path_to_root[node_id] = path

    descendants = {}
    for row in sorted(metadata_rows, key=lambda item: int(item["depth"]), reverse=True):
        node_id = row["node_id"]
        members = [node_id]
        for child_id in children.get(node_id, []):
            members.extend(descendants[child_id])
        descendants[node_id] = members

    return parent_map, children, depth_map, path_to_root, descendants


def tree_distance(node_a: str, node_b: str, depth_map: dict[str, int], path_to_root: dict[str, list[str]]) -> int:
    ancestors_a = set(path_to_root[node_a])
    lca = next(node for node in path_to_root[node_b] if node in ancestors_a)
    return depth_map[node_a] + depth_map[node_b] - 2 * depth_map[lca]


def depth_radius_from_checkpoint(checkpoint_path: Path, metadata_rows: list[dict[str, str]]) -> dict[str, float]:
    embeddings, objects, _ = checkpoint_embeddings_and_objects(checkpoint_path)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=np.float64)
    radii = np.linalg.norm(embeddings, axis=1)
    return {
        "pearson": pearson_corr(depths, radii),
        "spearman": spearman_corr(depths, radii),
    }


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
    subprocess.run([sys.executable, "setup.py", "build_ext", "--inplace"], cwd=POINCARE_REF_DIR, env=env, check=True)


def import_hype_modules():
    sys.path.insert(0, str(POINCARE_REF_DIR))
    from hype import build_model  # noqa: WPS433
    from hype.checkpoint import LocalCheckpoint  # noqa: WPS433
    from hype.graph import eval_reconstruction, load_edge_list  # noqa: WPS433
    from hype.graph_dataset import BatchedDataset  # noqa: WPS433
    from hype.rsgd import RiemannianSGD  # noqa: WPS433

    return build_model, LocalCheckpoint, load_edge_list, BatchedDataset, RiemannianSGD, eval_reconstruction
