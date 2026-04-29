#!/usr/bin/env python3

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

from disease90_common import (
    average_precision,
    checkpoint_embeddings_and_objects,
    import_hype_modules,
    pearson_corr,
    poincare_distance_matrix,
    read_metadata_tsv,
    read_relations_csv,
    sample_pairwise_distances,
    spearman_corr,
)


def summarize(values: np.ndarray) -> dict[str, float]:
    if values.size == 0:
        return {"count": 0, "mean": float("nan"), "median": float("nan"), "min": float("nan"), "max": float("nan")}
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def safe_ratio(numerator: float, denominator: float) -> float:
    if denominator == 0.0 or np.isnan(denominator):
        return float("nan")
    return float(numerator / denominator)


def pairwise_poincare_distances(embeddings: np.ndarray) -> np.ndarray:
    rows = []
    for index in range(embeddings.shape[0]):
        rows.append(poincare_distance_matrix(embeddings[index : index + 1], embeddings))
    return np.vstack(rows)


def project_to_ball_np(values: np.ndarray, eps: float = 1e-5) -> np.ndarray:
    norms = np.linalg.norm(values, axis=-1, keepdims=True)
    scales = np.minimum(1.0, (1.0 - eps) / np.maximum(norms, eps))
    return values * scales


def sample_angular_distances(
    embeddings: np.ndarray,
    index_pairs: list[tuple[int, int]],
    max_pairs: int = 4000,
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
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    directions = embeddings / np.maximum(norms, 1e-8)
    distances = [1.0 - float(np.dot(directions[left], directions[right])) for left, right in pairs]
    return np.asarray(distances, dtype=np.float64)


def compute_radius_structure_metrics(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_rows: list[dict[str, str]],
) -> dict[str, object]:
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    radii = np.linalg.norm(embeddings, axis=1)
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=np.float64)

    mean_radius_by_depth = {}
    for depth in sorted({int(row["depth"]) for row in metadata_rows}):
        indices = [node_to_index[node_id] for node_id in objects if int(metadata_map[node_id]["depth"]) == depth]
        if indices:
            mean_radius_by_depth[str(depth)] = float(radii[indices].mean())

    adjacent_depth_gaps = {}
    adjacent_depth_quantile_gaps = {}
    depth_overlap_rates = {}
    ordered_depths = sorted(int(depth) for depth in mean_radius_by_depth)
    for left_depth, right_depth in zip(ordered_depths[:-1], ordered_depths[1:]):
        gap = mean_radius_by_depth[str(right_depth)] - mean_radius_by_depth[str(left_depth)]
        adjacent_depth_gaps[f"{left_depth}_to_{right_depth}"] = float(gap)
        left_indices = [node_to_index[node_id] for node_id in objects if int(metadata_map[node_id]["depth"]) == left_depth]
        right_indices = [node_to_index[node_id] for node_id in objects if int(metadata_map[node_id]["depth"]) == right_depth]
        left_radii = radii[left_indices]
        right_radii = radii[right_indices]
        quantile_gap = float(np.quantile(right_radii, 0.10) - np.quantile(left_radii, 0.90))
        adjacent_depth_quantile_gaps[f"{left_depth}_to_{right_depth}"] = quantile_gap
        overlap_rate = float(np.mean(right_radii[:, None] <= left_radii[None, :]))
        depth_overlap_rates[f"{left_depth}_to_{right_depth}"] = overlap_rate

    minimum_adjacent_gap = (
        float(min(adjacent_depth_gaps.values())) if adjacent_depth_gaps else float("nan")
    )
    minimum_adjacent_quantile_gap = (
        float(min(adjacent_depth_quantile_gaps.values())) if adjacent_depth_quantile_gaps else float("nan")
    )
    monotonic_depth_means = bool(adjacent_depth_gaps) and all(gap > 0.0 for gap in adjacent_depth_gaps.values())
    positive_adjacent_depth_quantile_gap_count = sum(
        1 for gap in adjacent_depth_quantile_gaps.values() if gap > 0.0
    )

    parent_child_violations = 0
    parent_child_pairs = 0
    for row in metadata_rows:
        parent_id = row["parent_id"]
        node_id = row["node_id"]
        if parent_id and parent_id != "0" and parent_id in node_to_index and node_id in node_to_index:
            parent_child_pairs += 1
            if radii[node_to_index[node_id]] <= radii[node_to_index[parent_id]]:
                parent_child_violations += 1

    leaf_radii = np.asarray(
        [radii[node_to_index[node_id]] for node_id in objects if metadata_map[node_id]["selectable"] == "Y"],
        dtype=np.float64,
    )
    internal_radii = np.asarray(
        [radii[node_to_index[node_id]] for node_id in objects if metadata_map[node_id]["selectable"] != "Y"],
        dtype=np.float64,
    )

    return {
        "depth_radius": {
            "pearson": pearson_corr(depths, radii),
            "spearman": spearman_corr(depths, radii),
        },
        "mean_radius_by_depth": mean_radius_by_depth,
        "adjacent_depth_gaps": adjacent_depth_gaps,
        "adjacent_depth_quantile_gaps": adjacent_depth_quantile_gaps,
        "minimum_adjacent_quantile_gap": minimum_adjacent_quantile_gap,
        "positive_adjacent_depth_quantile_gap_count": positive_adjacent_depth_quantile_gap_count,
        "depth_overlap_rates": depth_overlap_rates,
        "minimum_adjacent_gap": minimum_adjacent_gap,
        "monotonic_depth_means": monotonic_depth_means,
        "parent_child_radial_violation_rate": (
            float(parent_child_violations / parent_child_pairs) if parent_child_pairs else float("nan")
        ),
        "leaf_internal_radius_ratio": safe_ratio(
            float(leaf_radii.mean()) if leaf_radii.size else float("nan"),
            float(internal_radii.mean()) if internal_radii.size else float("nan"),
        ),
        "leaf_mean_radius": float(leaf_radii.mean()) if leaf_radii.size else float("nan"),
        "internal_mean_radius": float(internal_radii.mean()) if internal_radii.size else float("nan"),
        "radius_summary": {
            "leaf": summarize(leaf_radii),
            "internal": summarize(internal_radii),
        },
    }


def compute_branch_geometry_metrics(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_rows: list[dict[str, str]],
) -> dict[str, object]:
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    branch_groups = defaultdict(list)
    for node_id in objects:
        row = metadata_map[node_id]
        if int(row["depth"]) == 0:
            continue
        branch_groups[row["top_branch_id"]].append(node_to_index[node_id])

    branch_ids = sorted(branch for branch, indices in branch_groups.items() if indices)
    if len(branch_ids) < 2:
        return {
            "branch_silhouette": float("nan"),
            "top_branch_centroid_distances": {
                "count": 0,
                "mean": float("nan"),
                "median": float("nan"),
                "min": float("nan"),
                "max": float("nan"),
            },
        }

    included_indices = [index for branch in branch_ids for index in branch_groups[branch]]
    compact_embeddings = embeddings[included_indices]
    compact_distances = pairwise_poincare_distances(compact_embeddings)
    original_to_compact = {original: compact for compact, original in enumerate(included_indices)}

    silhouettes = []
    for branch_id in branch_ids:
        same_original = branch_groups[branch_id]
        same_compact = [original_to_compact[index] for index in same_original]
        for original_index in same_original:
            compact_index = original_to_compact[original_index]
            same_without_self = [index for index in same_compact if index != compact_index]
            a = float(compact_distances[compact_index, same_without_self].mean()) if same_without_self else 0.0
            other_means = []
            for other_branch in branch_ids:
                if other_branch == branch_id:
                    continue
                other_compact = [original_to_compact[index] for index in branch_groups[other_branch]]
                other_means.append(float(compact_distances[compact_index, other_compact].mean()))
            b = min(other_means)
            denominator = max(a, b)
            if denominator > 0.0:
                silhouettes.append((b - a) / denominator)

    centroids = []
    for branch_id in branch_ids:
        centroid = embeddings[branch_groups[branch_id]].mean(axis=0, keepdims=True)
        centroids.append(project_to_ball_np(centroid)[0])
    centroid_distances = pairwise_poincare_distances(np.asarray(centroids, dtype=np.float64))
    tri = np.triu_indices(len(branch_ids), k=1)
    centroid_pair_distances = centroid_distances[tri]

    return {
        "branch_silhouette": float(np.mean(silhouettes)) if silhouettes else float("nan"),
        "top_branch_centroid_distances": summarize(centroid_pair_distances),
    }


def compute_branch_pair_diagnostics(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_rows: list[dict[str, str]],
) -> dict[str, object]:
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    branch_groups = defaultdict(list)
    depth_groups = defaultdict(list)
    for node_id in objects:
        row = metadata_map[node_id]
        depth = int(row["depth"])
        depth_groups[depth].append(node_id)
        if depth > 0:
            branch_groups[row["top_branch_id"]].append(node_id)

    within_branch_pairs = []
    across_branch_pairs = []
    same_depth_within_branch_pairs = []
    same_depth_across_branch_pairs = []
    branch_ids = sorted(branch for branch, node_ids in branch_groups.items() if node_ids)
    for branch_id in branch_ids:
        branch_nodes = [node_to_index[node_id] for node_id in branch_groups[branch_id]]
        for left_index in range(len(branch_nodes)):
            for right_index in range(left_index + 1, len(branch_nodes)):
                within_branch_pairs.append((branch_nodes[left_index], branch_nodes[right_index]))

    for _, node_ids in depth_groups.items():
        non_root = [node_id for node_id in node_ids if int(metadata_map[node_id]["depth"]) > 0]
        for left_index in range(len(non_root)):
            for right_index in range(left_index + 1, len(non_root)):
                left = non_root[left_index]
                right = non_root[right_index]
                pair = (node_to_index[left], node_to_index[right])
                if metadata_map[left]["top_branch_id"] == metadata_map[right]["top_branch_id"]:
                    same_depth_within_branch_pairs.append(pair)
                else:
                    same_depth_across_branch_pairs.append(pair)

    for left_index, left_branch in enumerate(branch_ids):
        for right_branch in branch_ids[left_index + 1 :]:
            left_nodes = [node_to_index[node_id] for node_id in branch_groups[left_branch][:30]]
            right_nodes = [node_to_index[node_id] for node_id in branch_groups[right_branch][:30]]
            for left_node in left_nodes:
                for right_node in right_nodes:
                    across_branch_pairs.append((left_node, right_node))

    within_branch_distances = sample_pairwise_distances(embeddings, within_branch_pairs, max_pairs=4000, seed=42)
    across_branch_distances = sample_pairwise_distances(embeddings, across_branch_pairs, max_pairs=4000, seed=42)
    same_depth_within_branch_distances = sample_pairwise_distances(
        embeddings,
        same_depth_within_branch_pairs,
        max_pairs=4000,
        seed=43,
    )
    same_depth_across_branch_distances = sample_pairwise_distances(
        embeddings,
        same_depth_across_branch_pairs,
        max_pairs=4000,
        seed=43,
    )
    angular_within_branch_distances = sample_angular_distances(embeddings, within_branch_pairs, max_pairs=4000, seed=44)
    angular_across_branch_distances = sample_angular_distances(embeddings, across_branch_pairs, max_pairs=4000, seed=44)

    within_branch_mean = float(np.mean(within_branch_distances)) if within_branch_distances.size else float("nan")
    across_branch_mean = float(np.mean(across_branch_distances)) if across_branch_distances.size else float("nan")
    same_depth_within_branch_mean = (
        float(np.mean(same_depth_within_branch_distances)) if same_depth_within_branch_distances.size else float("nan")
    )
    same_depth_across_branch_mean = (
        float(np.mean(same_depth_across_branch_distances)) if same_depth_across_branch_distances.size else float("nan")
    )
    angular_within_branch_mean = (
        float(np.mean(angular_within_branch_distances)) if angular_within_branch_distances.size else float("nan")
    )
    angular_across_branch_mean = (
        float(np.mean(angular_across_branch_distances)) if angular_across_branch_distances.size else float("nan")
    )

    return {
        "branch_separation": {
            "within_branch": summarize(within_branch_distances),
            "across_branch": summarize(across_branch_distances),
        },
        "same_depth_branch_separation": {
            "within_branch": summarize(same_depth_within_branch_distances),
            "across_branch": summarize(same_depth_across_branch_distances),
        },
        "angular_branch_separation": {
            "within_branch": summarize(angular_within_branch_distances),
            "across_branch": summarize(angular_across_branch_distances),
        },
        "ratios": {
            "within_branch_to_across_branch_mean": safe_ratio(within_branch_mean, across_branch_mean),
            "same_depth_within_branch_to_across_branch_mean": safe_ratio(
                same_depth_within_branch_mean,
                same_depth_across_branch_mean,
            ),
            "angular_within_branch_to_across_branch_mean": safe_ratio(
                angular_within_branch_mean,
                angular_across_branch_mean,
            ),
        },
    }


def compute_disease90_metrics(
    checkpoint_path: Path,
    metadata_rows: list[dict[str, str]],
    relations: list[tuple[str, str, float]],
) -> dict[str, object]:
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, checkpoint = checkpoint_embeddings_and_objects(checkpoint_path)
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}

    adjacency = defaultdict(set)
    ancestor_sets = defaultdict(set)
    for child_id, ancestor_id, _ in relations:
        if child_id not in node_to_index or ancestor_id not in node_to_index:
            continue
        adjacency[node_to_index[child_id]].add(node_to_index[ancestor_id])
        ancestor_sets[child_id].add(ancestor_id)

    build_model, _, _, _, _, eval_reconstruction = import_hype_modules()
    model = build_model(checkpoint["conf"], len(objects))
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        mean_rank, map_rank = eval_reconstruction(adjacency, model)

    radius_structure = compute_radius_structure_metrics(embeddings, objects, metadata_rows)
    branch_geometry = compute_branch_geometry_metrics(embeddings, objects, metadata_rows)

    candidate_ancestors = {
        node_id: [other for other in objects if int(metadata_map[other]["depth"]) < int(metadata_map[node_id]["depth"])]
        for node_id in objects
    }
    parent_ranks = []
    parent_ranks_by_depth = defaultdict(list)
    ancestor_ap = []
    ancestor_mean_rank = []
    for node_id in objects:
        parent_id = metadata_map[node_id]["parent_id"]
        if not parent_id or parent_id not in node_to_index:
            continue
        candidates = candidate_ancestors[node_id]
        candidate_indices = [node_to_index[candidate] for candidate in candidates]
        distances = poincare_distance_matrix(
            embeddings[node_to_index[node_id] : node_to_index[node_id] + 1],
            embeddings[candidate_indices],
        )
        ordering = np.argsort(distances)
        ordered_candidates = [candidates[index] for index in ordering]
        rank = ordered_candidates.index(parent_id) + 1
        depth = int(metadata_map[node_id]["depth"])
        parent_ranks.append(rank)
        parent_ranks_by_depth[depth].append(rank)

        labels = np.asarray([1 if candidate in ancestor_sets[node_id] else 0 for candidate in candidates], dtype=np.int64)
        scores = -distances
        ancestor_ap.append(average_precision(labels, scores))
        positive_ranks = [ordered_candidates.index(candidate) + 1 for candidate in ancestor_sets[node_id] if candidate in ordered_candidates]
        if positive_ranks:
            ancestor_mean_rank.append(float(np.mean(positive_ranks)))

    siblings_by_parent = defaultdict(list)
    depth_groups = defaultdict(list)
    for node_id in objects:
        row = metadata_map[node_id]
        depth_groups[int(row["depth"])].append(node_id)
        if row["parent_id"] and row["parent_id"] in node_to_index:
            siblings_by_parent[row["parent_id"]].append(node_id)

    sibling_pairs = []
    for sibling_ids in siblings_by_parent.values():
        for left_index in range(len(sibling_ids)):
            for right_index in range(left_index + 1, len(sibling_ids)):
                sibling_pairs.append((node_to_index[sibling_ids[left_index]], node_to_index[sibling_ids[right_index]]))
    non_sibling_pairs = []
    for _, node_ids in depth_groups.items():
        for left_index in range(len(node_ids)):
            for right_index in range(left_index + 1, len(node_ids)):
                left = node_ids[left_index]
                right = node_ids[right_index]
                if metadata_map[left]["parent_id"] != metadata_map[right]["parent_id"]:
                    non_sibling_pairs.append((node_to_index[left], node_to_index[right]))
    sibling_distances = sample_pairwise_distances(embeddings, sibling_pairs, max_pairs=4000, seed=42)
    non_sibling_distances = sample_pairwise_distances(embeddings, non_sibling_pairs, max_pairs=4000, seed=42)

    sibling_mean = float(np.mean(sibling_distances)) if sibling_distances.size else float("nan")
    non_sibling_mean = float(np.mean(non_sibling_distances)) if non_sibling_distances.size else float("nan")
    branch_pair_diagnostics = compute_branch_pair_diagnostics(embeddings, objects, metadata_rows)

    metrics = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "reconstruction": {"mean_rank": float(mean_rank), "map_rank": float(map_rank)},
        "depth_radius": radius_structure["depth_radius"],
        "radius_structure": {
            "mean_radius_by_depth": radius_structure["mean_radius_by_depth"],
            "adjacent_depth_gaps": radius_structure["adjacent_depth_gaps"],
            "adjacent_depth_quantile_gaps": radius_structure["adjacent_depth_quantile_gaps"],
            "minimum_adjacent_quantile_gap": radius_structure["minimum_adjacent_quantile_gap"],
            "positive_adjacent_depth_quantile_gap_count": radius_structure[
                "positive_adjacent_depth_quantile_gap_count"
            ],
            "depth_overlap_rates": radius_structure["depth_overlap_rates"],
            "minimum_adjacent_gap": radius_structure["minimum_adjacent_gap"],
            "monotonic_depth_means": radius_structure["monotonic_depth_means"],
            "parent_child_radial_violation_rate": radius_structure["parent_child_radial_violation_rate"],
            "leaf_internal_radius_ratio": radius_structure["leaf_internal_radius_ratio"],
            "leaf_mean_radius": radius_structure["leaf_mean_radius"],
            "internal_mean_radius": radius_structure["internal_mean_radius"],
        },
        "parent_ranking": {
            "mean_rank": float(np.mean(parent_ranks)),
            "median_rank": float(np.median(parent_ranks)),
            "depth_stratified_mean_rank": {
                str(depth): float(np.mean(ranks)) for depth, ranks in sorted(parent_ranks_by_depth.items())
            },
        },
        "ancestor_ranking": {
            "mean_average_precision": float(np.nanmean(np.asarray(ancestor_ap, dtype=np.float64))),
            "mean_positive_rank": float(np.mean(ancestor_mean_rank)),
        },
        "sibling_cohesion": {
            "siblings": summarize(sibling_distances),
            "same_depth_non_siblings": summarize(non_sibling_distances),
        },
        "branch_separation": branch_pair_diagnostics["branch_separation"],
        "same_depth_branch_separation": branch_pair_diagnostics["same_depth_branch_separation"],
        "angular_branch_separation": branch_pair_diagnostics["angular_branch_separation"],
        "branch_geometry": branch_geometry,
        "radius_summary": radius_structure["radius_summary"],
        "ratios": {
            "sibling_to_non_sibling_mean": safe_ratio(sibling_mean, non_sibling_mean),
            **branch_pair_diagnostics["ratios"],
        },
    }
    metrics["gate_deficits"] = gate_deficits(metrics)
    metrics["gate_deficit_score"] = float(metrics["gate_deficits"]["total"])
    return metrics


def passes_acceptance_floors(
    metrics: dict[str, object],
    floor_reconstruction_map: float = 0.30,
    floor_parent_mean_rank: float = 1.20,
    floor_ancestor_map: float = 0.95,
    floor_sibling_ratio: float = 0.30,
    floor_branch_ratio: float = 0.35,
    floor_min_adjacent_gap: float = 0.0,
    floor_positive_quantile_gaps: int = 3,
    ceiling_radial_violation: float = 0.10,
) -> bool:
    deficits = gate_deficits(
        metrics,
        floor_reconstruction_map=floor_reconstruction_map,
        floor_parent_mean_rank=floor_parent_mean_rank,
        floor_ancestor_map=floor_ancestor_map,
        floor_sibling_ratio=floor_sibling_ratio,
        floor_branch_ratio=floor_branch_ratio,
        floor_min_adjacent_gap=floor_min_adjacent_gap,
        floor_positive_quantile_gaps=floor_positive_quantile_gaps,
        ceiling_radial_violation=ceiling_radial_violation,
    )
    return bool(deficits["total"] == 0.0)


def normalized_floor_deficit(value: float, floor: float) -> float:
    if np.isnan(value):
        return 1.0
    if floor == 0.0:
        return max(0.0, -value)
    return max(0.0, (floor - value) / abs(floor))


def normalized_ceiling_deficit(value: float, ceiling: float) -> float:
    if np.isnan(value):
        return 1.0
    if ceiling == 0.0:
        return max(0.0, value)
    return max(0.0, (value - ceiling) / abs(ceiling))


def gate_deficits(
    metrics: dict[str, object],
    floor_reconstruction_map: float = 0.30,
    floor_parent_mean_rank: float = 1.20,
    floor_ancestor_map: float = 0.95,
    floor_sibling_ratio: float = 0.30,
    floor_branch_ratio: float = 0.35,
    floor_min_adjacent_gap: float = 0.0,
    floor_positive_quantile_gaps: int = 3,
    ceiling_radial_violation: float = 0.10,
) -> dict[str, float]:
    radius_structure = metrics["radius_structure"]
    deficits = {
        "reconstruction_map": normalized_floor_deficit(
            float(metrics["reconstruction"]["map_rank"]),
            floor_reconstruction_map,
        ),
        "parent_mean_rank": normalized_ceiling_deficit(
            float(metrics["parent_ranking"]["mean_rank"]),
            floor_parent_mean_rank,
        ),
        "ancestor_map": normalized_floor_deficit(
            float(metrics["ancestor_ranking"]["mean_average_precision"]),
            floor_ancestor_map,
        ),
        "sibling_ratio": normalized_ceiling_deficit(
            float(metrics["ratios"]["sibling_to_non_sibling_mean"]),
            floor_sibling_ratio,
        ),
        "branch_ratio": normalized_ceiling_deficit(
            float(metrics["ratios"]["within_branch_to_across_branch_mean"]),
            floor_branch_ratio,
        ),
        "minimum_adjacent_gap": normalized_floor_deficit(
            float(radius_structure["minimum_adjacent_gap"]),
            floor_min_adjacent_gap,
        ),
        "positive_quantile_gaps": normalized_floor_deficit(
            float(radius_structure["positive_adjacent_depth_quantile_gap_count"]),
            float(floor_positive_quantile_gaps),
        ),
        "radial_violation": normalized_ceiling_deficit(
            float(radius_structure["parent_child_radial_violation_rate"]),
            ceiling_radial_violation,
        ),
    }
    deficits["total"] = float(sum(deficits.values()))
    return deficits


def gate_deficit_rank_key(metrics: dict[str, object]) -> tuple[float, ...]:
    deficits = metrics.get("gate_deficits") or gate_deficits(metrics)
    radius_structure = metrics["radius_structure"]
    return (
        -float(deficits["total"]),
        -float(deficits["branch_ratio"]),
        -float(deficits["sibling_ratio"]),
        -float(deficits["radial_violation"]),
        float(radius_structure["positive_adjacent_depth_quantile_gap_count"]),
        float(radius_structure["minimum_adjacent_quantile_gap"]),
        -float(metrics["ratios"]["within_branch_to_across_branch_mean"]),
        float(metrics["ancestor_ranking"]["mean_average_precision"]),
        float(metrics["reconstruction"]["map_rank"]),
    )


def geometry_rank_key(metrics: dict[str, object]) -> tuple[float, ...]:
    return gate_deficit_rank_key(metrics)


def depth_first_rank_key(metrics: dict[str, object]) -> tuple[float, ...]:
    return geometry_rank_key(metrics)


def load_metadata_and_relations(
    metadata_tsv: Path,
    relations_csv: Path,
) -> tuple[list[dict[str, str]], list[tuple[str, str, float]]]:
    return read_metadata_tsv(metadata_tsv), read_relations_csv(relations_csv)
