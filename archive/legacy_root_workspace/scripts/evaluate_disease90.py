#!/usr/bin/env python3

import argparse
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVAL_JSON,
    DEFAULT_EVAL_MD,
    DEFAULT_METADATA_TSV,
    DEFAULT_RELATIONS_CSV,
    average_precision,
    checkpoint_embeddings_and_objects,
    import_hype_modules,
    parse_args_with_defaults,
    pearson_corr,
    poincare_distance_matrix,
    read_metadata_tsv,
    read_relations_csv,
    sample_pairwise_distances,
    spearman_corr,
    write_json,
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


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate disease-90 Poincare embeddings")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--relations-csv", type=Path, default=DEFAULT_RELATIONS_CSV)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_EVAL_MD)
    args = parse_args_with_defaults(parser)

    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, checkpoint = checkpoint_embeddings_and_objects(args.checkpoint)
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    if set(node_to_index) != set(metadata_map):
        missing = sorted(set(node_to_index) - set(metadata_map))
        raise ValueError(f"Metadata/checkpoint node mismatch, example missing: {missing[:5]}")

    relations = read_relations_csv(args.relations_csv)
    adjacency = defaultdict(set)
    ancestor_sets = defaultdict(set)
    for child_id, ancestor_id, _ in relations:
        adjacency[node_to_index[child_id]].add(node_to_index[ancestor_id])
        ancestor_sets[child_id].add(ancestor_id)

    build_model, eval_reconstruction = import_hype_modules()
    model = build_model(checkpoint["conf"], len(objects))
    model.load_state_dict(checkpoint["model"])
    with torch.no_grad():
        mean_rank, map_rank = eval_reconstruction(adjacency, model)

    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=np.float64)
    radii = np.linalg.norm(embeddings, axis=1)
    depth_radius = {
        "pearson": pearson_corr(depths, radii),
        "spearman": spearman_corr(depths, radii),
    }

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
        distances = poincare_distance_matrix(embeddings[node_to_index[node_id] : node_to_index[node_id] + 1], embeddings[candidate_indices])
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
    leaf_radii = []
    internal_radii = []
    for node_id in objects:
        row = metadata_map[node_id]
        depth_groups[int(row["depth"])].append(node_id)
        if row["parent_id"] and row["parent_id"] in node_to_index:
            siblings_by_parent[row["parent_id"]].append(node_id)
        if row["selectable"] == "Y":
            leaf_radii.append(radii[node_to_index[node_id]])
        else:
            internal_radii.append(radii[node_to_index[node_id]])

    sibling_pairs = []
    for sibling_ids in siblings_by_parent.values():
        for i in range(len(sibling_ids)):
            for j in range(i + 1, len(sibling_ids)):
                sibling_pairs.append((node_to_index[sibling_ids[i]], node_to_index[sibling_ids[j]]))
    non_sibling_pairs = []
    for _, node_ids in depth_groups.items():
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                left = node_ids[i]
                right = node_ids[j]
                if metadata_map[left]["parent_id"] != metadata_map[right]["parent_id"]:
                    non_sibling_pairs.append((node_to_index[left], node_to_index[right]))
    sibling_distances = sample_pairwise_distances(embeddings, sibling_pairs, max_pairs=4000, seed=42)
    non_sibling_distances = sample_pairwise_distances(embeddings, non_sibling_pairs, max_pairs=4000, seed=42)

    branch_groups = defaultdict(list)
    for node_id in objects:
        branch_groups[metadata_map[node_id]["top_branch_id"]].append(node_id)
    within_branch_pairs = []
    across_branch_pairs = []
    branch_ids = sorted(branch_groups)
    for branch_id in branch_ids:
        branch_nodes = [node_to_index[node_id] for node_id in branch_groups[branch_id] if metadata_map[node_id]["depth"] != "0"]
        for i in range(len(branch_nodes)):
            for j in range(i + 1, len(branch_nodes)):
                within_branch_pairs.append((branch_nodes[i], branch_nodes[j]))
    for left_idx, left_branch in enumerate(branch_ids):
        for right_branch in branch_ids[left_idx + 1 :]:
            left_nodes = [node_to_index[node_id] for node_id in branch_groups[left_branch][:30]]
            right_nodes = [node_to_index[node_id] for node_id in branch_groups[right_branch][:30]]
            for left_node in left_nodes:
                for right_node in right_nodes:
                    across_branch_pairs.append((left_node, right_node))
    within_branch_distances = sample_pairwise_distances(embeddings, within_branch_pairs, max_pairs=4000, seed=42)
    across_branch_distances = sample_pairwise_distances(embeddings, across_branch_pairs, max_pairs=4000, seed=42)

    metrics = {
        "checkpoint_epoch": checkpoint.get("epoch"),
        "reconstruction": {"mean_rank": float(mean_rank), "map_rank": float(map_rank)},
        "depth_radius": depth_radius,
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
        "branch_separation": {
            "within_branch": summarize(within_branch_distances),
            "across_branch": summarize(across_branch_distances),
        },
        "radius_summary": {
            "leaf": summarize(np.asarray(leaf_radii, dtype=np.float64)),
            "internal": summarize(np.asarray(internal_radii, dtype=np.float64)),
        },
    }
    write_json(args.out_json, metrics)

    lines = [
        "# Disease-90 embedding evaluation",
        "",
        f"- Reconstruction mean rank: {metrics['reconstruction']['mean_rank']:.4f}",
        f"- Reconstruction MAP: {metrics['reconstruction']['map_rank']:.4f}",
        f"- Depth/radius Spearman: {metrics['depth_radius']['spearman']:.4f}",
        f"- Parent mean rank: {metrics['parent_ranking']['mean_rank']:.4f}",
        f"- Ancestor MAP: {metrics['ancestor_ranking']['mean_average_precision']:.4f}",
        f"- Mean sibling distance: {metrics['sibling_cohesion']['siblings']['mean']:.4f}",
        f"- Mean same-depth non-sibling distance: {metrics['sibling_cohesion']['same_depth_non_siblings']['mean']:.4f}",
        f"- Mean within-branch distance: {metrics['branch_separation']['within_branch']['mean']:.4f}",
        f"- Mean across-branch distance: {metrics['branch_separation']['across_branch']['mean']:.4f}",
        "",
        "Hierarchy looks healthier when depth/radius correlation is positive, parent rank is close to 1, and sibling distances are smaller than same-depth non-sibling distances.",
    ]
    args.out_md.write_text("\n".join(lines), encoding="utf-8")
    print(f"Wrote metrics to {args.out_json}")
    print(f"Wrote summary to {args.out_md}")


if __name__ == "__main__":
    main()
