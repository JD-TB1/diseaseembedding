#!/usr/bin/env python3

import argparse
import os
from collections import defaultdict
from pathlib import Path

SCRIPT_BASE = Path(__file__).resolve().parents[1]
PLOT_CACHE_DIR = SCRIPT_BASE / "logs" / "mplconfig"
PLOT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PLOT_CACHE_DIR))
os.environ.setdefault("XDG_CACHE_HOME", str(PLOT_CACHE_DIR))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_METADATA_TSV,
    PLOTS_DIR,
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    read_metadata_tsv,
    sample_pairwise_distances,
)


def pca_to_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def project_to_unit_disk(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    radii = np.linalg.norm(values, axis=1, keepdims=True)
    scales = np.minimum(1.0, (1.0 - eps) / np.maximum(radii, eps))
    return values * scales


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize disease-90 Poincare embeddings")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    args = parse_args_with_defaults(parser)

    args.plots_dir.mkdir(parents=True, exist_ok=True)

    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, _ = checkpoint_embeddings_and_objects(args.checkpoint)
    missing = [node_id for node_id in objects if node_id not in metadata_map]
    if missing:
        raise ValueError(f"Metadata missing checkpoint node ids, example: {missing[:5]}")

    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    parents = {
        row["node_id"]: row["parent_id"]
        for row in metadata_rows
        if row["parent_id"] and row["parent_id"] in node_to_index
    }
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=np.int64)
    branches = [metadata_map[node_id]["top_branch_id"] for node_id in objects]
    radii = np.linalg.norm(embeddings, axis=1)

    if embeddings.shape[1] == 2:
        embedding_2d = embeddings.copy()
    else:
        embedding_2d = project_to_unit_disk(pca_to_2d(embeddings))

    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0)
    ax.add_artist(circle)
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.35,
            alpha=0.3,
            color="gray",
        )
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=depths, s=16, cmap="viridis")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="depth")
    ax.set_title("Disease-90 Poincare disk colored by depth")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.plots_dir / "poincare_disk_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    unique_branches = sorted({branch for branch in branches})
    branch_map = {branch: index for index, branch in enumerate(unique_branches)}
    branch_values = np.asarray([branch_map[branch] for branch in branches], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 8))
    circle = plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0)
    ax.add_artist(circle)
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.35,
            alpha=0.3,
            color="gray",
        )
    scatter = ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=branch_values, s=16, cmap="tab20")
    plt.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="top branch")
    ax.set_title("Disease-90 Poincare disk colored by top branch")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.plots_dir / "poincare_disk_branch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.scatter(depths, radii, s=14, alpha=0.7)
    ax.set_xlabel("tree depth")
    ax.set_ylabel("Poincare radius")
    ax.set_title("Depth vs radius")
    fig.savefig(args.plots_dir / "depth_vs_radius.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    siblings_by_parent = defaultdict(list)
    depth_groups = defaultdict(list)
    for node_id in objects:
        row = metadata_map[node_id]
        depth_groups[int(row["depth"])].append(node_id)
        if row["parent_id"] and row["parent_id"] in node_to_index:
            siblings_by_parent[row["parent_id"]].append(node_id)

    sibling_pairs = []
    for sibling_ids in siblings_by_parent.values():
        for i in range(len(sibling_ids)):
            for j in range(i + 1, len(sibling_ids)):
                sibling_pairs.append((node_to_index[sibling_ids[i]], node_to_index[sibling_ids[j]]))
    non_sibling_pairs = []
    for depth, node_ids in depth_groups.items():
        depth_indices = [node_to_index[node_id] for node_id in node_ids]
        for i in range(len(depth_indices)):
            for j in range(i + 1, len(depth_indices)):
                left = objects[depth_indices[i]]
                right = objects[depth_indices[j]]
                if metadata_map[left]["parent_id"] != metadata_map[right]["parent_id"]:
                    non_sibling_pairs.append((depth_indices[i], depth_indices[j]))
    sibling_distances = sample_pairwise_distances(embeddings, sibling_pairs, max_pairs=4000, seed=42)
    non_sibling_distances = sample_pairwise_distances(embeddings, non_sibling_pairs, max_pairs=4000, seed=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    data = [sibling_distances, non_sibling_distances]
    ax.boxplot(data, tick_labels=["siblings", "same-depth non-siblings"], showfliers=False)
    ax.set_ylabel("Poincare distance")
    ax.set_title("Sibling cohesion summary")
    fig.savefig(args.plots_dir / "sibling_distance_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    branch_groups = defaultdict(list)
    for node_id in objects:
        branch_groups[metadata_map[node_id]["top_branch_id"]].append(node_id)
    within_pairs = []
    across_pairs = []
    branch_ids = sorted(branch_groups)
    for branch_id, node_ids in branch_groups.items():
        indices = [node_to_index[node_id] for node_id in node_ids if metadata_map[node_id]["depth"] != "0"]
        for i in range(len(indices)):
            for j in range(i + 1, len(indices)):
                within_pairs.append((indices[i], indices[j]))
    for left_index, left_branch in enumerate(branch_ids):
        for right_branch in branch_ids[left_index + 1 :]:
            left_nodes = [node_to_index[node_id] for node_id in branch_groups[left_branch]]
            right_nodes = [node_to_index[node_id] for node_id in branch_groups[right_branch]]
            for i in left_nodes[:30]:
                for j in right_nodes[:30]:
                    across_pairs.append((i, j))
    within_distances = sample_pairwise_distances(embeddings, within_pairs, max_pairs=4000, seed=42)
    across_distances = sample_pairwise_distances(embeddings, across_pairs, max_pairs=4000, seed=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(
        [within_distances, across_distances],
        tick_labels=["within branch", "across branches"],
        showfliers=False,
    )
    ax.set_ylabel("Poincare distance")
    ax.set_title("Branch separation summary")
    fig.savefig(args.plots_dir / "branch_separation_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
