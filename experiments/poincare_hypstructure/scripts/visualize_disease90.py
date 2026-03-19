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

DEFAULT_LABEL_ROUTE = ["90", "1200", "43400", "43410", "43420"]


def pca_to_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def project_to_unit_disk(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    radii = np.linalg.norm(values, axis=1, keepdims=True)
    scales = np.minimum(1.0, (1.0 - eps) / np.maximum(radii, eps))
    return values * scales


def route_label(row: dict[str, str]) -> str:
    if row["depth"] == "0":
        return row["coding"]
    if row["depth"] == "1":
        return row["coding"]
    return row["coding"]


def branch_label(row: dict[str, str]) -> str:
    return row["coding"].replace("Block ", "")


def main() -> None:
    parser = argparse.ArgumentParser(description="Visualize disease-90 Poincare+HypStructure embeddings")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--plots-dir", type=Path, default=PLOTS_DIR)
    parser.add_argument(
        "--label-route-node-ids",
        nargs="+",
        default=DEFAULT_LABEL_ROUTE,
        help="Node ids to label on an additional branch-colored disk plot.",
    )
    args = parse_args_with_defaults(parser)

    args.plots_dir.mkdir(parents=True, exist_ok=True)
    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    embeddings, objects, _ = checkpoint_embeddings_and_objects(args.checkpoint)
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=np.int64)
    branches = [metadata_map[node_id]["top_branch_id"] for node_id in objects]
    radii = np.linalg.norm(embeddings, axis=1)

    parents = {
        row["node_id"]: row["parent_id"]
        for row in metadata_rows
        if row["parent_id"] and row["parent_id"] in node_to_index
    }

    embedding_2d = embeddings.copy() if embeddings.shape[1] == 2 else project_to_unit_disk(pca_to_2d(embeddings))

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_artist(plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0))
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
    ax.set_title("Disease-90 Poincare+HypStructure disk colored by depth")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.plots_dir / "poincare_disk_depth.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    branch_ids = sorted(set(branches))
    branch_map = {branch: index for index, branch in enumerate(branch_ids)}
    branch_values = np.asarray([branch_map[branch] for branch in branches], dtype=np.int64)
    fig, ax = plt.subplots(figsize=(8, 8))
    ax.add_artist(plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0))
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
    ax.set_title("Disease-90 Poincare+HypStructure disk colored by top branch")
    ax.set_aspect("equal")
    ax.axis("off")
    fig.savefig(args.plots_dir / "poincare_disk_branch.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    branch_centroids = []
    for branch_id in branch_ids:
        if branch_id == "90":
            continue
        branch_indices = [index for index, node_id in enumerate(objects) if metadata_map[node_id]["top_branch_id"] == branch_id]
        centroid = embedding_2d[branch_indices].mean(axis=0)
        branch_centroids.append(
            {
                "branch_id": branch_id,
                "label": branch_label(metadata_map[branch_id]),
                "value": branch_map[branch_id],
                "centroid_x": float(centroid[0]),
                "centroid_y": float(centroid[1]),
            }
        )

    left_items = sorted(
        [item for item in branch_centroids if item["centroid_x"] <= 0.0],
        key=lambda item: item["centroid_y"],
        reverse=True,
    )
    right_items = sorted(
        [item for item in branch_centroids if item["centroid_x"] > 0.0],
        key=lambda item: item["centroid_y"],
        reverse=True,
    )

    x_min = float(embedding_2d[:, 0].min())
    x_max = float(embedding_2d[:, 0].max())
    y_min = float(embedding_2d[:, 1].min())
    y_max = float(embedding_2d[:, 1].max())
    x_span = max(x_max - x_min, 1e-6)
    y_span = max(y_max - y_min, 1e-6)

    def assign_label_positions(items: list[dict[str, object]], side: str) -> None:
        if not items:
            return
        y_positions = np.linspace(y_max, y_min, num=len(items))
        x_position = x_min - 0.32 * x_span if side == "left" else x_max + 0.32 * x_span
        alignment = "right" if side == "left" else "left"
        for item, y_position in zip(items, y_positions):
            item["label_x"] = x_position
            item["label_y"] = float(y_position)
            item["ha"] = alignment

    assign_label_positions(left_items, "left")
    assign_label_positions(right_items, "right")

    fig, ax = plt.subplots(figsize=(11, 9))
    ax.add_artist(plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0))
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.25,
            alpha=0.18,
            color="gray",
        )
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=branch_values, s=12, cmap="tab20", alpha=0.7)
    for item in left_items + right_items:
        color = plt.cm.tab20(item["value"] / max(len(branch_ids) - 1, 1))
        ax.scatter(
            [item["centroid_x"]],
            [item["centroid_y"]],
            s=90,
            facecolors="white",
            edgecolors=[color],
            linewidths=1.5,
            zorder=5,
        )
        ax.annotate(
            item["label"],
            xy=(item["centroid_x"], item["centroid_y"]),
            xytext=(item["label_x"], item["label_y"]),
            textcoords="data",
            ha=item["ha"],
            va="center",
            fontsize=10,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": color, "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            zorder=6,
        )
    if "90" in node_to_index:
        root_index = node_to_index["90"]
        ax.scatter(
            [embedding_2d[root_index, 0]],
            [embedding_2d[root_index, 1]],
            s=100,
            facecolors="white",
            edgecolors="black",
            linewidths=1.4,
            zorder=7,
        )
        ax.annotate(
            "Chapter IX",
            xy=(embedding_2d[root_index, 0], embedding_2d[root_index, 1]),
            xytext=(embedding_2d[root_index, 0] + 0.18 * x_span, embedding_2d[root_index, 1] - 0.02 * y_span),
            textcoords="data",
            fontsize=10,
            color="black",
            bbox={"boxstyle": "round,pad=0.18", "fc": "white", "ec": "black", "alpha": 0.95},
            arrowprops={"arrowstyle": "-", "color": "black", "lw": 1.0},
            zorder=7,
        )
    ax.set_title("Disease-90 top-branch centroids labeled on the Poincare disk")
    ax.set_aspect("equal")
    ax.set_xlim(x_min - 0.38 * x_span, x_max + 0.38 * x_span)
    ax.set_ylim(y_min - 0.08 * y_span, y_max + 0.08 * y_span)
    ax.axis("off")
    fig.savefig(args.plots_dir / "poincare_disk_branch_labeled_centroids.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    route_node_ids = [node_id for node_id in args.label_route_node_ids if node_id in node_to_index]
    if route_node_ids:
        fig, ax = plt.subplots(figsize=(9, 9))
        ax.add_artist(plt.Circle((0.0, 0.0), 1.0, color="black", fill=False, linewidth=1.0))
        for child_id, parent_id in parents.items():
            child_index = node_to_index[child_id]
            parent_index = node_to_index[parent_id]
            ax.plot(
                [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
                [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
                linewidth=0.3,
                alpha=0.2,
                color="gray",
            )
        ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=branch_values, s=14, cmap="tab20", alpha=0.75)

        route_indices = [node_to_index[node_id] for node_id in route_node_ids]
        route_points = embedding_2d[route_indices]
        ax.plot(
            route_points[:, 0],
            route_points[:, 1],
            color="crimson",
            linewidth=1.8,
            marker="o",
            markersize=4,
            zorder=4,
        )
        ax.scatter(
            route_points[:, 0],
            route_points[:, 1],
            s=64,
            facecolors="white",
            edgecolors="crimson",
            linewidths=1.2,
            zorder=5,
        )

        offsets = [(8, 8), (10, -16), (10, 10), (18, -18), (-48, 12)]
        for point_index, node_id in enumerate(route_node_ids):
            row = metadata_map[node_id]
            dx, dy = offsets[point_index % len(offsets)]
            ax.annotate(
                route_label(row),
                xy=(embedding_2d[node_to_index[node_id], 0], embedding_2d[node_to_index[node_id], 1]),
                xytext=(dx, dy),
                textcoords="offset points",
                fontsize=9,
                color="crimson",
                bbox={"boxstyle": "round,pad=0.15", "fc": "white", "ec": "crimson", "alpha": 0.9},
                arrowprops={"arrowstyle": "-", "color": "crimson", "lw": 0.8},
                zorder=6,
            )

        title_parts = [metadata_map[node_id]["coding"] for node_id in route_node_ids]
        ax.set_title("Disease-90 branch plot with labeled route: " + " > ".join(title_parts))
        ax.set_aspect("equal")
        ax.axis("off")
        fig.savefig(args.plots_dir / "poincare_disk_branch_labeled_route.png", dpi=300, bbox_inches="tight")
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
    for _, node_ids in depth_groups.items():
        for i in range(len(node_ids)):
            for j in range(i + 1, len(node_ids)):
                left = node_ids[i]
                right = node_ids[j]
                if metadata_map[left]["parent_id"] != metadata_map[right]["parent_id"]:
                    non_sibling_pairs.append((node_to_index[left], node_to_index[right]))
    sibling_distances = sample_pairwise_distances(embeddings, sibling_pairs, max_pairs=4000, seed=42)
    non_sibling_distances = sample_pairwise_distances(embeddings, non_sibling_pairs, max_pairs=4000, seed=42)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([sibling_distances, non_sibling_distances], tick_labels=["siblings", "same-depth non-siblings"], showfliers=False)
    ax.set_ylabel("Poincare distance")
    ax.set_title("Sibling cohesion summary")
    fig.savefig(args.plots_dir / "sibling_distance_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)

    branch_groups = defaultdict(list)
    for node_id in objects:
        branch_groups[metadata_map[node_id]["top_branch_id"]].append(node_id)
    within_pairs = []
    across_pairs = []
    ordered_branch_ids = sorted(branch_groups)
    for branch_id in ordered_branch_ids:
        branch_indices = [node_to_index[node_id] for node_id in branch_groups[branch_id] if metadata_map[node_id]["depth"] != "0"]
        for i in range(len(branch_indices)):
            for j in range(i + 1, len(branch_indices)):
                within_pairs.append((branch_indices[i], branch_indices[j]))
    for left_index, left_branch in enumerate(ordered_branch_ids):
        for right_branch in ordered_branch_ids[left_index + 1 :]:
            left_nodes = [node_to_index[node_id] for node_id in branch_groups[left_branch][:30]]
            right_nodes = [node_to_index[node_id] for node_id in branch_groups[right_branch][:30]]
            for left in left_nodes:
                for right in right_nodes:
                    across_pairs.append((left, right))
    within_branch = sample_pairwise_distances(embeddings, within_pairs, max_pairs=4000, seed=42)
    across_branch = sample_pairwise_distances(embeddings, across_pairs, max_pairs=4000, seed=42)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot([within_branch, across_branch], tick_labels=["within branch", "across branches"], showfliers=False)
    ax.set_ylabel("Poincare distance")
    ax.set_title("Branch separation summary")
    fig.savefig(args.plots_dir / "branch_separation_summary.png", dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Wrote plots to {args.plots_dir}")


if __name__ == "__main__":
    main()
