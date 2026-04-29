#!/usr/bin/env python3

import argparse
import json
import os
import textwrap
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
from matplotlib.patches import Circle, FancyBboxPatch, Rectangle

from disease90_common import (
    DEFAULT_CHECKPOINT,
    DEFAULT_EVAL_JSON,
    DEFAULT_METADATA_TSV,
    PROJECT_DIR,
    checkpoint_embeddings_and_objects,
    parse_args_with_defaults,
    read_metadata_tsv,
)


POSTER_DIR = PROJECT_DIR / "visualizations" / "poster_assets"

NAVY = "#16324F"
SLATE = "#4F647A"
TEAL = "#1E9BB0"
CYAN = "#77C6D3"
GOLD = "#E3A72F"
CORAL = "#E45B5B"
LIGHT_BG = "#F7FAFC"
CARD_BG = "#FFFFFF"
EDGE = "#D7E1EA"
TEXT = "#142433"
MUTED = "#5F7285"
BRANCH_COLORS = [
    "#4E79A7",
    "#F28E2B",
    "#E15759",
    "#76B7B2",
    "#59A14F",
    "#EDC948",
    "#B07AA1",
    "#FF9DA7",
    "#9C755F",
    "#BAB0AB",
]


def pca_to_2d(values: np.ndarray) -> np.ndarray:
    centered = values - values.mean(axis=0, keepdims=True)
    _, _, vt = np.linalg.svd(centered, full_matrices=False)
    return centered @ vt[:2].T


def project_to_unit_disk(values: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    radii = np.linalg.norm(values, axis=1, keepdims=True)
    scales = np.minimum(1.0, (1.0 - eps) / np.maximum(radii, eps))
    return values * scales


def style_card_background(fig: plt.Figure, title: str, subtitle: str) -> tuple[plt.Axes, plt.Axes]:
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.96,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.05, 0.93, title, fontsize=22, fontweight="bold", color=NAVY, ha="left", va="center")
    fig.text(0.05, 0.885, wrap_text(subtitle, 88), fontsize=11.5, color=MUTED, ha="left", va="center", linespacing=1.3)
    plot_ax = fig.add_axes([0.05, 0.11, 0.58, 0.72], zorder=2)
    note_ax = fig.add_axes([0.68, 0.12, 0.26, 0.72], zorder=2)
    note_ax.axis("off")
    return plot_ax, note_ax


def wrap_text(text: str, width: int) -> str:
    chunks: list[str] = []
    for block in text.split("\n"):
        wrapped = textwrap.wrap(block, width=width, break_long_words=False, break_on_hyphens=False)
        chunks.extend(wrapped or [""])
    return "\n".join(chunks)


def draw_note_card(ax: plt.Axes, y: float, height: float, title: str, body_lines: list[str], accent: str) -> None:
    patch = FancyBboxPatch(
        (0.0, y),
        1.0,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.02,rounding_size=0.04",
        linewidth=1.1,
        edgecolor=EDGE,
        facecolor="#FBFDFF",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((0.0, y + height - 0.06), 1.0, 0.06, transform=ax.transAxes, color=accent, alpha=0.95))
    ax.text(0.06, y + height - 0.085, title, transform=ax.transAxes, ha="left", va="top", fontsize=11.5, fontweight="bold", color=NAVY)
    current_y = y + height - 0.16
    for line in body_lines:
        wrapped = wrap_text(line, 38).split("\n")
        ax.text(0.07, current_y, f"• {wrapped[0]}", transform=ax.transAxes, ha="left", va="top", fontsize=9.8, color=TEXT)
        current_y -= 0.052
        for continuation in wrapped[1:]:
            ax.text(0.105, current_y, continuation, transform=ax.transAxes, ha="left", va="top", fontsize=9.8, color=TEXT)
            current_y -= 0.044
        current_y -= 0.026


def save_panel(fig: plt.Figure, output_base: Path) -> None:
    output_base.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_base.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(output_base.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)


def render_branch_panel(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    parents = {
        row["node_id"]: row["parent_id"]
        for row in metadata_map.values()
        if row["parent_id"] and row["parent_id"] in node_to_index
    }
    embedding_2d = embeddings.copy() if embeddings.shape[1] == 2 else project_to_unit_disk(pca_to_2d(embeddings))
    embedding_2d = embedding_2d - embedding_2d.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(embedding_2d, axis=1).max()
    if max_norm > 0:
        embedding_2d = embedding_2d * (0.88 / max_norm)
    branch_ids = sorted({metadata_map[node_id]["top_branch_id"] for node_id in objects})
    branch_to_value = {branch_id: index for index, branch_id in enumerate(branch_ids)}
    branch_values = np.asarray([branch_to_value[metadata_map[node_id]["top_branch_id"]] for node_id in objects])

    fig = plt.figure(figsize=(12.5, 8.0))
    ax, note_ax = style_card_background(
        fig,
        "Branch-aware hyperbolic geometry",
        "Display-normalized Poincare disk projection of the disease-90 hybrid embedding, colored by top-level branch.",
    )

    for radius, alpha in [(1.0, 0.9), (0.66, 0.22), (0.33, 0.16)]:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=NAVY,
                linewidth=1.3 if radius == 1.0 else 0.9,
                alpha=alpha,
                linestyle="-" if radius == 1.0 else "--",
            )
        )
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.35,
            alpha=0.12,
            color="#90A4B8",
            zorder=1,
        )

    colors = [BRANCH_COLORS[value % len(BRANCH_COLORS)] for value in branch_values]
    ax.scatter(
        embedding_2d[:, 0],
        embedding_2d[:, 1],
        c=colors,
        s=20,
        alpha=0.82,
        edgecolors="white",
        linewidths=0.25,
        zorder=2,
    )

    branch_centroids = []
    for branch_id in branch_ids:
        if branch_id == "90":
            continue
        idx = [node_to_index[node_id] for node_id in objects if metadata_map[node_id]["top_branch_id"] == branch_id]
        centroid = embedding_2d[idx].mean(axis=0)
        branch_centroids.append((branch_id, centroid))

    label_candidates = sorted(
        branch_centroids,
        key=lambda item: len([node_id for node_id in objects if metadata_map[node_id]["top_branch_id"] == item[0]]),
        reverse=True,
    )[:8]
    for branch_id, centroid in label_candidates:
        color = BRANCH_COLORS[branch_to_value[branch_id] % len(BRANCH_COLORS)]
        label = metadata_map[branch_id]["coding"].replace("Block ", "")
        ax.scatter([centroid[0]], [centroid[1]], s=90, facecolors="white", edgecolors=color, linewidths=1.4, zorder=4)
        ax.annotate(
            label,
            xy=(centroid[0], centroid[1]),
            xytext=(14, 10 if centroid[1] <= 0 else -12),
            textcoords="offset points",
            fontsize=9.5,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.20", "fc": "white", "ec": color, "alpha": 0.96},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 1.0},
            zorder=5,
        )

    ax.set_aspect("equal")
    ax.set_xlim(-1.08, 1.08)
    ax.set_ylim(-1.08, 1.08)
    ax.axis("off")

    branch_metrics = metrics["branch_separation"]
    draw_note_card(
        note_ax,
        0.52,
        0.42,
        "What this figure shows",
        [
            "Distinct branch regions in the Poincare disk.",
            "Within-branch distances are smaller than across-branch distances.",
            "Parent-child geometry remains coherent.",
        ],
        TEAL,
    )
    draw_note_card(
        note_ax,
        0.08,
        0.34,
        "Separation summary",
        [
            f"Within branch mean: {branch_metrics['within_branch']['mean']:.4f}",
            f"Across branch mean: {branch_metrics['across_branch']['mean']:.4f}",
            "Main gain: branch clusters are now readable.",
        ],
        GOLD,
    )
    save_panel(fig, output_base)


def render_depth_panel(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=int)
    radii = np.linalg.norm(embeddings, axis=1)
    unique_depths = sorted(np.unique(depths))

    fig = plt.figure(figsize=(12.5, 8.0))
    ax, note_ax = style_card_background(
        fig,
        "Radial depth ordering",
        "Radius should increase with hierarchy depth; the current hybrid model improves this trend but does not fully separate shells.",
    )

    for depth in unique_depths:
        ax.axvspan(depth - 0.45, depth + 0.45, color="#F4F8FB" if depth % 2 == 0 else "#FAFCFE", zorder=0)

    positions = []
    point_colors = []
    for depth in unique_depths:
        indices = np.where(depths == depth)[0]
        jitter = np.linspace(-0.16, 0.16, num=len(indices)) if len(indices) > 1 else np.array([0.0])
        positions.extend(depth + jitter)
        point_colors.extend([TEAL if depth > 0 else NAVY] * len(indices))

    ax.scatter(positions, radii, s=26, c=point_colors, alpha=0.70, edgecolors="white", linewidths=0.3, zorder=3)

    means = [float(radii[depths == depth].mean()) for depth in unique_depths]
    medians = [float(np.median(radii[depths == depth])) for depth in unique_depths]
    ax.plot(unique_depths, means, color=CORAL, linewidth=2.4, marker="o", markersize=7, label="mean radius", zorder=4)
    ax.plot(unique_depths, medians, color=GOLD, linewidth=2.0, marker="D", markersize=5.5, label="median radius", zorder=4, alpha=0.95)

    ax.set_xlim(min(unique_depths) - 0.55, max(unique_depths) + 0.55)
    ax.set_xlabel("Tree depth", fontsize=11.5, color=TEXT)
    ax.set_ylabel("Poincare radius", fontsize=11.5, color=TEXT)
    ax.tick_params(labelsize=10, colors=TEXT)
    ax.grid(axis="y", color=EDGE, linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#B5C4D4")
    ax.spines["bottom"].set_color("#B5C4D4")
    ax.legend(frameon=False, loc="upper left", fontsize=10)

    radius_metrics = metrics["radius_summary"]
    draw_note_card(
        note_ax,
        0.56,
        0.38,
        "Depth statistics",
        [
            f"Depth-radius Spearman: {metrics['depth_radius']['spearman']:.4f}",
            f"Leaf mean radius: {radius_metrics['leaf']['mean']:.4f}",
            f"Internal mean radius: {radius_metrics['internal']['mean']:.4f}",
        ],
        CORAL,
    )
    draw_note_card(
        note_ax,
        0.10,
        0.34,
        "Interpretation",
        [
            "Leaves sit farther out than internal nodes.",
            "Adjacent shells still overlap.",
        ],
        CYAN,
    )
    save_panel(fig, output_base)


def render_loss_panel(output_base: Path) -> None:
    fig = plt.figure(figsize=(12.5, 8.0))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.96,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.05, 0.93, "Loss design", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.885, wrap_text("The hybrid objective combines relation reconstruction with global structure alignment and radial ordering.", 90), fontsize=11.5, color=MUTED)
    fig.text(
        0.05,
        0.80,
        "L_total = L_edge + α L_CPCC + β L_radial",
        fontsize=24,
        color=TEXT,
    )

    cards = [
        (
            0.05,
            0.58,
            0.90,
            0.12,
            TEAL,
            "L_edge",
            "L_edge = CE(-[d_B(u,v+), d_B(u,v1-), ...], y=0)",
            ["Original Poincare relation loss", "Branch discrimination + ancestor reconstruction"],
        ),
        (
            0.05,
            0.41,
            0.90,
            0.12,
            GOLD,
            "L_CPCC",
            "L_CPCC = 1 - corr({d_B(μ_gi, μ_gj)}, {d_tree(gi, gj)})",
            ["Borrowed from HypStructure", "Aligns embedding geometry with tree structure"],
        ),
        (
            0.05,
            0.24,
            0.90,
            0.12,
            CORAL,
            "L_radial",
            "L_radial = (1/|E|) Σ_(c,p∈E) max(0, r(p)+δ-r(c)),  r(x)=||x||",
            ["Pushes children farther from the origin", "Targets radial depth ordering"],
        ),
    ]

    for x, y, w, h, color, label, formula, lines in cards:
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                edgecolor=EDGE,
                facecolor="#FBFDFF",
            )
        )
        fig.add_artist(Rectangle((x, y + h - 0.035), w, 0.035, transform=fig.transFigure, color=color, alpha=0.96))
        fig.text(x + 0.02, y + h - 0.047, label, fontsize=12.5, color=NAVY, va="top", fontweight="bold")
        fig.text(x + 0.16, y + h - 0.047, wrap_text(formula, 62), fontsize=13.5, color=TEXT, va="top", linespacing=1.25)
        fig.text(x + 0.16, y + 0.018, wrap_text(lines[0], 42), fontsize=10.6, color=MUTED, va="bottom")
        fig.text(x + 0.56, y + 0.018, wrap_text(lines[1], 38), fontsize=10.6, color=MUTED, va="bottom")

    fig.text(0.05, 0.14, "Design rationale", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(
        0.05,
        0.08,
        wrap_text(
            "The old HypStructure-style setup produced clear level structure but weak branch separation. "
            "The new hybrid objective keeps CPCC for global hierarchy shape and adds the original "
            "Poincare edge loss so branch geometry becomes visually and quantitatively distinct.",
            94,
        ),
        fontsize=11.2,
        color=TEXT,
        va="top",
        linespacing=1.35,
    )
    save_panel(fig, output_base)


def render_metrics_panel(output_base: Path, metrics: dict[str, object]) -> None:
    fig = plt.figure(figsize=(12.5, 8.0))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.96,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.05, 0.93, "Quantitative summary", fontsize=22, fontweight="bold", color=NAVY)
    fig.text(0.05, 0.885, "The hybrid model recovers hierarchy and branch structure simultaneously, with radial depth ordering still improving.", fontsize=11.5, color=MUTED)

    stats = [
        ("Reconstruction MAP", f"{metrics['reconstruction']['map_rank']:.4f}", TEAL),
        ("Parent mean rank", f"{metrics['parent_ranking']['mean_rank']:.4f}", GOLD),
        ("Ancestor MAP", f"{metrics['ancestor_ranking']['mean_average_precision']:.4f}", CYAN),
        ("Depth-radius Spearman", f"{metrics['depth_radius']['spearman']:.4f}", CORAL),
        ("Within-branch dist.", f"{metrics['branch_separation']['within_branch']['mean']:.4f}", TEAL),
        ("Across-branch dist.", f"{metrics['branch_separation']['across_branch']['mean']:.4f}", GOLD),
    ]

    for idx, (label, value, color) in enumerate(stats):
        col = idx % 3
        row = idx // 3
        x = 0.05 + col * 0.305
        y = 0.56 - row * 0.26
        w = 0.27
        h = 0.18
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.012,rounding_size=0.02",
                linewidth=1.0,
                edgecolor=EDGE,
                facecolor="#FBFDFF",
            )
        )
        fig.add_artist(Rectangle((x, y + h - 0.03), w, 0.03, transform=fig.transFigure, color=color, alpha=0.96))
        fig.text(x + 0.02, y + 0.11, label, fontsize=11.2, color=MUTED)
        fig.text(x + 0.02, y + 0.045, value, fontsize=22, fontweight="bold", color=NAVY)

    fig.text(0.05, 0.18, "Takeaway", fontsize=13, fontweight="bold", color=NAVY)
    fig.text(
        0.05,
        0.11,
        wrap_text(
            "Branch structure is now convincing: siblings are compact, across-branch distances are larger, "
            "and parent ranking stays close to 1. The main remaining limitation is incomplete separation "
            "between adjacent depth shells in radius.",
            94,
        ),
        fontsize=11.1,
        color=TEXT,
        va="top",
        linespacing=1.35,
    )
    save_panel(fig, output_base)


def style_short_card(fig: plt.Figure, title: str, subtitle: str) -> tuple[plt.Axes, plt.Axes]:
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.012, 0.05),
            0.976,
            0.90,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.04, 0.88, title, fontsize=20, fontweight="bold", color=NAVY, ha="left", va="center")
    fig.text(0.04, 0.82, wrap_text(subtitle, 110), fontsize=10.8, color=MUTED, ha="left", va="center", linespacing=1.25)
    plot_ax = fig.add_axes([0.04, 0.15, 0.62, 0.60], zorder=2)
    note_ax = fig.add_axes([0.69, 0.16, 0.26, 0.60], zorder=2)
    note_ax.axis("off")
    return plot_ax, note_ax


def style_normal_card(fig: plt.Figure, title: str, subtitle: str) -> tuple[plt.Axes, plt.Axes]:
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.96,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.05, 0.93, title, fontsize=21, fontweight="bold", color=NAVY, ha="left", va="center")
    fig.text(0.05, 0.875, wrap_text(subtitle, 68), fontsize=11.0, color=MUTED, ha="left", va="center", linespacing=1.25)
    plot_ax = fig.add_axes([0.06, 0.14, 0.58, 0.66], zorder=2)
    note_ax = fig.add_axes([0.69, 0.16, 0.25, 0.62], zorder=2)
    note_ax.axis("off")
    return plot_ax, note_ax


def draw_short_sidebox(
    ax: plt.Axes,
    y: float,
    title: str,
    lines: list[str],
    accent: str,
    *,
    height: float = 0.39,
) -> None:
    patch = FancyBboxPatch(
        (0.0, y),
        1.0,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor="#FBFDFF",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((0.0, y + height - 0.07), 1.0, 0.07, transform=ax.transAxes, color=accent, alpha=0.95))
    ax.text(0.06, y + height - 0.11, title, transform=ax.transAxes, ha="left", va="top", fontsize=11.0, fontweight="bold", color=NAVY)
    current_y = y + height - 0.19
    for line in lines:
        wrapped = wrap_text(line, 32).split("\n")
        ax.text(0.07, current_y, f"• {wrapped[0]}", transform=ax.transAxes, ha="left", va="top", fontsize=9.6, color=TEXT)
        current_y -= 0.060
        for continuation in wrapped[1:]:
            ax.text(0.105, current_y, continuation, transform=ax.transAxes, ha="left", va="top", fontsize=9.6, color=TEXT)
            current_y -= 0.050
        current_y -= 0.030


def estimate_short_sidebox_height(lines: list[str], *, wrap_width: int = 32) -> float:
    height = 0.19
    for line in lines:
        wrapped = wrap_text(line, wrap_width).split("\n")
        height += 0.060
        if len(wrapped) > 1:
            height += 0.050 * (len(wrapped) - 1)
        height += 0.030
    height += 0.05
    return height


def estimate_sidebox_height(lines: list[str], *, wrap_width: int) -> float:
    height = 0.19
    for line in lines:
        wrapped = wrap_text(line, wrap_width).split("\n")
        height += 0.058
        if len(wrapped) > 1:
            height += 0.046 * (len(wrapped) - 1)
        height += 0.028
    height += 0.05
    return height


def draw_normal_sidebox(
    ax: plt.Axes,
    y: float,
    title: str,
    lines: list[str],
    accent: str,
    *,
    height: float,
    wrap_width: int = 24,
) -> None:
    patch = FancyBboxPatch(
        (0.0, y),
        1.0,
        height,
        transform=ax.transAxes,
        boxstyle="round,pad=0.016,rounding_size=0.03",
        linewidth=1.0,
        edgecolor=EDGE,
        facecolor="#FBFDFF",
    )
    ax.add_patch(patch)
    ax.add_patch(Rectangle((0.0, y + height - 0.07), 1.0, 0.07, transform=ax.transAxes, color=accent, alpha=0.95))
    ax.text(0.06, y + height - 0.11, title, transform=ax.transAxes, ha="left", va="top", fontsize=11.4, fontweight="bold", color=NAVY)
    current_y = y + height - 0.19
    for line in lines:
        wrapped = wrap_text(line, wrap_width).split("\n")
        ax.text(0.07, current_y, f"• {wrapped[0]}", transform=ax.transAxes, ha="left", va="top", fontsize=10.0, color=TEXT)
        current_y -= 0.060
        for continuation in wrapped[1:]:
            ax.text(0.105, current_y, continuation, transform=ax.transAxes, ha="left", va="top", fontsize=10.0, color=TEXT)
            current_y -= 0.048
        current_y -= 0.028


def render_branch_panel_short(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    parents = {
        row["node_id"]: row["parent_id"]
        for row in metadata_map.values()
        if row["parent_id"] and row["parent_id"] in node_to_index
    }
    embedding_2d = embeddings.copy() if embeddings.shape[1] == 2 else project_to_unit_disk(pca_to_2d(embeddings))
    embedding_2d = embedding_2d - embedding_2d.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(embedding_2d, axis=1).max()
    if max_norm > 0:
        embedding_2d = embedding_2d * (0.88 / max_norm)
    branch_ids = sorted({metadata_map[node_id]["top_branch_id"] for node_id in objects})
    branch_to_value = {branch_id: index for index, branch_id in enumerate(branch_ids)}
    branch_values = np.asarray([branch_to_value[metadata_map[node_id]["top_branch_id"]] for node_id in objects])

    fig = plt.figure(figsize=(14.5, 4.8))
    ax, note_ax = style_short_card(
        fig,
        "Branch-aware hyperbolic geometry",
        "Display-normalized Poincare disk view of the disease-90 hybrid embedding, colored by top-level branch.",
    )
    ax.set_position([0.08, 0.04, 0.58, 0.74])
    note_ax.set_position([0.76, 0.12, 0.19, 0.70])
    for radius, alpha in [(1.0, 0.9), (0.66, 0.18), (0.33, 0.12)]:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=NAVY,
                linewidth=1.2 if radius == 1.0 else 0.8,
                alpha=alpha,
                linestyle="-" if radius == 1.0 else "--",
            )
        )
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.30,
            alpha=0.11,
            color="#90A4B8",
            zorder=1,
        )
    colors = [BRANCH_COLORS[value % len(BRANCH_COLORS)] for value in branch_values]
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=16, alpha=0.82, edgecolors="white", linewidths=0.22, zorder=2)
    branch_centroids = []
    for branch_id in branch_ids:
        if branch_id == "90":
            continue
        idx = [node_to_index[node_id] for node_id in objects if metadata_map[node_id]["top_branch_id"] == branch_id]
        centroid = embedding_2d[idx].mean(axis=0)
        branch_centroids.append((branch_id, centroid))
    label_candidates = sorted(
        branch_centroids,
        key=lambda item: len([node_id for node_id in objects if metadata_map[node_id]["top_branch_id"] == item[0]]),
        reverse=True,
    )[:6]
    for branch_id, centroid in label_candidates:
        color = BRANCH_COLORS[branch_to_value[branch_id] % len(BRANCH_COLORS)]
        label = metadata_map[branch_id]["coding"].replace("Block ", "")
        ax.scatter([centroid[0]], [centroid[1]], s=70, facecolors="white", edgecolors=color, linewidths=1.2, zorder=4)
        ax.annotate(
            label,
            xy=(centroid[0], centroid[1]),
            xytext=(10, 8 if centroid[1] <= 0 else -10),
            textcoords="offset points",
            fontsize=8.8,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.96},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
            zorder=5,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")
    branch_metrics = metrics["branch_separation"]
    what_lines = [
        "Distinct branch regions in the Poincare disk.",
        "Parent-child geometry remains coherent.",
    ]
    key_lines = [
        f"Within branch mean: {branch_metrics['within_branch']['mean']:.4f}",
        f"Across branch mean: {branch_metrics['across_branch']['mean']:.4f}",
    ]
    what_height = estimate_short_sidebox_height(what_lines)
    key_height = estimate_short_sidebox_height(key_lines)
    top_margin = 0.03
    gap = 0.07
    bottom_margin = 0.03
    what_y = 1.0 - top_margin - what_height
    key_y = what_y - gap - key_height
    if key_y < bottom_margin:
        key_y = bottom_margin
    draw_short_sidebox(
        note_ax,
        what_y,
        "What it shows",
        what_lines,
        TEAL,
        height=what_height,
    )
    draw_short_sidebox(
        note_ax,
        key_y,
        "Key numbers",
        key_lines,
        GOLD,
        height=key_height,
    )
    save_panel(fig, output_base)


def render_branch_panel_normal(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    node_to_index = {node_id: index for index, node_id in enumerate(objects)}
    parents = {
        row["node_id"]: row["parent_id"]
        for row in metadata_map.values()
        if row["parent_id"] and row["parent_id"] in node_to_index
    }
    embedding_2d = embeddings.copy() if embeddings.shape[1] == 2 else project_to_unit_disk(pca_to_2d(embeddings))
    embedding_2d = embedding_2d - embedding_2d.mean(axis=0, keepdims=True)
    max_norm = np.linalg.norm(embedding_2d, axis=1).max()
    if max_norm > 0:
        embedding_2d = embedding_2d * (0.88 / max_norm)
    branch_ids = sorted({metadata_map[node_id]["top_branch_id"] for node_id in objects})
    branch_to_value = {branch_id: index for index, branch_id in enumerate(branch_ids)}
    branch_values = np.asarray([branch_to_value[metadata_map[node_id]["top_branch_id"]] for node_id in objects])

    fig = plt.figure(figsize=(9.2, 6.8))
    ax, note_ax = style_normal_card(
        fig,
        "Branch-aware hyperbolic geometry",
        "Display-normalized Poincare disk view of the disease-90 hybrid embedding, colored by top-level branch.",
    )
    ax.set_position([0.06, 0.13, 0.60, 0.64])
    note_ax.set_position([0.70, 0.16, 0.24, 0.60])
    for radius, alpha in [(1.0, 0.9), (0.66, 0.18), (0.33, 0.12)]:
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                radius,
                facecolor="none",
                edgecolor=NAVY,
                linewidth=1.2 if radius == 1.0 else 0.8,
                alpha=alpha,
                linestyle="-" if radius == 1.0 else "--",
            )
        )
    for child_id, parent_id in parents.items():
        child_index = node_to_index[child_id]
        parent_index = node_to_index[parent_id]
        ax.plot(
            [embedding_2d[child_index, 0], embedding_2d[parent_index, 0]],
            [embedding_2d[child_index, 1], embedding_2d[parent_index, 1]],
            linewidth=0.28,
            alpha=0.11,
            color="#90A4B8",
            zorder=1,
        )
    colors = [BRANCH_COLORS[value % len(BRANCH_COLORS)] for value in branch_values]
    ax.scatter(embedding_2d[:, 0], embedding_2d[:, 1], c=colors, s=18, alpha=0.82, edgecolors="white", linewidths=0.22, zorder=2)
    branch_centroids = []
    for branch_id in branch_ids:
        if branch_id == "90":
            continue
        idx = [node_to_index[node_id] for node_id in objects if metadata_map[node_id]["top_branch_id"] == branch_id]
        centroid = embedding_2d[idx].mean(axis=0)
        branch_centroids.append((branch_id, centroid))
    label_candidates = sorted(
        branch_centroids,
        key=lambda item: len([node_id for node_id in objects if metadata_map[node_id]["top_branch_id"] == item[0]]),
        reverse=True,
    )[:6]
    for branch_id, centroid in label_candidates:
        color = BRANCH_COLORS[branch_to_value[branch_id] % len(BRANCH_COLORS)]
        label = metadata_map[branch_id]["coding"].replace("Block ", "")
        ax.scatter([centroid[0]], [centroid[1]], s=70, facecolors="white", edgecolors=color, linewidths=1.2, zorder=4)
        ax.annotate(
            label,
            xy=(centroid[0], centroid[1]),
            xytext=(10, 8 if centroid[1] <= 0 else -10),
            textcoords="offset points",
            fontsize=9.0,
            color=TEXT,
            bbox={"boxstyle": "round,pad=0.16", "fc": "white", "ec": color, "alpha": 0.96},
            arrowprops={"arrowstyle": "-", "color": color, "lw": 0.9},
            zorder=5,
        )
    ax.set_aspect("equal")
    ax.set_xlim(-1.04, 1.04)
    ax.set_ylim(-1.04, 1.04)
    ax.axis("off")
    branch_metrics = metrics["branch_separation"]
    what_lines = [
        "Distinct branch regions in the Poincare disk.",
        "Parent-child geometry remains coherent.",
    ]
    key_lines = [
        f"Within branch mean: {branch_metrics['within_branch']['mean']:.4f}",
        f"Across branch mean: {branch_metrics['across_branch']['mean']:.4f}",
    ]
    wrap_width = 24
    what_height = estimate_sidebox_height(what_lines, wrap_width=wrap_width)
    key_height = estimate_sidebox_height(key_lines, wrap_width=wrap_width)
    top_margin = 0.03
    gap = 0.07
    bottom_margin = 0.03
    what_y = 1.0 - top_margin - what_height
    key_y = max(bottom_margin, what_y - gap - key_height)
    draw_normal_sidebox(note_ax, what_y, "What it shows", what_lines, TEAL, height=what_height, wrap_width=wrap_width)
    draw_normal_sidebox(note_ax, key_y, "Key numbers", key_lines, GOLD, height=key_height, wrap_width=wrap_width)
    save_panel(fig, output_base)


def render_depth_panel_short(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=int)
    radii = np.linalg.norm(embeddings, axis=1)
    unique_depths = sorted(np.unique(depths))

    fig = plt.figure(figsize=(14.5, 4.8))
    ax, note_ax = style_short_card(
        fig,
        "Radial depth ordering",
        "Radius should increase with hierarchy depth; the hybrid model improves this trend but adjacent shells still overlap.",
    )
    for depth in unique_depths:
        ax.axvspan(depth - 0.45, depth + 0.45, color="#F4F8FB" if depth % 2 == 0 else "#FAFCFE", zorder=0)
    positions = []
    point_colors = []
    for depth in unique_depths:
        indices = np.where(depths == depth)[0]
        jitter = np.linspace(-0.16, 0.16, num=len(indices)) if len(indices) > 1 else np.array([0.0])
        positions.extend(depth + jitter)
        point_colors.extend([TEAL if depth > 0 else NAVY] * len(indices))
    ax.scatter(positions, radii, s=22, c=point_colors, alpha=0.68, edgecolors="white", linewidths=0.24, zorder=3)
    means = [float(radii[depths == depth].mean()) for depth in unique_depths]
    medians = [float(np.median(radii[depths == depth])) for depth in unique_depths]
    ax.plot(unique_depths, means, color=CORAL, linewidth=2.1, marker="o", markersize=6, label="mean radius", zorder=4)
    ax.plot(unique_depths, medians, color=GOLD, linewidth=1.8, marker="D", markersize=5, label="median radius", zorder=4, alpha=0.95)
    ax.set_xlim(min(unique_depths) - 0.55, max(unique_depths) + 0.55)
    ax.set_xlabel("Tree depth", fontsize=10.8, color=TEXT)
    ax.set_ylabel("Poincare radius", fontsize=10.8, color=TEXT)
    ax.tick_params(labelsize=9.5, colors=TEXT)
    ax.grid(axis="y", color=EDGE, linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#B5C4D4")
    ax.spines["bottom"].set_color("#B5C4D4")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    radius_metrics = metrics["radius_summary"]
    stats_lines = [
        f"Spearman: {metrics['depth_radius']['spearman']:.4f}",
        f"Leaf mean radius: {radius_metrics['leaf']['mean']:.4f}",
    ]
    interp_lines = [
        "Leaves sit farther out than internal nodes.",
        "Adjacent shells still overlap.",
    ]
    stats_height = estimate_short_sidebox_height(stats_lines)
    interp_height = estimate_short_sidebox_height(interp_lines)
    top_margin = 0.03
    gap = 0.07
    bottom_margin = 0.03
    stats_y = 1.0 - top_margin - stats_height
    interp_y = stats_y - gap - interp_height
    if interp_y < bottom_margin:
        interp_y = bottom_margin
    draw_short_sidebox(
        note_ax,
        stats_y,
        "Depth statistics",
        stats_lines,
        CORAL,
        height=stats_height,
    )
    draw_short_sidebox(
        note_ax,
        interp_y,
        "Interpretation",
        interp_lines,
        CYAN,
        height=interp_height,
    )
    save_panel(fig, output_base)


def render_depth_panel_normal(
    embeddings: np.ndarray,
    objects: list[str],
    metadata_map: dict[str, dict[str, str]],
    output_base: Path,
    metrics: dict[str, object],
) -> None:
    depths = np.asarray([int(metadata_map[node_id]["depth"]) for node_id in objects], dtype=int)
    radii = np.linalg.norm(embeddings, axis=1)
    unique_depths = sorted(np.unique(depths))

    fig = plt.figure(figsize=(9.2, 6.8))
    ax, note_ax = style_normal_card(
        fig,
        "Radial depth ordering",
        "Radius should increase with hierarchy depth; the hybrid model improves this trend but adjacent shells still overlap.",
    )
    for depth in unique_depths:
        ax.axvspan(depth - 0.45, depth + 0.45, color="#F4F8FB" if depth % 2 == 0 else "#FAFCFE", zorder=0)
    positions = []
    point_colors = []
    for depth in unique_depths:
        indices = np.where(depths == depth)[0]
        jitter = np.linspace(-0.16, 0.16, num=len(indices)) if len(indices) > 1 else np.array([0.0])
        positions.extend(depth + jitter)
        point_colors.extend([TEAL if depth > 0 else NAVY] * len(indices))
    ax.scatter(positions, radii, s=24, c=point_colors, alpha=0.68, edgecolors="white", linewidths=0.24, zorder=3)
    means = [float(radii[depths == depth].mean()) for depth in unique_depths]
    medians = [float(np.median(radii[depths == depth])) for depth in unique_depths]
    ax.plot(unique_depths, means, color=CORAL, linewidth=2.1, marker="o", markersize=6, label="mean radius", zorder=4)
    ax.plot(unique_depths, medians, color=GOLD, linewidth=1.8, marker="D", markersize=5, label="median radius", zorder=4, alpha=0.95)
    ax.set_xlim(min(unique_depths) - 0.55, max(unique_depths) + 0.55)
    ax.set_xlabel("Tree depth", fontsize=11.0, color=TEXT)
    ax.set_ylabel("Poincare radius", fontsize=11.0, color=TEXT)
    ax.tick_params(labelsize=9.8, colors=TEXT)
    ax.grid(axis="y", color=EDGE, linewidth=0.8, alpha=0.8)
    ax.spines[["top", "right"]].set_visible(False)
    ax.spines["left"].set_color("#B5C4D4")
    ax.spines["bottom"].set_color("#B5C4D4")
    ax.legend(frameon=False, loc="upper left", fontsize=9.5)
    radius_metrics = metrics["radius_summary"]
    stats_lines = [
        f"Spearman: {metrics['depth_radius']['spearman']:.4f}",
        f"Leaf mean radius: {radius_metrics['leaf']['mean']:.4f}",
    ]
    interp_lines = [
        "Leaves sit farther out than internal nodes.",
        "Adjacent shells still overlap.",
    ]
    wrap_width = 23
    stats_height = estimate_sidebox_height(stats_lines, wrap_width=wrap_width)
    interp_height = estimate_sidebox_height(interp_lines, wrap_width=wrap_width)
    top_margin = 0.03
    gap = 0.07
    bottom_margin = 0.03
    stats_y = 1.0 - top_margin - stats_height
    interp_y = max(bottom_margin, stats_y - gap - interp_height)
    draw_normal_sidebox(note_ax, stats_y, "Depth statistics", stats_lines, CORAL, height=stats_height, wrap_width=wrap_width)
    draw_normal_sidebox(note_ax, interp_y, "Interpretation", interp_lines, CYAN, height=interp_height, wrap_width=wrap_width)
    save_panel(fig, output_base)


def render_loss_panel_short(output_base: Path) -> None:
    fig = plt.figure(figsize=(14.5, 4.4))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.012, 0.05),
            0.976,
            0.90,
            transform=fig.transFigure,
            boxstyle="round,pad=0.010,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.04, 0.885, "Loss design", fontsize=20, fontweight="bold", color=NAVY)
    fig.text(
        0.04,
        0.785,
        wrap_text("The hybrid objective combines relation reconstruction with global structure alignment and radial ordering.", 92),
        fontsize=10.6,
        color=MUTED,
        linespacing=1.25,
    )
    fig.text(0.04, 0.635, "L_total = L_edge + α L_CPCC + β L_radial", fontsize=19.5, color=TEXT)

    rows = [
        (0.04, 0.420, TEAL, "L_edge", "CE(-[d_B(u,v+), d_B(u,v1-), ...], y=0)", "Poincare relation loss", "Branch discrimination + ancestor reconstruction"),
        (0.04, 0.235, GOLD, "L_CPCC", "1 - corr({d_B(μ_gi, μ_gj)}, {d_tree(gi, gj)})", "Borrowed from HypStructure", "Aligns embedding geometry with tree structure"),
        (0.04, 0.050, CORAL, "L_radial", "(1/|E|) Σ max(0, r(p)+δ-r(c))", "Child farther from origin", "Targets radial depth ordering"),
    ]
    for x, y, color, label, formula, left_note, right_note in rows:
        w, h = 0.92, 0.122
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=EDGE,
                facecolor="#FBFDFF",
            )
        )
        fig.add_artist(Rectangle((x, y + h - 0.026), w, 0.026, transform=fig.transFigure, color=color, alpha=0.96))
        fig.text(x + 0.018, y + 0.072, label, fontsize=11.8, fontweight="bold", color=NAVY, va="center")
        fig.text(x + 0.018, y + 0.002, left_note, fontsize=8.4, color=MUTED, va="bottom")
        fig.text(x + 0.13, y + 0.060, formula, fontsize=11.2, color=TEXT, va="center")
        fig.text(x + 0.60, y + 0.032, right_note, fontsize=8.5, color=MUTED, va="bottom")
    save_panel(fig, output_base)


def render_loss_panel_normal(output_base: Path) -> None:
    fig = plt.figure(figsize=(7.6, 7.1))
    fig.patch.set_facecolor(LIGHT_BG)
    fig.add_artist(
        FancyBboxPatch(
            (0.015, 0.02),
            0.97,
            0.96,
            transform=fig.transFigure,
            boxstyle="round,pad=0.012,rounding_size=0.02",
            linewidth=1.2,
            edgecolor=EDGE,
            facecolor=CARD_BG,
            zorder=0,
        )
    )
    fig.text(0.05, 0.93, "Loss design", fontsize=21, fontweight="bold", color=NAVY)
    fig.text(
        0.05,
        0.855,
        wrap_text("The hybrid objective combines relation reconstruction with global structure alignment and radial ordering.", 54),
        fontsize=11.0,
        color=MUTED,
        linespacing=1.25,
    )
    fig.text(0.05, 0.73, "L_total = L_edge + α L_CPCC + β L_radial", fontsize=18.8, color=TEXT)

    rows = [
        (0.09, 0.49, TEAL, "L_edge", "CE(-[d_B(u,v+), d_B(u,v1-), ...], y=0)", "Poincare relation loss", "Branch discrimination + ancestor reconstruction"),
        (0.09, 0.29, GOLD, "L_CPCC", "1 - corr({d_B(μ_gi, μ_gj)}, {d_tree(gi, gj)})", "Borrowed from HypStructure", "Aligns embedding geometry with tree structure"),
        (0.09, 0.09, CORAL, "L_radial", "(1/|E|) Σ max(0, r(p)+δ-r(c))", "Child farther from origin", "Targets radial depth ordering"),
    ]
    for x, y, color, label, formula, left_note, right_note in rows:
        w, h = 0.82, 0.13
        fig.add_artist(
            FancyBboxPatch(
                (x, y),
                w,
                h,
                transform=fig.transFigure,
                boxstyle="round,pad=0.010,rounding_size=0.018",
                linewidth=1.0,
                edgecolor=EDGE,
                facecolor="#FBFDFF",
            )
        )
        fig.add_artist(Rectangle((x, y + h - 0.026), w, 0.026, transform=fig.transFigure, color=color, alpha=0.96))
        fig.text(x + 0.022, y + 0.078, label, fontsize=12.0, fontweight="bold", color=NAVY, va="center")
        fig.text(x + 0.022, y + 0.020, left_note, fontsize=8.8, color=MUTED, va="bottom")
        fig.text(x + 0.26, y + 0.078, wrap_text(formula, 32), fontsize=11.0, color=TEXT, va="center", linespacing=1.1)
        fig.text(x + 0.58, y + 0.048, wrap_text(right_note, 18), fontsize=9.0, color=MUTED, va="center", linespacing=1.18)
    save_panel(fig, output_base)


def main() -> None:
    parser = argparse.ArgumentParser(description="Render poster-ready panels for the disease-90 hyperbolic embedding section.")
    parser.add_argument("--checkpoint", type=Path, default=Path(f"{DEFAULT_CHECKPOINT}.best"))
    parser.add_argument("--metadata-tsv", type=Path, default=DEFAULT_METADATA_TSV)
    parser.add_argument("--eval-json", type=Path, default=DEFAULT_EVAL_JSON)
    parser.add_argument("--output-dir", type=Path, default=POSTER_DIR / "embedding_panels")
    args = parse_args_with_defaults(parser)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    embeddings, objects, _ = checkpoint_embeddings_and_objects(args.checkpoint)
    metadata_rows = read_metadata_tsv(args.metadata_tsv)
    metadata_map = {row["node_id"]: row for row in metadata_rows}
    metrics = json.loads(args.eval_json.read_text(encoding="utf-8"))

    render_branch_panel(embeddings, objects, metadata_map, args.output_dir / "embedding_branch_structure", metrics)
    render_depth_panel(embeddings, objects, metadata_map, args.output_dir / "embedding_depth_ordering", metrics)
    render_loss_panel(args.output_dir / "embedding_loss_design")
    render_metrics_panel(args.output_dir / "embedding_quant_summary", metrics)
    short_dir = args.output_dir / "short"
    short_dir.mkdir(parents=True, exist_ok=True)
    render_branch_panel_short(embeddings, objects, metadata_map, short_dir / "embedding_branch_structure_short", metrics)
    render_depth_panel_short(embeddings, objects, metadata_map, short_dir / "embedding_depth_ordering_short", metrics)
    render_loss_panel_short(short_dir / "embedding_loss_design_short")
    normal_dir = args.output_dir / "normal"
    normal_dir.mkdir(parents=True, exist_ok=True)
    render_branch_panel_normal(embeddings, objects, metadata_map, normal_dir / "embedding_branch_structure_normal", metrics)
    render_depth_panel_normal(embeddings, objects, metadata_map, normal_dir / "embedding_depth_ordering_normal", metrics)
    render_loss_panel_normal(normal_dir / "embedding_loss_design_normal")
    print(f"Wrote poster panels to {args.output_dir}")


if __name__ == "__main__":
    main()
