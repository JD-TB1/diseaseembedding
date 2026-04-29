from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "poincare_hypstructure" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from disease90_metrics import (  # noqa: E402
    compute_branch_geometry_metrics,
    compute_branch_pair_diagnostics,
    compute_radius_structure_metrics,
    gate_deficits,
)


def test_radius_metrics_capture_quantile_gaps_and_overlap() -> None:
    objects = ["root", "a", "b", "a_leaf", "b_leaf"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "selectable": "N", "top_branch_id": "root"},
        {"node_id": "a", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "a"},
        {"node_id": "b", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "b"},
        {"node_id": "a_leaf", "parent_id": "a", "depth": "2", "selectable": "Y", "top_branch_id": "a"},
        {"node_id": "b_leaf", "parent_id": "b", "depth": "2", "selectable": "Y", "top_branch_id": "b"},
    ]
    embeddings = np.asarray(
        [[0.05, 0.0], [0.30, 0.0], [-0.30, 0.0], [0.70, 0.0], [-0.70, 0.0]],
        dtype=np.float64,
    )

    metrics = compute_radius_structure_metrics(embeddings, objects, metadata_rows)

    assert metrics["positive_adjacent_depth_quantile_gap_count"] == 2
    assert all(value > 0.0 for value in metrics["adjacent_depth_quantile_gaps"].values())
    assert all(value == 0.0 for value in metrics["depth_overlap_rates"].values())


def test_branch_geometry_reports_separated_branch_silhouette() -> None:
    objects = ["root", "a", "b", "a_leaf", "b_leaf"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "selectable": "N", "top_branch_id": "root"},
        {"node_id": "a", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "a"},
        {"node_id": "b", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "b"},
        {"node_id": "a_leaf", "parent_id": "a", "depth": "2", "selectable": "Y", "top_branch_id": "a"},
        {"node_id": "b_leaf", "parent_id": "b", "depth": "2", "selectable": "Y", "top_branch_id": "b"},
    ]
    embeddings = np.asarray(
        [[0.05, 0.0], [0.30, 0.0], [-0.30, 0.0], [0.70, 0.0], [-0.70, 0.0]],
        dtype=np.float64,
    )

    metrics = compute_branch_geometry_metrics(embeddings, objects, metadata_rows)

    assert metrics["branch_silhouette"] > 0.45
    assert metrics["top_branch_centroid_distances"]["mean"] > 0.0


def test_branch_pair_diagnostics_capture_same_depth_and_angular_ratios() -> None:
    objects = ["root", "a1", "a2", "b1", "b2"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "selectable": "N", "top_branch_id": "root"},
        {"node_id": "a1", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "a"},
        {"node_id": "a2", "parent_id": "a1", "depth": "1", "selectable": "Y", "top_branch_id": "a"},
        {"node_id": "b1", "parent_id": "root", "depth": "1", "selectable": "N", "top_branch_id": "b"},
        {"node_id": "b2", "parent_id": "b1", "depth": "1", "selectable": "Y", "top_branch_id": "b"},
    ]
    embeddings = np.asarray(
        [[0.0, 0.0], [0.20, 0.0], [0.25, 0.0], [0.0, 0.20], [0.0, 0.25]],
        dtype=np.float64,
    )

    diagnostics = compute_branch_pair_diagnostics(embeddings, objects, metadata_rows)

    assert diagnostics["ratios"]["within_branch_to_across_branch_mean"] < 0.50
    assert diagnostics["ratios"]["same_depth_within_branch_to_across_branch_mean"] < 1.0
    assert diagnostics["ratios"]["angular_within_branch_to_across_branch_mean"] < 0.05


def test_gate_deficits_rank_metric_threshold_failures() -> None:
    passing_metrics = {
        "reconstruction": {"map_rank": 0.31},
        "parent_ranking": {"mean_rank": 1.10},
        "ancestor_ranking": {"mean_average_precision": 0.96},
        "ratios": {
            "sibling_to_non_sibling_mean": 0.25,
            "within_branch_to_across_branch_mean": 0.30,
        },
        "radius_structure": {
            "minimum_adjacent_gap": 0.001,
            "positive_adjacent_depth_quantile_gap_count": 3,
            "parent_child_radial_violation_rate": 0.05,
        },
    }
    failing_metrics = {
        "reconstruction": {"map_rank": 0.20},
        "parent_ranking": {"mean_rank": 1.50},
        "ancestor_ranking": {"mean_average_precision": 0.90},
        "ratios": {
            "sibling_to_non_sibling_mean": 0.40,
            "within_branch_to_across_branch_mean": 0.70,
        },
        "radius_structure": {
            "minimum_adjacent_gap": -0.002,
            "positive_adjacent_depth_quantile_gap_count": 1,
            "parent_child_radial_violation_rate": 0.20,
        },
    }

    assert gate_deficits(passing_metrics)["total"] == 0.0
    assert gate_deficits(failing_metrics)["total"] > gate_deficits(passing_metrics)["total"]
