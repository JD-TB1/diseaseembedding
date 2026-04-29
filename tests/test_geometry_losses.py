from __future__ import annotations

import sys
from pathlib import Path

import torch


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "experiments" / "poincare_hypstructure" / "scripts"
sys.path.insert(0, str(SCRIPT_DIR))

from hybrid_losses import (  # noqa: E402
    BranchAngularSeparationLoss,
    BranchContrastiveMarginLoss,
    BranchTeacherLayoutLoss,
    DepthBandLoss,
    DepthQuantileMarginLoss,
)


def test_depth_band_loss_is_near_zero_for_target_radii() -> None:
    objects = ["0", "1", "2", "3", "4"]
    metadata_rows = [
        {"node_id": node_id, "parent_id": str(index - 1), "depth": str(index), "top_branch_id": "1"}
        for index, node_id in enumerate(objects)
    ]
    embeddings = torch.tensor(
        [[0.05, 0.0], [0.18, 0.0], [0.35, 0.0], [0.55, 0.0], [0.75, 0.0]],
        dtype=torch.double,
    )

    loss = DepthBandLoss(metadata_rows, objects)(embeddings)

    assert float(loss) < 1e-12


def test_depth_quantile_margin_loss_tracks_adjacent_depth_overlap() -> None:
    objects = ["root", "a", "b", "a_leaf", "b_leaf"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "top_branch_id": "root"},
        {"node_id": "a", "parent_id": "root", "depth": "1", "top_branch_id": "a"},
        {"node_id": "b", "parent_id": "root", "depth": "1", "top_branch_id": "b"},
        {"node_id": "a_leaf", "parent_id": "a", "depth": "2", "top_branch_id": "a"},
        {"node_id": "b_leaf", "parent_id": "b", "depth": "2", "top_branch_id": "b"},
    ]
    separated = torch.tensor(
        [[0.01, 0.0], [0.10, 0.0], [-0.11, 0.0], [0.30, 0.0], [-0.31, 0.0]],
        dtype=torch.double,
    )
    overlapping = torch.tensor(
        [[0.01, 0.0], [0.10, 0.0], [-0.30, 0.0], [0.20, 0.0], [-0.21, 0.0]],
        dtype=torch.double,
    )

    loss_fn = DepthQuantileMarginLoss(metadata_rows, objects, margin=0.01)

    assert float(loss_fn(separated)) < 1e-12
    assert float(loss_fn(overlapping)) > 0.04


def test_branch_angular_loss_rewards_coherent_separated_branches() -> None:
    objects = ["root", "a1", "a2", "b1", "b2"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "top_branch_id": "root"},
        {"node_id": "a1", "parent_id": "root", "depth": "1", "top_branch_id": "a1"},
        {"node_id": "a2", "parent_id": "a1", "depth": "2", "top_branch_id": "a1"},
        {"node_id": "b1", "parent_id": "root", "depth": "1", "top_branch_id": "b1"},
        {"node_id": "b2", "parent_id": "b1", "depth": "2", "top_branch_id": "b1"},
    ]
    coherent = torch.tensor(
        [[0.0, 0.0], [0.4, 0.0], [0.5, 0.0], [-0.4, 0.0], [-0.5, 0.0]],
        dtype=torch.double,
    )
    mixed = torch.tensor(
        [[0.0, 0.0], [0.4, 0.0], [-0.5, 0.0], [-0.4, 0.0], [0.5, 0.0]],
        dtype=torch.double,
    )

    loss_fn = BranchAngularSeparationLoss(metadata_rows, objects, cos_margin=0.2)

    assert float(loss_fn(coherent)) < float(loss_fn(mixed))


def test_branch_teacher_layout_loss_is_zero_for_matching_teacher() -> None:
    objects = ["root", "a1", "a2", "b1", "b2"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "top_branch_id": "root"},
        {"node_id": "a1", "parent_id": "root", "depth": "1", "top_branch_id": "a"},
        {"node_id": "a2", "parent_id": "a1", "depth": "2", "top_branch_id": "a"},
        {"node_id": "b1", "parent_id": "root", "depth": "1", "top_branch_id": "b"},
        {"node_id": "b2", "parent_id": "b1", "depth": "2", "top_branch_id": "b"},
    ]
    teacher = torch.tensor(
        [[0.0, 0.0], [0.20, 0.0], [0.30, 0.0], [0.0, 0.20], [0.0, 0.30]],
        dtype=torch.double,
    )
    swapped = torch.tensor(
        [[0.0, 0.0], [0.0, 0.20], [0.0, 0.30], [0.20, 0.0], [0.30, 0.0]],
        dtype=torch.double,
    )

    loss_fn = BranchTeacherLayoutLoss(metadata_rows, objects, teacher)

    assert float(loss_fn(teacher)) < 1e-12
    assert float(loss_fn(swapped)) > 0.5


def test_branch_contrastive_margin_loss_penalizes_mixed_same_depth_branches() -> None:
    objects = ["root", "a1", "a2", "b1", "b2"]
    metadata_rows = [
        {"node_id": "root", "parent_id": "0", "depth": "0", "top_branch_id": "root"},
        {"node_id": "a1", "parent_id": "root", "depth": "1", "top_branch_id": "a"},
        {"node_id": "a2", "parent_id": "root", "depth": "1", "top_branch_id": "a"},
        {"node_id": "b1", "parent_id": "root", "depth": "1", "top_branch_id": "b"},
        {"node_id": "b2", "parent_id": "root", "depth": "1", "top_branch_id": "b"},
    ]
    separated = torch.tensor(
        [[0.0, 0.0], [0.10, 0.0], [0.12, 0.0], [-0.10, 0.0], [-0.12, 0.0]],
        dtype=torch.double,
    )
    mixed = torch.tensor(
        [[0.0, 0.0], [0.10, 0.0], [0.12, 0.0], [0.11, 0.01], [0.13, -0.01]],
        dtype=torch.double,
    )

    loss_fn = BranchContrastiveMarginLoss(metadata_rows, objects, margin=0.20)

    assert float(loss_fn(separated)) < float(loss_fn(mixed))
