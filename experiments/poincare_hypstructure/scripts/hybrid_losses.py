#!/usr/bin/env python3

from collections import defaultdict

import torch
from torch import nn

from disease90_common import (
    build_tree_helpers,
    condensed_poincare_distance_torch,
    poincare_mean_torch,
    project_to_ball_torch,
    torch_pearson_corr,
    tree_distance,
)


class GlobalHierarchyCPCCLoss(nn.Module):
    def __init__(self, metadata_rows, objects, min_group_size: int = 2):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        _, children, depth_map, path_to_root, descendants = build_tree_helpers(metadata_rows)

        group_ids = []
        member_tensors = []
        for row in sorted(metadata_rows, key=lambda item: (int(item["depth"]), item["node_id"])):
            node_id = row["node_id"]
            members = [node_to_index[member] for member in descendants[node_id] if member in node_to_index]
            if len(members) >= min_group_size:
                group_ids.append(node_id)
                member_tensors.append(torch.tensor(members, dtype=torch.long))

        pairwise_tree_distances = []
        for left_index in range(len(group_ids)):
            for right_index in range(left_index + 1, len(group_ids)):
                pairwise_tree_distances.append(
                    float(tree_distance(group_ids[left_index], group_ids[right_index], depth_map, path_to_root))
                )

        self.group_member_tensors = member_tensors
        self.group_ids = group_ids
        self.register_buffer("target_tree_distances", torch.tensor(pairwise_tree_distances, dtype=torch.double))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if len(self.group_member_tensors) < 2:
            return embeddings.new_tensor(0.0)

        projected = project_to_ball_torch(embeddings)
        representatives = []
        for member_indices in self.group_member_tensors:
            members = projected.index_select(0, member_indices.to(projected.device))
            representatives.append(poincare_mean_torch(members, dim=0))
        representatives = torch.stack(representatives, dim=0)
        pairwise_embedding_distances = condensed_poincare_distance_torch(representatives)
        target_distances = self.target_tree_distances.to(pairwise_embedding_distances.device)
        corr = torch_pearson_corr(pairwise_embedding_distances, target_distances)
        return 1.0 - corr


class RadialOrderLoss(nn.Module):
    def __init__(self, metadata_rows, objects, margin: float = 0.02):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        child_indices = []
        parent_indices = []
        for row in metadata_rows:
            parent_id = row["parent_id"]
            node_id = row["node_id"]
            if parent_id and parent_id != "0" and node_id in node_to_index and parent_id in node_to_index:
                child_indices.append(node_to_index[node_id])
                parent_indices.append(node_to_index[parent_id])
        self.margin = margin
        self.register_buffer("child_indices", torch.tensor(child_indices, dtype=torch.long))
        self.register_buffer("parent_indices", torch.tensor(parent_indices, dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.child_indices.numel() == 0:
            return embeddings.new_tensor(0.0)
        projected = project_to_ball_torch(embeddings)
        radii = torch.linalg.vector_norm(projected, dim=-1)
        child_radii = radii.index_select(0, self.child_indices.to(radii.device))
        parent_radii = radii.index_select(0, self.parent_indices.to(radii.device))
        return torch.relu(parent_radii + self.margin - child_radii).mean()
