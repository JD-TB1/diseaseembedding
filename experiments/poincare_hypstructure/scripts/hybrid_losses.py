#!/usr/bin/env python3

from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn

from disease90_common import (
    build_tree_helpers,
    condensed_poincare_distance_torch,
    poincare_distance_torch,
    poincare_mean_torch,
    project_to_ball_torch,
    torch_pearson_corr,
    tree_distance,
)

DEFAULT_DEPTH_BANDS = (0.05, 0.18, 0.35, 0.55, 0.75)


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


class DepthBandLoss(nn.Module):
    def __init__(
        self,
        metadata_rows,
        objects,
        target_radii: tuple[float, ...] = DEFAULT_DEPTH_BANDS,
    ):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        node_indices = []
        targets = []
        for row in metadata_rows:
            node_id = row["node_id"]
            if node_id not in node_to_index:
                continue
            depth = int(row["depth"])
            target = target_radii[min(depth, len(target_radii) - 1)]
            node_indices.append(node_to_index[node_id])
            targets.append(float(target))
        self.register_buffer("node_indices", torch.tensor(node_indices, dtype=torch.long))
        self.register_buffer("target_radii", torch.tensor(targets, dtype=torch.double))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.node_indices.numel() == 0:
            return embeddings.new_tensor(0.0)
        projected = project_to_ball_torch(embeddings)
        radii = torch.linalg.vector_norm(projected, dim=-1)
        selected = radii.index_select(0, self.node_indices.to(radii.device))
        targets = self.target_radii.to(device=radii.device, dtype=radii.dtype)
        return torch.mean((selected - targets) ** 2)


class DepthQuantileMarginLoss(nn.Module):
    def __init__(self, metadata_rows, objects, margin: float = 0.001):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        depth_to_indices = defaultdict(list)
        for row in metadata_rows:
            node_id = row["node_id"]
            if node_id in node_to_index:
                depth_to_indices[int(row["depth"])].append(node_to_index[node_id])
        self.margin = margin
        self.depth_tensors = [
            (depth, torch.tensor(indices, dtype=torch.long))
            for depth, indices in sorted(depth_to_indices.items())
            if indices
        ]

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if len(self.depth_tensors) < 2:
            return embeddings.new_tensor(0.0)
        projected = project_to_ball_torch(embeddings)
        radii = torch.linalg.vector_norm(projected, dim=-1)
        losses = []
        for (_, left_indices), (_, right_indices) in zip(self.depth_tensors[:-1], self.depth_tensors[1:]):
            left = radii.index_select(0, left_indices.to(radii.device))
            right = radii.index_select(0, right_indices.to(radii.device))
            left_high = torch.quantile(left, 0.90)
            right_low = torch.quantile(right, 0.10)
            losses.append(torch.relu(left_high + self.margin - right_low))
        return torch.stack(losses).mean() if losses else embeddings.new_tensor(0.0)


class BranchAngularSeparationLoss(nn.Module):
    def __init__(self, metadata_rows, objects, cos_margin: float = 0.2):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        branch_to_indices = defaultdict(list)
        for row in metadata_rows:
            node_id = row["node_id"]
            if node_id not in node_to_index or int(row["depth"]) == 0:
                continue
            branch_to_indices[row["top_branch_id"]].append(node_to_index[node_id])

        self.cos_margin = cos_margin
        self.branch_member_tensors = [
            torch.tensor(indices, dtype=torch.long)
            for _, indices in sorted(branch_to_indices.items())
            if indices
        ]

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.branch_member_tensors:
            return embeddings.new_tensor(0.0)

        projected = project_to_ball_torch(embeddings)
        norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1e-8)
        directions = projected / norms

        branch_centroids = []
        cohesion_losses = []
        for member_indices in self.branch_member_tensors:
            members = directions.index_select(0, member_indices.to(directions.device))
            centroid = F.normalize(members.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
            branch_centroids.append(centroid)
            cohesion_losses.append(1.0 - torch.sum(members * centroid, dim=-1).mean())

        cohesion = torch.stack(cohesion_losses).mean()
        if len(branch_centroids) < 2:
            return cohesion

        centroids = torch.stack(branch_centroids, dim=0)
        cosine_matrix = centroids @ centroids.T
        pair_indices = torch.triu_indices(centroids.size(0), centroids.size(0), offset=1, device=centroids.device)
        pairwise_cosine = cosine_matrix[pair_indices[0], pair_indices[1]]
        separation = torch.relu(pairwise_cosine - self.cos_margin).mean()
        return cohesion + separation


class BranchTeacherLayoutLoss(nn.Module):
    def __init__(self, metadata_rows, objects, teacher_embeddings: torch.Tensor):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        branch_to_indices = defaultdict(list)
        for row in metadata_rows:
            node_id = row["node_id"]
            if node_id not in node_to_index or int(row["depth"]) == 0:
                continue
            branch_to_indices[row["top_branch_id"]].append(node_to_index[node_id])

        projected_teacher = project_to_ball_torch(teacher_embeddings.detach().clone().double())
        teacher_norms = torch.linalg.vector_norm(projected_teacher, dim=-1, keepdim=True).clamp_min(1e-8)
        teacher_directions = projected_teacher / teacher_norms

        member_tensors = []
        teacher_centroids = []
        teacher_cohesion = []
        for _, indices in sorted(branch_to_indices.items()):
            if not indices:
                continue
            index_tensor = torch.tensor(indices, dtype=torch.long)
            members = teacher_directions.index_select(0, index_tensor)
            centroid = F.normalize(members.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
            member_tensors.append(index_tensor)
            teacher_centroids.append(centroid)
            teacher_cohesion.append(1.0 - torch.sum(members * centroid, dim=-1).mean())

        self.branch_member_tensors = member_tensors
        if teacher_centroids:
            self.register_buffer("teacher_centroids", torch.stack(teacher_centroids, dim=0))
            self.register_buffer("teacher_cohesion", torch.stack(teacher_cohesion, dim=0))
        else:
            self.register_buffer("teacher_centroids", torch.empty((0, teacher_embeddings.size(1)), dtype=torch.double))
            self.register_buffer("teacher_cohesion", torch.empty((0,), dtype=torch.double))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.branch_member_tensors:
            return embeddings.new_tensor(0.0)

        projected = project_to_ball_torch(embeddings)
        norms = torch.linalg.vector_norm(projected, dim=-1, keepdim=True).clamp_min(1e-8)
        directions = projected / norms

        centroid_losses = []
        cohesion_losses = []
        teacher_centroids = self.teacher_centroids.to(device=directions.device, dtype=directions.dtype)
        teacher_cohesion = self.teacher_cohesion.to(device=directions.device, dtype=directions.dtype)
        for branch_index, member_indices in enumerate(self.branch_member_tensors):
            members = directions.index_select(0, member_indices.to(directions.device))
            centroid = F.normalize(members.mean(dim=0, keepdim=True), p=2, dim=-1).squeeze(0)
            centroid_losses.append(1.0 - torch.sum(centroid * teacher_centroids[branch_index]))
            cohesion = 1.0 - torch.sum(members * centroid, dim=-1).mean()
            cohesion_losses.append(torch.relu(cohesion - teacher_cohesion[branch_index]))

        return torch.stack(centroid_losses).mean() + torch.stack(cohesion_losses).mean()


class BranchContrastiveMarginLoss(nn.Module):
    def __init__(
        self,
        metadata_rows,
        objects,
        margin: float = 0.02,
        hard_negative_k: int = 0,
    ):
        super().__init__()
        node_to_index = {node_id: index for index, node_id in enumerate(objects)}
        depth_branch_to_indices = defaultdict(list)
        for row in metadata_rows:
            node_id = row["node_id"]
            if node_id not in node_to_index or int(row["depth"]) == 0:
                continue
            key = (int(row["depth"]), row["top_branch_id"])
            depth_branch_to_indices[key].append(node_to_index[node_id])

        depth_to_keys = defaultdict(list)
        for key in depth_branch_to_indices:
            depth_to_keys[key[0]].append(key)

        self.margin = margin
        self.hard_negative_k = max(int(hard_negative_k), 0)
        self.group_tensors = []
        self.negative_tensors = []
        for key, indices in sorted(depth_branch_to_indices.items()):
            negative_indices = []
            for other_key in depth_to_keys[key[0]]:
                if other_key == key:
                    continue
                negative_indices.extend(depth_branch_to_indices[other_key])
            if indices and negative_indices:
                self.group_tensors.append(torch.tensor(indices, dtype=torch.long))
                self.negative_tensors.append(torch.tensor(sorted(set(negative_indices)), dtype=torch.long))

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if not self.group_tensors:
            return embeddings.new_tensor(0.0)

        projected = project_to_ball_torch(embeddings)
        group_losses = []
        for group_indices, negative_indices in zip(self.group_tensors, self.negative_tensors):
            members = projected.index_select(0, group_indices.to(projected.device))
            negatives = projected.index_select(0, negative_indices.to(projected.device))

            if members.size(0) > 1:
                centroid = project_to_ball_torch(members.mean(dim=0, keepdim=True)).expand_as(members)
                positive = poincare_distance_torch(members, centroid).mean()
            else:
                positive = embeddings.new_tensor(0.0)

            expanded_members = members[:, None, :].expand(members.size(0), negatives.size(0), members.size(1))
            expanded_negatives = negatives[None, :, :].expand_as(expanded_members)
            negative_distances = poincare_distance_torch(
                expanded_members.reshape(-1, members.size(1)),
                expanded_negatives.reshape(-1, members.size(1)),
            ).reshape(members.size(0), negatives.size(0))
            if self.hard_negative_k > 0 and self.hard_negative_k < negative_distances.size(1):
                negative_distances = torch.topk(
                    negative_distances,
                    k=self.hard_negative_k,
                    dim=1,
                    largest=False,
                ).values
            negative = torch.relu(self.margin - negative_distances).mean()
            group_losses.append(positive + negative)

        return torch.stack(group_losses).mean() if group_losses else embeddings.new_tensor(0.0)
