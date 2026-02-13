from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from anim_ml.data.rig_data_generator import build_adjacency
from anim_ml.utils.skeleton import SMPL_22_PARENT_INDICES


@dataclass
class RigPropagationConfig:
    num_joints: int = 22
    node_feature_dim: int = 128
    edge_feature_dim: int = 32
    hidden_dim: int = 256
    ffn_dim: int = 2048
    num_message_passing_layers: int = 4
    num_joint_types: int = 13
    joint_type_embed_dim: int = 32
    input_feature_dim: int = 10
    dropout: float = 0.1


class EdgeModel(nn.Module):
    def __init__(self, node_dim: int, edge_dim: int) -> None:
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(node_dim * 2 + edge_dim, edge_dim * 4),
            nn.ReLU(),
            nn.Linear(edge_dim * 4, edge_dim),
        )

    def forward(
        self,
        src_features: torch.Tensor,
        tgt_features: torch.Tensor,
        edge_attr: torch.Tensor,
    ) -> torch.Tensor:
        combined = torch.cat([src_features, tgt_features, edge_attr], dim=-1)
        return self.mlp(combined)


class GatedNodeUpdate(nn.Module):
    def __init__(self, node_dim: int) -> None:
        super().__init__()
        self.reset_gate = nn.Linear(node_dim * 2, node_dim)
        self.update_gate = nn.Linear(node_dim * 2, node_dim)
        self.candidate = nn.Linear(node_dim * 2, node_dim)

    def forward(self, node: torch.Tensor, message: torch.Tensor) -> torch.Tensor:
        combined = torch.cat([node, message], dim=-1)
        reset = torch.sigmoid(self.reset_gate(combined))
        update = torch.sigmoid(self.update_gate(combined))

        reset_combined = torch.cat([reset * node, message], dim=-1)
        candidate = torch.tanh(self.candidate(reset_combined))

        return (1 - update) * node + update * candidate


class MessagePassingLayer(nn.Module):
    def __init__(self, config: RigPropagationConfig) -> None:
        super().__init__()
        node_dim = config.node_feature_dim
        edge_dim = config.edge_feature_dim

        self.edge_model = EdgeModel(node_dim, edge_dim)
        self.message_projection = nn.Linear(edge_dim, node_dim)
        self.gated_update = GatedNodeUpdate(node_dim)

        self.norm1 = nn.LayerNorm(node_dim)
        self.norm2 = nn.LayerNorm(node_dim)

        self.ffn = nn.Sequential(
            nn.Linear(node_dim, config.ffn_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.ffn_dim, node_dim),
            nn.Dropout(config.dropout),
        )

    def forward(
        self,
        node_features: torch.Tensor,
        edge_attr: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        aggregation_matrix: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(node_features)

        src_features = _gather_nodes(normed, source_indices)
        tgt_features = _gather_nodes(normed, target_indices)

        edge_messages = self.edge_model(src_features, tgt_features, edge_attr)

        aggregated = torch.matmul(aggregation_matrix, edge_messages)
        projected = self.message_projection(aggregated)

        node_features = node_features + self.gated_update(normed, projected)

        normed = self.norm2(node_features)
        node_features = node_features + self.ffn(normed)

        return node_features, edge_messages


class RigPropagationModel(nn.Module):
    def __init__(
        self,
        config: RigPropagationConfig | None = None,
        adjacency: torch.Tensor | None = None,
    ) -> None:
        super().__init__()
        if config is None:
            config = RigPropagationConfig()
        self.config = config

        if adjacency is None:
            adj_np = build_adjacency(SMPL_22_PARENT_INDICES)
            adjacency = torch.from_numpy(adj_np)  # type: ignore[no-any-return]

        source_idx = adjacency[0].long()
        target_idx = adjacency[1].long()
        num_edges = source_idx.shape[0]

        self.register_buffer("source_indices", source_idx)
        self.register_buffer("target_indices", target_idx)

        agg_matrix = torch.zeros(config.num_joints, num_edges)
        for edge_i in range(num_edges):
            tgt = int(target_idx[edge_i].item())
            agg_matrix[tgt, edge_i] = 1.0
        self.register_buffer("aggregation_matrix", agg_matrix)

        edge_direction = _compute_edge_direction(
            source_idx, target_idx, SMPL_22_PARENT_INDICES,
        )
        self.register_buffer("edge_direction_raw", edge_direction)

        self.input_projection = nn.Linear(
            config.input_feature_dim, config.node_feature_dim,
        )
        self.joint_type_embedding = nn.Embedding(
            config.num_joint_types, config.joint_type_embed_dim,
        )
        self.joint_type_projection = nn.Linear(
            config.joint_type_embed_dim, config.node_feature_dim,
        )
        self.edge_direction_embedding = nn.Embedding(2, config.edge_feature_dim)

        self.blocks = nn.ModuleList([
            MessagePassingLayer(config)
            for _ in range(config.num_message_passing_layers)
        ])

        self.output_norm = nn.LayerNorm(config.node_feature_dim)
        self.delta_head = nn.Linear(config.node_feature_dim, 4)
        self.confidence_head = nn.Linear(config.node_feature_dim, 1)

    def forward(
        self,
        joint_features: torch.Tensor,
        joint_types: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        source_indices: torch.Tensor = self.source_indices  # type: ignore[assignment]
        target_indices: torch.Tensor = self.target_indices  # type: ignore[assignment]
        aggregation_matrix: torch.Tensor = self.aggregation_matrix  # type: ignore[assignment]
        edge_direction_raw: torch.Tensor = self.edge_direction_raw  # type: ignore[assignment]

        node_embed = self.input_projection(joint_features)
        type_embed = self.joint_type_projection(
            self.joint_type_embedding(joint_types),
        )
        node_embed = node_embed + type_embed

        edge_attr = self.edge_direction_embedding(edge_direction_raw)
        batch_size = joint_features.shape[0]
        edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)

        agg = aggregation_matrix.unsqueeze(0).expand(batch_size, -1, -1)

        for block in self.blocks:
            node_embed, edge_attr = block(
                node_embed, edge_attr, source_indices, target_indices, agg,
            )

        node_out = self.output_norm(node_embed)

        raw_deltas = self.delta_head(node_out)
        rotation_deltas = _normalize_quaternions(raw_deltas)
        confidence = torch.sigmoid(self.confidence_head(node_out))

        return rotation_deltas, confidence


def _gather_nodes(
    node_features: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    expanded = indices.unsqueeze(0).unsqueeze(-1)
    expanded = expanded.expand(node_features.shape[0], -1, node_features.shape[2])
    return torch.gather(node_features, 1, expanded)


def _compute_edge_direction(
    source_idx: torch.Tensor,
    target_idx: torch.Tensor,
    parent_indices: list[int],
) -> torch.Tensor:
    num_edges = source_idx.shape[0]
    direction = torch.zeros(num_edges, dtype=torch.long)

    for i in range(num_edges):
        src = int(source_idx[i].item())
        tgt = int(target_idx[i].item())
        if parent_indices[tgt] == src:
            direction[i] = 0
        else:
            direction[i] = 1

    return direction


def _normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    norms = (quats * quats).sum(dim=-1, keepdim=True).sqrt()
    norms = torch.clamp(norms, min=1e-8)
    return quats / norms


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dict(
    config_dict: dict[str, object],
    adjacency: torch.Tensor | None = None,
) -> RigPropagationModel:
    config = RigPropagationConfig(**config_dict)  # type: ignore[arg-type]
    return RigPropagationModel(config, adjacency=adjacency)
