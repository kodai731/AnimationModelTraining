from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from anim_ml.models.bone_encoder import BoneEncoderConfig, BoneIdentityEncoder


@dataclass
class RigPropagationConfig:
    max_joints: int = 64
    max_edges: int = 126
    node_feature_dim: int = 128
    edge_feature_dim: int = 32
    hidden_dim: int = 256
    ffn_dim: int = 2048
    num_message_passing_layers: int = 4
    input_feature_dim: int = 9
    dropout: float = 0.1
    vocab_size: int = 64
    token_length: int = 32
    char_embed_dim: int = 32
    conv_channels: int = 64
    bone_context_dim: int = 64
    topology_dim: int = 6


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
        edge_mask: torch.Tensor,
        max_joints: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        normed = self.norm1(node_features)

        src_features = _gather_nodes(normed, source_indices)
        tgt_features = _gather_nodes(normed, target_indices)

        edge_messages = self.edge_model(src_features, tgt_features, edge_attr)
        edge_messages = edge_messages * edge_mask.unsqueeze(-1)

        aggregated = _scatter_mean(edge_messages, target_indices, max_joints)
        projected = self.message_projection(aggregated)

        node_features = node_features + self.gated_update(normed, projected)

        normed = self.norm2(node_features)
        node_features = node_features + self.ffn(normed)

        return node_features, edge_messages


class RigPropagationModel(nn.Module):
    def __init__(self, config: RigPropagationConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = RigPropagationConfig()
        self.config = config

        self.input_projection = nn.Linear(
            config.input_feature_dim, config.node_feature_dim,
        )

        bone_cfg = BoneEncoderConfig(
            vocab_size=config.vocab_size,
            token_length=config.token_length,
            char_embed_dim=config.char_embed_dim,
            conv_channels=config.conv_channels,
            bone_context_dim=config.bone_context_dim,
            topology_dim=config.topology_dim,
        )
        self.bone_identity_encoder = BoneIdentityEncoder(bone_cfg)
        self.bone_identity_projection = nn.Linear(
            config.bone_context_dim, config.node_feature_dim,
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
        topology_features: torch.Tensor,
        bone_name_tokens: torch.Tensor,
        joint_mask: torch.Tensor,
        source_indices: torch.Tensor,
        target_indices: torch.Tensor,
        edge_direction: torch.Tensor,
        edge_mask: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch_size = joint_features.shape[0]
        max_joints = joint_features.shape[1]

        flat_topo = topology_features.reshape(-1, topology_features.shape[-1])
        flat_tokens = bone_name_tokens.reshape(-1, bone_name_tokens.shape[-1])
        flat_bone_context = self.bone_identity_encoder(flat_topo, flat_tokens)
        bone_context = flat_bone_context.reshape(batch_size, max_joints, -1)

        node_embed = self.input_projection(joint_features)
        node_embed = node_embed + self.bone_identity_projection(bone_context)

        edge_attr = self.edge_direction_embedding(edge_direction)
        edge_attr = edge_attr.unsqueeze(0).expand(batch_size, -1, -1)
        batch_edge_mask = edge_mask.unsqueeze(0).expand(batch_size, -1)

        for block in self.blocks:
            node_embed, edge_attr = block(
                node_embed, edge_attr, source_indices, target_indices,
                batch_edge_mask, max_joints,
            )

        node_out = self.output_norm(node_embed)

        raw_deltas = self.delta_head(node_out)
        rotation_deltas = _normalize_quaternions(raw_deltas)
        confidence = torch.sigmoid(self.confidence_head(node_out))

        mask_3d = joint_mask.unsqueeze(-1)
        rotation_deltas = rotation_deltas * mask_3d
        confidence = confidence * mask_3d

        return rotation_deltas, confidence


def _gather_nodes(
    node_features: torch.Tensor,
    indices: torch.Tensor,
) -> torch.Tensor:
    expanded = indices.unsqueeze(0).unsqueeze(-1)
    expanded = expanded.expand(node_features.shape[0], -1, node_features.shape[2])
    return torch.gather(node_features, 1, expanded)


def _scatter_mean(
    edge_messages: torch.Tensor,
    target_indices: torch.Tensor,
    max_nodes: int,
) -> torch.Tensor:
    batch_size = edge_messages.shape[0]
    feat_dim = edge_messages.shape[2]

    result = torch.zeros(
        batch_size, max_nodes, feat_dim,
        device=edge_messages.device, dtype=edge_messages.dtype,
    )
    counts = torch.zeros(
        batch_size, max_nodes, 1,
        device=edge_messages.device, dtype=edge_messages.dtype,
    )

    idx = target_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, feat_dim)
    cnt_idx = target_indices.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, 1)

    result.scatter_add_(1, idx, edge_messages)
    counts.scatter_add_(1, cnt_idx, torch.ones_like(edge_messages[:, :, :1]))

    return result / counts.clamp(min=1.0)


def _normalize_quaternions(quats: torch.Tensor) -> torch.Tensor:
    norms = (quats * quats).sum(dim=-1, keepdim=True).sqrt()
    norms = torch.clamp(norms, min=1e-8)
    return quats / norms


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dict(config_dict: dict[str, object]) -> RigPropagationModel:
    config = RigPropagationConfig(**config_dict)  # type: ignore[arg-type]
    return RigPropagationModel(config)
