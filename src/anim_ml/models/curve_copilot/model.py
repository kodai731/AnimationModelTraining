from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class CurveCopilotConfig:
    d_model: int = 128
    n_heads: int = 4
    d_ff: int = 512
    n_layers: int = 6
    max_seq: int = 8
    dropout: float = 0.1
    num_property_types: int = 9
    keyframe_dim: int = 6
    vocab_size: int = 64
    token_length: int = 32
    char_embed_dim: int = 32
    conv_channels: int = 64
    bone_context_dim: int = 64
    topology_dim: int = 6


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int, dropout: float) -> None:
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv_proj = nn.Linear(d_model, d_model * 3)
        self.out_proj = nn.Linear(d_model, d_model)
        self.attn_dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        batch, seq_len, d_model = x.shape

        qkv = self.qkv_proj(x).reshape(batch, seq_len, 3, self.n_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv.unbind(0)

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if mask is not None:
            scores = scores.masked_fill(mask.unsqueeze(0).unsqueeze(0), float("-inf"))

        weights = torch.softmax(scores, dim=-1)
        weights = self.attn_dropout(weights)

        out = torch.matmul(weights, v)
        out = out.transpose(1, 2).reshape(batch, seq_len, d_model)

        return self.out_proj(out)


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = SelfAttention(d_model, n_heads, dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        x = x + self.attn(normed, mask)

        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x


class BoneIdentityEncoder(nn.Module):
    def __init__(self, config: CurveCopilotConfig) -> None:
        super().__init__()
        self.char_embedding = nn.Embedding(config.vocab_size, config.char_embed_dim)
        self.conv = nn.Conv1d(config.char_embed_dim, config.conv_channels, kernel_size=3, padding=1)
        self.name_proj = nn.Linear(config.conv_channels, config.bone_context_dim // 2)

        self.topo_proj = nn.Linear(config.topology_dim, config.bone_context_dim // 2)

    def forward(
        self,
        topology_features: torch.Tensor,
        bone_name_tokens: torch.Tensor,
    ) -> torch.Tensor:
        char_embed = self.char_embedding(bone_name_tokens)
        conv_out = self.conv(char_embed.transpose(1, 2))
        conv_out = torch.relu(conv_out)
        pooled = conv_out.mean(dim=2)
        name_embed = self.name_proj(pooled)

        topo_embed = torch.relu(self.topo_proj(topology_features))

        return torch.cat([name_embed, topo_embed], dim=-1)


class CurveCopilotModel(nn.Module):
    def __init__(self, config: CurveCopilotConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = CurveCopilotConfig()
        self.config = config

        self.keyframe_projection = nn.Linear(config.keyframe_dim, config.d_model)
        self.property_type_embedding = nn.Embedding(config.num_property_types, config.d_model)
        self.bone_identity_encoder = BoneIdentityEncoder(config)
        self.bone_context_projection = nn.Linear(config.bone_context_dim, config.d_model)
        self.positional_embedding = nn.Embedding(config.max_seq + 1, config.d_model)
        self.query_time_projection = nn.Linear(1, config.d_model)

        self.blocks = nn.ModuleList([
            CausalTransformerBlock(config.d_model, config.n_heads, config.d_ff, config.dropout)
            for _ in range(config.n_layers)
        ])

        self.output_norm = nn.LayerNorm(config.d_model)
        self.prediction_head = nn.Linear(config.d_model, 6)
        self.confidence_head = nn.Linear(config.d_model, 1)

        full_mask = torch.triu(
            torch.ones(config.max_seq + 1, config.max_seq + 1), diagonal=1,
        ).bool()
        self.register_buffer("causal_mask", full_mask)
        self.register_buffer("kf_positions", torch.arange(config.max_seq))
        self.register_buffer("query_position", torch.tensor([config.max_seq]))

    def forward(
        self,
        context_keyframes: torch.Tensor,
        property_type: torch.Tensor,
        topology_features: torch.Tensor,
        bone_name_tokens: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = context_keyframes.shape[1]
        total_len = seq_len + 1

        kf_embed = self.keyframe_projection(context_keyframes)

        prop_embed = self.property_type_embedding(property_type)
        bone_context = self.bone_identity_encoder(topology_features, bone_name_tokens)
        bone_embed = self.bone_context_projection(bone_context)
        condition = (prop_embed + bone_embed).unsqueeze(1)

        kf_positions: torch.Tensor = self.kf_positions[:seq_len]  # type: ignore[assignment]
        kf_embed = kf_embed + condition + self.positional_embedding(kf_positions).unsqueeze(0)

        query_position: torch.Tensor = self.query_position  # type: ignore[assignment]
        query_token = (
            self.query_time_projection(query_time.unsqueeze(-1))
            + condition.squeeze(1)
            + self.positional_embedding(query_position)
        ).unsqueeze(1)

        x = torch.cat([kf_embed, query_token], dim=1)

        causal_mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        mask = causal_mask[:total_len, :total_len]
        for block in self.blocks:
            x = block(x, mask)

        last_token = self.output_norm(x)[:, -1, :]
        prediction = self.prediction_head(last_token)
        confidence = torch.sigmoid(self.confidence_head(last_token.detach()))

        return prediction, confidence


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dict(config_dict: dict[str, object]) -> CurveCopilotModel:
    config = CurveCopilotConfig(**config_dict)  # type: ignore[arg-type]
    return CurveCopilotModel(config)
