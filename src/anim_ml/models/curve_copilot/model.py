from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from anim_ml.models.bone_encoder import BoneEncoderConfig, BoneIdentityEncoder
from anim_ml.models.periodic_autoencoder.model import PAEConfig, PAEEncoder


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
    periodic_time_encoding: bool = False
    num_experts: int = 3
    use_expert_mixing: bool = False
    use_pae: bool = False
    pae_window_size: int = 64
    pae_latent_channels: int = 5


class PeriodicTimeEncoder(nn.Module):
    def forward(self, t: torch.Tensor) -> torch.Tensor:
        two_pi_t = 2.0 * torch.pi * t
        return torch.stack([t, torch.sin(two_pi_t), torch.cos(two_pi_t)], dim=-1)


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


class GatingNetwork(nn.Module):
    def __init__(
        self, num_property_types: int, d_model: int, num_experts: int,
    ) -> None:
        super().__init__()
        self.property_embedding = nn.Embedding(num_property_types, d_model)
        self.gate_proj = nn.Linear(d_model, num_experts)

    def forward(self, property_type: torch.Tensor) -> torch.Tensor:
        emb = self.property_embedding(property_type)
        return torch.softmax(self.gate_proj(emb), dim=-1)


class PropertyAdaptiveBlock(nn.Module):
    def __init__(
        self, d_model: int, n_heads: int, d_ff: int, dropout: float, num_experts: int,
    ) -> None:
        super().__init__()
        self.experts = nn.ModuleList([
            CausalTransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(num_experts)
        ])

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor, gate_weights: torch.Tensor,
    ) -> torch.Tensor:
        expert_outputs = torch.stack(
            [expert(x, mask) for expert in self.experts], dim=1,
        )
        weights = gate_weights.unsqueeze(-1).unsqueeze(-1)
        return (expert_outputs * weights).sum(dim=1)


class CurveCopilotModel(nn.Module):
    def __init__(self, config: CurveCopilotConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = CurveCopilotConfig()
        self.config = config

        self.periodic_time_encoding = config.periodic_time_encoding
        if config.periodic_time_encoding:
            self.time_encoder = PeriodicTimeEncoder()

        kf_proj_dim = (
            config.keyframe_dim + 2 if config.periodic_time_encoding else config.keyframe_dim
        )
        self.keyframe_projection = nn.Linear(kf_proj_dim, config.d_model)
        self.property_type_embedding = nn.Embedding(config.num_property_types, config.d_model)
        bone_cfg = BoneEncoderConfig(
            vocab_size=config.vocab_size,
            token_length=config.token_length,
            char_embed_dim=config.char_embed_dim,
            conv_channels=config.conv_channels,
            bone_context_dim=config.bone_context_dim,
            topology_dim=config.topology_dim,
        )
        self.bone_identity_encoder = BoneIdentityEncoder(bone_cfg)
        self.bone_context_projection = nn.Linear(config.bone_context_dim, config.d_model)
        self.positional_embedding = nn.Embedding(config.max_seq + 1, config.d_model)
        query_time_input_dim = 3 if config.periodic_time_encoding else 1
        self.query_time_projection = nn.Linear(query_time_input_dim, config.d_model)

        self.gating: GatingNetwork | None = None

        if config.use_expert_mixing:
            self.gating = GatingNetwork(
                config.num_property_types, config.d_model, config.num_experts,
            )
            self.blocks = nn.ModuleList([
                PropertyAdaptiveBlock(
                    config.d_model, config.n_heads, config.d_ff,
                    config.dropout, config.num_experts,
                )
                for _ in range(config.n_layers)
            ])
        else:
            self.blocks = nn.ModuleList([
                CausalTransformerBlock(
                    config.d_model, config.n_heads, config.d_ff, config.dropout,
                )
                for _ in range(config.n_layers)
            ])

        self.pae_encoder: PAEEncoder | None = None
        self.phase_projection: nn.Linear | None = None
        if config.use_pae:
            pae_cfg = PAEConfig(
                window_size=config.pae_window_size,
                latent_channels=config.pae_latent_channels,
                feature_dim=config.pae_latent_channels * 3,
            )
            self.pae_encoder = PAEEncoder(pae_cfg)
            self.phase_projection = nn.Linear(pae_cfg.feature_dim, config.d_model)

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
        curve_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = context_keyframes.shape[1]
        total_len = seq_len + 1

        if self.periodic_time_encoding:
            kf_time = context_keyframes[:, :, 0]
            kf_time_encoded = self.time_encoder(kf_time)
            kf_rest = context_keyframes[:, :, 1:]
            kf_augmented = torch.cat([kf_time_encoded, kf_rest], dim=-1)
            kf_embed = self.keyframe_projection(kf_augmented)
        else:
            kf_embed = self.keyframe_projection(context_keyframes)

        prop_embed = self.property_type_embedding(property_type)
        bone_context = self.bone_identity_encoder(topology_features, bone_name_tokens)
        bone_embed = self.bone_context_projection(bone_context)
        condition = (prop_embed + bone_embed).unsqueeze(1)

        if self.pae_encoder is not None and curve_window is not None:
            phase_features = self.pae_encoder(curve_window)
            phase_embed = self.phase_projection(phase_features).unsqueeze(1)  # type: ignore[union-attr]
            condition = condition + phase_embed

        kf_positions: torch.Tensor = self.kf_positions[:seq_len]  # type: ignore[assignment]
        kf_embed = kf_embed + condition + self.positional_embedding(kf_positions).unsqueeze(0)

        query_position: torch.Tensor = self.query_position  # type: ignore[assignment]

        if self.periodic_time_encoding:
            query_time_input = self.time_encoder(query_time)
        else:
            query_time_input = query_time.unsqueeze(-1)

        query_token = (
            self.query_time_projection(query_time_input)
            + condition.squeeze(1)
            + self.positional_embedding(query_position)
        ).unsqueeze(1)

        x = torch.cat([kf_embed, query_token], dim=1)

        causal_mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        mask = causal_mask[:total_len, :total_len]

        if self.gating is not None:
            gate_weights = self.gating(property_type)
            for block in self.blocks:
                x = block(x, mask, gate_weights)
        else:
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
