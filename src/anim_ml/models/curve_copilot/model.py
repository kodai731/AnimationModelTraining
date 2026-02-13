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
    num_joint_categories: int = 7
    keyframe_dim: int = 6


class CausalTransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        normed = self.norm1(x)
        attn_out, _ = self.attn(normed, normed, normed, attn_mask=mask, need_weights=False)
        x = x + attn_out

        normed = self.norm2(x)
        x = x + self.ffn(normed)

        return x


class CurveCopilotModel(nn.Module):
    def __init__(self, config: CurveCopilotConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = CurveCopilotConfig()
        self.config = config

        self.keyframe_projection = nn.Linear(config.keyframe_dim, config.d_model)
        self.property_type_embedding = nn.Embedding(config.num_property_types, config.d_model)
        self.joint_category_embedding = nn.Embedding(config.num_joint_categories, config.d_model)
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

    def forward(
        self,
        context_keyframes: torch.Tensor,
        property_type: torch.Tensor,
        joint_category: torch.Tensor,
        query_time: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        seq_len = context_keyframes.shape[1]
        total_len = seq_len + 1

        kf_embed = self.keyframe_projection(context_keyframes)

        prop_embed = self.property_type_embedding(property_type)
        joint_embed = self.joint_category_embedding(joint_category)
        condition = (prop_embed + joint_embed).unsqueeze(1)

        positions = torch.arange(seq_len, device=context_keyframes.device)
        kf_embed = kf_embed + condition + self.positional_embedding(positions).unsqueeze(0)

        query_pos = torch.tensor([seq_len], device=context_keyframes.device)
        query_token = (
            self.query_time_projection(query_time.unsqueeze(-1))
            + condition.squeeze(1)
            + self.positional_embedding(query_pos)
        ).unsqueeze(1)

        x = torch.cat([kf_embed, query_token], dim=1)

        causal_mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        mask = causal_mask[:total_len, :total_len]
        for block in self.blocks:
            x = block(x, mask)

        last_token = self.output_norm(x)[:, -1, :]
        prediction = self.prediction_head(last_token)
        confidence = torch.sigmoid(self.confidence_head(last_token))

        return prediction, confidence


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dict(config_dict: dict[str, object]) -> CurveCopilotModel:
    config = CurveCopilotConfig(**config_dict)  # type: ignore[arg-type]
    return CurveCopilotModel(config)
