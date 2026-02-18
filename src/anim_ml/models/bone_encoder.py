from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class BoneEncoderConfig:
    vocab_size: int = 64
    token_length: int = 32
    char_embed_dim: int = 32
    conv_channels: int = 64
    bone_context_dim: int = 64
    topology_dim: int = 6


class BoneIdentityEncoder(nn.Module):
    def __init__(self, config: BoneEncoderConfig) -> None:
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
