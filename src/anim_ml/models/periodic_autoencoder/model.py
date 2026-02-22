from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class PAEConfig:
    window_size: int = 64
    latent_channels: int = 5
    feature_dim: int = 15


class PAEEncoder(nn.Module):
    def __init__(self, config: PAEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PAEConfig()
        self.config = config

        self.convs = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )

        conv_out_size = 64 * (config.window_size // 8)
        self.fc = nn.Linear(conv_out_size, config.feature_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.unsqueeze(1)
        h = self.convs(h)
        h = h.flatten(1)
        return self.fc(h)


class PAEDecoder(nn.Module):
    def __init__(self, config: PAEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PAEConfig()
        self.config = config

        spatial_dim = config.window_size // 8
        self.fc = nn.Linear(config.feature_dim, 64 * spatial_dim)
        self.spatial_dim = spatial_dim

        self.deconvs = nn.Sequential(
            nn.ConvTranspose1d(64, 64, kernel_size=3, stride=2, padding=1, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(64, 32, kernel_size=5, stride=2, padding=2, output_padding=1),
            nn.ReLU(),
            nn.ConvTranspose1d(32, 1, kernel_size=7, stride=2, padding=3, output_padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.fc(z)
        h = h.view(h.shape[0], 64, self.spatial_dim)
        h = self.deconvs(h)
        return h.squeeze(1)


class PeriodicAutoencoder(nn.Module):
    def __init__(self, config: PAEConfig | None = None) -> None:
        super().__init__()
        if config is None:
            config = PAEConfig()
        self.config = config
        self.encoder = PAEEncoder(config)
        self.decoder = PAEDecoder(config)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.encoder(x)
        reconstructed = self.decoder(features)
        return features, reconstructed
