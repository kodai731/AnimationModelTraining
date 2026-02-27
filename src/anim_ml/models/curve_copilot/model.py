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
    enhanced_gating: bool = False
    use_multi_resolution: bool = False
    multi_res_branch_dim: int = 32
    use_phase_detection: bool = False
    max_steps: int = 1


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
        self,
        num_property_types: int,
        d_model: int,
        num_experts: int,
        enhanced_gating: bool = False,
    ) -> None:
        super().__init__()
        self.property_embedding = nn.Embedding(num_property_types, d_model)

        self.motion_proj: nn.Linear | None = None
        if enhanced_gating:
            self.motion_proj = nn.Linear(3, d_model)
            self.gate_proj: nn.Module = nn.Sequential(
                nn.Linear(d_model, d_model // 2),
                nn.ReLU(),
                nn.Linear(d_model // 2, num_experts),
            )
        else:
            self.gate_proj = nn.Linear(d_model, num_experts)

    def forward(
        self, property_type: torch.Tensor, motion_state: torch.Tensor | None = None,
    ) -> torch.Tensor:
        emb = self.property_embedding(property_type)
        if self.motion_proj is not None and motion_state is not None:
            emb = emb + self.motion_proj(motion_state)
        return torch.softmax(self.gate_proj(emb), dim=-1)


class MultiResolutionEncoder(nn.Module):
    def __init__(self, window_size: int, branch_dim: int, d_model: int) -> None:
        super().__init__()
        self.low_pool = nn.AvgPool1d(kernel_size=16, stride=16)
        self.low_conv = nn.Conv1d(1, branch_dim, kernel_size=3, padding=1)
        self.low_proj = nn.Linear(branch_dim, branch_dim)

        self.mid_pool = nn.AvgPool1d(kernel_size=4, stride=4)
        self.mid_conv = nn.Conv1d(1, branch_dim, kernel_size=3, padding=1)
        self.mid_proj = nn.Linear(branch_dim, branch_dim)

        self.high_conv = nn.Conv1d(1, branch_dim, kernel_size=3, padding=1)
        self.high_proj = nn.Linear(branch_dim, branch_dim)

        self.fusion = nn.Linear(branch_dim * 3, d_model)

    def _encode_branch(
        self, x: torch.Tensor, conv: nn.Conv1d, proj: nn.Linear,
    ) -> torch.Tensor:
        h = torch.relu(conv(x.unsqueeze(1)))
        return proj(h.mean(dim=-1))

    def forward(self, curve_window: torch.Tensor) -> torch.Tensor:
        x_3d = curve_window.unsqueeze(1)

        low = self._encode_branch(
            self.low_pool(x_3d).squeeze(1), self.low_conv, self.low_proj,
        )
        mid = self._encode_branch(
            self.mid_pool(x_3d).squeeze(1), self.mid_conv, self.mid_proj,
        )
        high = self._encode_branch(
            curve_window[:, -16:], self.high_conv, self.high_proj,
        )

        return self.fusion(torch.cat([low, mid, high], dim=-1))


class PhaseDetector(nn.Module):
    def __init__(self, d_model: int) -> None:
        super().__init__()
        self.conv_layers = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=7, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.ReLU(),
            nn.Conv1d(64, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
        )
        self.phase_head = nn.Linear(512, 2)
        self.period_head = nn.Linear(512, 1)
        self.amplitude_head = nn.Linear(512, 1)
        self.context_proj = nn.Linear(4, d_model)
        self.query_proj = nn.Linear(4, d_model)

    def extract_phase_params(
        self, curve_window: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        h = self.conv_layers(curve_window.unsqueeze(1)).flatten(1)

        phase_xy = self.phase_head(h)
        base_phase = torch.atan2(phase_xy[:, 0], phase_xy[:, 1])

        period = torch.nn.functional.softplus(self.period_head(h).squeeze(-1)) + 0.01
        amplitude = self.amplitude_head(h).squeeze(-1)

        return base_phase, period, amplitude

    def compute_context_conditioning(
        self,
        base_phase: torch.Tensor,
        period: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        features = torch.stack([
            torch.sin(base_phase), torch.cos(base_phase), amplitude, period,
        ], dim=-1)
        return self.context_proj(features)

    def compute_query_phase_encoding(
        self,
        query_time: torch.Tensor,
        base_phase: torch.Tensor,
        period: torch.Tensor,
        amplitude: torch.Tensor,
    ) -> torch.Tensor:
        relative_phase = 2.0 * torch.pi * (query_time / period) + base_phase
        features = torch.stack([
            torch.sin(relative_phase), torch.cos(relative_phase), amplitude, period,
        ], dim=-1)
        return self.query_proj(features)


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

        motion_feature_dim = 2
        kf_proj_dim = config.keyframe_dim + motion_feature_dim
        if config.periodic_time_encoding:
            kf_proj_dim += 2
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
        self.positional_embedding = nn.Embedding(config.max_seq + config.max_steps, config.d_model)
        query_time_input_dim = 3 if config.periodic_time_encoding else 1
        self.query_time_projection = nn.Linear(query_time_input_dim, config.d_model)

        self.gating: GatingNetwork | None = None

        if config.use_expert_mixing:
            self.gating = GatingNetwork(
                config.num_property_types, config.d_model, config.num_experts,
                config.enhanced_gating,
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

        self.multi_res_encoder: MultiResolutionEncoder | None = None
        if config.use_multi_resolution:
            self.multi_res_encoder = MultiResolutionEncoder(
                config.pae_window_size, config.multi_res_branch_dim, config.d_model,
            )

        self.phase_detector: PhaseDetector | None = None
        if config.use_phase_detection:
            self.phase_detector = PhaseDetector(config.d_model)

        self.output_norm = nn.LayerNorm(config.d_model)
        self.prediction_head = nn.Linear(config.d_model, 5)
        self.confidence_head = nn.Linear(config.d_model, 1)

        total_positions = config.max_seq + config.max_steps
        full_mask = torch.triu(
            torch.ones(total_positions, total_positions), diagonal=1,
        ).bool()
        self.register_buffer("causal_mask", full_mask)
        self.register_buffer("kf_positions", torch.arange(config.max_seq))
        self.register_buffer(
            "query_positions",
            torch.arange(config.max_seq, config.max_seq + config.max_steps),
        )

    def _compute_motion_features(self, context_keyframes: torch.Tensor) -> torch.Tensor:
        values = context_keyframes[:, :, 1]

        velocity = torch.zeros_like(values)
        velocity[:, 1:] = values[:, 1:] - values[:, :-1]

        acceleration = torch.zeros_like(values)
        acceleration[:, 2:] = velocity[:, 2:] - velocity[:, 1:-1]

        return torch.stack([velocity, acceleration], dim=-1)

    def _compute_motion_state(self, motion_features: torch.Tensor) -> torch.Tensor:
        velocity = motion_features[:, :, 0]
        acceleration = motion_features[:, :, 1]

        velocity_magnitude = velocity.abs().mean(dim=1)
        acceleration_magnitude = acceleration.abs().mean(dim=1)
        velocity_trend = velocity[:, -1]

        return torch.stack(
            [velocity_magnitude, acceleration_magnitude, velocity_trend], dim=-1,
        )

    def forward(
        self,
        context_keyframes: torch.Tensor,
        property_type: torch.Tensor,
        topology_features: torch.Tensor,
        bone_name_tokens: torch.Tensor,
        query_times: torch.Tensor,
        curve_window: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if query_times.ndim == 1:
            query_times = query_times.unsqueeze(-1)

        num_steps = query_times.shape[1]
        seq_len = context_keyframes.shape[1]
        total_len = seq_len + num_steps

        motion_features = self._compute_motion_features(context_keyframes)

        if self.periodic_time_encoding:
            kf_time = context_keyframes[:, :, 0]
            kf_time_encoded = self.time_encoder(kf_time)
            kf_rest = context_keyframes[:, :, 1:]
            kf_augmented = torch.cat([kf_time_encoded, kf_rest, motion_features], dim=-1)
            kf_embed = self.keyframe_projection(kf_augmented)
        else:
            kf_augmented = torch.cat([context_keyframes, motion_features], dim=-1)
            kf_embed = self.keyframe_projection(kf_augmented)

        prop_embed = self.property_type_embedding(property_type)
        bone_context = self.bone_identity_encoder(topology_features, bone_name_tokens)
        bone_embed = self.bone_context_projection(bone_context)
        condition = (prop_embed + bone_embed).unsqueeze(1)

        if self.pae_encoder is not None and curve_window is not None:
            phase_features = self.pae_encoder(curve_window)
            phase_embed = self.phase_projection(phase_features).unsqueeze(1)  # type: ignore[union-attr]
            condition = condition + phase_embed

        if self.multi_res_encoder is not None and curve_window is not None:
            multi_res_embed = self.multi_res_encoder(curve_window).unsqueeze(1)
            condition = condition + multi_res_embed

        base_phase = period = amplitude = None
        if self.phase_detector is not None and curve_window is not None:
            base_phase, period, amplitude = self.phase_detector.extract_phase_params(
                curve_window,
            )
            phase_context = self.phase_detector.compute_context_conditioning(
                base_phase, period, amplitude,
            ).unsqueeze(1)
            condition = condition + phase_context

        kf_positions: torch.Tensor = self.kf_positions[:seq_len]  # type: ignore[assignment]
        kf_embed = kf_embed + condition + self.positional_embedding(kf_positions).unsqueeze(0)

        query_positions: torch.Tensor = self.query_positions[:num_steps]  # type: ignore[assignment]
        condition_squeezed = condition.squeeze(1)
        query_tokens: list[torch.Tensor] = []

        for step in range(num_steps):
            step_time = query_times[:, step]

            if self.periodic_time_encoding:
                query_time_input = self.time_encoder(step_time)
            else:
                query_time_input = step_time.unsqueeze(-1)

            query_token = (
                self.query_time_projection(query_time_input)
                + condition_squeezed
                + self.positional_embedding(query_positions[step])
            )

            if (
                self.phase_detector is not None
                and base_phase is not None
                and period is not None
                and amplitude is not None
            ):
                phase_query_embed = self.phase_detector.compute_query_phase_encoding(
                    step_time, base_phase, period, amplitude,
                )
                query_token = query_token + phase_query_embed

            query_tokens.append(query_token.unsqueeze(1))

        x = torch.cat([kf_embed] + query_tokens, dim=1)

        causal_mask: torch.Tensor = self.causal_mask  # type: ignore[assignment]
        mask = causal_mask[:total_len, :total_len]

        if self.gating is not None:
            motion_state = (
                self._compute_motion_state(motion_features)
                if self.config.enhanced_gating
                else None
            )
            gate_weights = self.gating(property_type, motion_state)
            for block in self.blocks:
                x = block(x, mask, gate_weights)
        else:
            for block in self.blocks:
                x = block(x, mask)

        query_hidden = self.output_norm(x)[:, seq_len:, :]
        predictions = self.prediction_head(query_hidden)
        confidences = torch.sigmoid(self.confidence_head(query_hidden.detach()))

        return predictions, confidences


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def create_model_from_dict(config_dict: dict[str, object]) -> CurveCopilotModel:
    config = CurveCopilotConfig(**config_dict)  # type: ignore[arg-type]
    return CurveCopilotModel(config)
