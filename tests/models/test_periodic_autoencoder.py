from __future__ import annotations

import pytest
import torch

from anim_ml.models.periodic_autoencoder.model import (
    PAEConfig,
    PAEDecoder,
    PAEEncoder,
    PeriodicAutoencoder,
)


@pytest.mark.unit
class TestPAEEncoder:
    def test_output_shape(self) -> None:
        encoder = PAEEncoder()
        x = torch.randn(4, 64)
        out = encoder(x)
        assert out.shape == (4, 15)

    def test_batch_size_one(self) -> None:
        encoder = PAEEncoder()
        x = torch.randn(1, 64)
        out = encoder(x)
        assert out.shape == (1, 15)

    def test_custom_config(self) -> None:
        config = PAEConfig(window_size=64, latent_channels=8, feature_dim=24)
        encoder = PAEEncoder(config)
        x = torch.randn(2, 64)
        out = encoder(x)
        assert out.shape == (2, 24)


@pytest.mark.unit
class TestPAEDecoder:
    def test_output_shape(self) -> None:
        decoder = PAEDecoder()
        z = torch.randn(4, 15)
        out = decoder(z)
        assert out.shape == (4, 64)

    def test_batch_size_one(self) -> None:
        decoder = PAEDecoder()
        z = torch.randn(1, 15)
        out = decoder(z)
        assert out.shape == (1, 64)


@pytest.mark.unit
class TestPeriodicAutoencoder:
    def test_reconstruction_loop(self) -> None:
        pae = PeriodicAutoencoder()
        x = torch.randn(4, 64)
        features, reconstructed = pae(x)
        assert features.shape == (4, 15)
        assert reconstructed.shape == (4, 64)

    def test_gradient_flow(self) -> None:
        pae = PeriodicAutoencoder()
        x = torch.randn(4, 64)
        _, reconstructed = pae(x)
        loss = reconstructed.sum()
        loss.backward()

        for name, param in pae.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


@pytest.mark.unit
class TestParameterCount:
    def test_within_budget(self) -> None:
        pae = PeriodicAutoencoder()
        param_count = sum(p.numel() for p in pae.parameters())
        assert param_count < 100_000, f"PAE parameter count {param_count} exceeds 100K"

    def test_encoder_decoder_balanced(self) -> None:
        pae = PeriodicAutoencoder()
        enc_params = sum(p.numel() for p in pae.encoder.parameters())
        dec_params = sum(p.numel() for p in pae.decoder.parameters())
        ratio = enc_params / max(dec_params, 1)
        assert 0.5 < ratio < 2.0, f"Encoder/Decoder ratio {ratio:.2f} is unbalanced"
