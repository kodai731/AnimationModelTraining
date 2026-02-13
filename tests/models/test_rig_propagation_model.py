from __future__ import annotations

import pytest
import torch

from anim_ml.models.rig_propagation.model import (
    RigPropagationConfig,
    RigPropagationModel,
    count_parameters,
)


def _make_small_config() -> RigPropagationConfig:
    return RigPropagationConfig(
        node_feature_dim=32,
        edge_feature_dim=8,
        hidden_dim=64,
        ffn_dim=128,
        num_message_passing_layers=2,
        dropout=0.0,
    )


def _make_dummy_input(
    batch: int = 2, device: str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "joint_features": torch.randn(batch, 22, 10, device=device),
        "joint_types": torch.randint(0, 13, (batch, 22), device=device),
    }


@pytest.mark.unit
class TestForwardPass:
    def test_no_error(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input()
        rotation_deltas, confidence = model(**inputs)
        assert rotation_deltas is not None
        assert confidence is not None

    def test_output_shapes(self) -> None:
        batch = 4
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=batch)
        rotation_deltas, confidence = model(**inputs)
        assert rotation_deltas.shape == (batch, 22, 4)
        assert confidence.shape == (batch, 22, 1)

    def test_confidence_range(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=8)
        _, confidence = model(**inputs)
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()

    def test_unit_quaternion(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=4)
        rotation_deltas, _ = model(**inputs)
        norms = torch.norm(rotation_deltas, dim=-1)
        torch.testing.assert_close(norms, torch.ones_like(norms), atol=1e-5, rtol=0)


@pytest.mark.unit
class TestParameterCount:
    def test_within_budget(self) -> None:
        model = RigPropagationModel()
        params = count_parameters(model)
        assert 1_000_000 <= params <= 5_000_000, f"Parameter count {params} out of range"


@pytest.mark.unit
class TestGradientFlow:
    def test_all_parameters_receive_gradients(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input()
        rotation_deltas, confidence = model(**inputs)
        loss = rotation_deltas.sum() + confidence.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


@pytest.mark.unit
class TestMessagePropagation:
    def test_edit_affects_neighbor_joint(self) -> None:
        config = _make_small_config()
        model = RigPropagationModel(config)
        model.eval()

        base_input = _make_dummy_input(batch=1)

        with torch.no_grad():
            base_deltas, _ = model(**base_input)

        edited_input = {k: v.clone() for k, v in base_input.items()}
        edited_input["joint_features"][0, 0, 8] = 1.0
        edited_input["joint_features"][0, 0, 4:8] = torch.tensor([0.1, 0.2, 0.3, 0.9])

        with torch.no_grad():
            edited_deltas, _ = model(**edited_input)

        spine1_diff = (base_deltas[0, 3] - edited_deltas[0, 3]).abs().max().item()
        assert spine1_diff > 1e-6


@pytest.mark.unit
class TestBatchIndependence:
    def test_batch1_vs_batch2(self) -> None:
        config = _make_small_config()
        model = RigPropagationModel(config)
        model.eval()

        single_input = _make_dummy_input(batch=1)

        with torch.no_grad():
            single_deltas, single_conf = model(**single_input)

        double_features = single_input["joint_features"].repeat(2, 1, 1)
        double_types = single_input["joint_types"].repeat(2, 1)

        with torch.no_grad():
            double_deltas, double_conf = model(double_features, double_types)

        torch.testing.assert_close(
            single_deltas[0], double_deltas[0], atol=1e-5, rtol=0,
        )
        torch.testing.assert_close(
            single_conf[0], double_conf[0], atol=1e-5, rtol=0,
        )


@pytest.mark.unit
class TestVariableBatchSize:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=batch_size)
        rotation_deltas, confidence = model(**inputs)
        assert rotation_deltas.shape == (batch_size, 22, 4)
        assert confidence.shape == (batch_size, 22, 1)
