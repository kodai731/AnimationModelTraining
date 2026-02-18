from __future__ import annotations

import pytest
import torch

from anim_ml.data.rig_data_generator import MAX_EDGES, MAX_JOINTS
from anim_ml.models.rig_propagation.model import (
    RigPropagationConfig,
    RigPropagationModel,
    count_parameters,
)

SAMPLE_PARENT_INDICES = [
    -1, 0, 0, 0,
    1, 2, 3,
    4, 5, 6,
    9, 9, 9,
    10, 11, 12,
    14, 15,
    16, 17,
]
NUM_SAMPLE_JOINTS = len(SAMPLE_PARENT_INDICES)


def _make_small_config() -> RigPropagationConfig:
    return RigPropagationConfig(
        node_feature_dim=32,
        edge_feature_dim=8,
        hidden_dim=64,
        ffn_dim=128,
        num_message_passing_layers=2,
        dropout=0.0,
    )


def _build_edge_tensors(
    parent_indices: list[int],
    device: str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    src_list: list[int] = []
    tgt_list: list[int] = []
    for child, parent in enumerate(parent_indices):
        if parent == -1:
            continue
        src_list.extend([parent, child])
        tgt_list.extend([child, parent])

    num_edges = len(src_list)
    source = torch.zeros(MAX_EDGES, dtype=torch.long, device=device)
    target = torch.zeros(MAX_EDGES, dtype=torch.long, device=device)
    direction = torch.zeros(MAX_EDGES, dtype=torch.long, device=device)
    mask = torch.zeros(MAX_EDGES, dtype=torch.float, device=device)

    source[:num_edges] = torch.tensor(src_list)
    target[:num_edges] = torch.tensor(tgt_list)
    for i in range(num_edges):
        direction[i] = 0 if parent_indices[tgt_list[i]] == src_list[i] else 1
    mask[:num_edges] = 1.0

    return source, target, direction, mask


def _make_dummy_input(
    batch: int = 2,
    device: str = "cpu",
) -> dict[str, torch.Tensor]:
    joint_mask = torch.zeros(batch, MAX_JOINTS, device=device)
    joint_mask[:, :NUM_SAMPLE_JOINTS] = 1.0

    source, target, direction, edge_mask = _build_edge_tensors(SAMPLE_PARENT_INDICES, device)

    return {
        "joint_features": torch.randn(batch, MAX_JOINTS, 9, device=device),
        "topology_features": torch.randn(batch, MAX_JOINTS, 6, device=device),
        "bone_name_tokens": torch.randint(0, 44, (batch, MAX_JOINTS, 32), device=device),
        "joint_mask": joint_mask,
        "source_indices": source,
        "target_indices": target,
        "edge_direction": direction,
        "edge_mask": edge_mask,
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
        assert rotation_deltas.shape == (batch, MAX_JOINTS, 4)
        assert confidence.shape == (batch, MAX_JOINTS, 1)

    def test_confidence_range(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=8)
        _, confidence = model(**inputs)
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()

    def test_unit_quaternion_in_valid_joints(self) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=4)
        rotation_deltas, _ = model(**inputs)
        valid = rotation_deltas[:, :NUM_SAMPLE_JOINTS, :]
        norms = torch.norm(valid, dim=-1)
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
class TestMaskedOutput:
    def test_padding_joints_are_zero(self) -> None:
        model = RigPropagationModel(_make_small_config())
        model.eval()
        inputs = _make_dummy_input(batch=2)
        with torch.no_grad():
            rotation_deltas, confidence = model(**inputs)

        padding_deltas = rotation_deltas[:, NUM_SAMPLE_JOINTS:, :]
        padding_conf = confidence[:, NUM_SAMPLE_JOINTS:, :]
        torch.testing.assert_close(padding_deltas, torch.zeros_like(padding_deltas))
        torch.testing.assert_close(padding_conf, torch.zeros_like(padding_conf))


@pytest.mark.unit
class TestVariableBatchSize:
    @pytest.mark.parametrize("batch_size", [1, 4, 16])
    def test_different_batch_sizes(self, batch_size: int) -> None:
        model = RigPropagationModel(_make_small_config())
        inputs = _make_dummy_input(batch=batch_size)
        rotation_deltas, confidence = model(**inputs)
        assert rotation_deltas.shape == (batch_size, MAX_JOINTS, 4)
        assert confidence.shape == (batch_size, MAX_JOINTS, 1)
