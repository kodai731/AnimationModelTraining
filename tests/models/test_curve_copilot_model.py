from __future__ import annotations

import pytest
import torch

from anim_ml.models.curve_copilot.model import (
    CurveCopilotConfig,
    CurveCopilotModel,
    GatingNetwork,
    PeriodicTimeEncoder,
    PropertyAdaptiveBlock,
    count_parameters,
)


def _make_dummy_input(
    batch: int = 2, seq_len: int = 8, device: str = "cpu",
) -> dict[str, torch.Tensor]:
    return {
        "context_keyframes": torch.randn(batch, seq_len, 6, device=device),
        "property_type": torch.randint(0, 9, (batch,), device=device),
        "topology_features": torch.rand(batch, 6, device=device),
        "bone_name_tokens": torch.randint(0, 64, (batch, 32), device=device),
        "query_time": torch.rand(batch, device=device),
    }


@pytest.mark.unit
class TestForwardPass:
    def test_no_error(self) -> None:
        model = CurveCopilotModel()
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        assert prediction is not None
        assert confidence is not None

    def test_output_shapes(self) -> None:
        batch = 4
        model = CurveCopilotModel()
        inputs = _make_dummy_input(batch=batch)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (batch, 6)
        assert confidence.shape == (batch, 1)

    def test_confidence_range(self) -> None:
        model = CurveCopilotModel()
        inputs = _make_dummy_input(batch=16)
        _, confidence = model(**inputs)
        assert (confidence >= 0.0).all()
        assert (confidence <= 1.0).all()


@pytest.mark.unit
class TestParameterCount:
    def test_within_budget(self) -> None:
        model = CurveCopilotModel()
        params = count_parameters(model)
        assert 1_000_000 <= params <= 5_000_000, f"Parameter count {params} out of range"


@pytest.mark.unit
class TestCausalMasking:
    def test_future_tokens_do_not_affect_past(self) -> None:
        config = CurveCopilotConfig(n_layers=2, d_model=64, d_ff=128, n_heads=2, dropout=0.0)
        model = CurveCopilotModel(config)
        model.eval()

        batch = 1
        seq_len = 4

        base_input = _make_dummy_input(batch=batch, seq_len=seq_len)

        intermediate_outputs: list[torch.Tensor] = []

        def hook_fn(module: torch.nn.Module, input: object, output: torch.Tensor) -> None:
            intermediate_outputs.append(output.detach().clone())

        handle = model.blocks[0].register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**base_input)
        baseline_hidden = intermediate_outputs[0]

        modified_input = {k: v.clone() for k, v in base_input.items()}
        modified_input["context_keyframes"][:, -1, :] += 100.0

        intermediate_outputs.clear()
        with torch.no_grad():
            model(**modified_input)
        modified_hidden = intermediate_outputs[0]

        handle.remove()

        for pos in range(seq_len - 1):
            diff = (baseline_hidden[:, pos, :] - modified_hidden[:, pos, :]).abs().max().item()
            assert diff < 1e-5, f"Position {pos} changed by {diff} when modifying later token"


@pytest.mark.unit
class TestGradientFlow:
    def test_all_parameters_receive_gradients(self) -> None:
        model = CurveCopilotModel()
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        loss = prediction.sum() + confidence.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


@pytest.mark.unit
class TestVariableSequenceLength:
    @pytest.mark.parametrize("seq_len", [1, 2, 4, 8])
    def test_different_lengths(self, seq_len: int) -> None:
        model = CurveCopilotModel()
        inputs = _make_dummy_input(batch=2, seq_len=seq_len)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)


def _make_periodic_config(**overrides: object) -> CurveCopilotConfig:
    defaults: dict[str, object] = {"periodic_time_encoding": True}
    defaults.update(overrides)
    return CurveCopilotConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestPeriodicTimeEncoder:
    def test_1d_input(self) -> None:
        encoder = PeriodicTimeEncoder()
        t = torch.tensor([0.0, 0.25, 0.5, 1.0])
        out = encoder(t)
        assert out.shape == (4, 3)

    def test_2d_input(self) -> None:
        encoder = PeriodicTimeEncoder()
        t = torch.rand(2, 8)
        out = encoder(t)
        assert out.shape == (2, 8, 3)

    def test_known_values(self) -> None:
        encoder = PeriodicTimeEncoder()
        t = torch.tensor([0.0])
        out = encoder(t)
        assert torch.allclose(out, torch.tensor([[0.0, 0.0, 1.0]]), atol=1e-6)


@pytest.mark.unit
class TestPeriodicTimeEncoding:
    def test_forward_pass(self) -> None:
        config = _make_periodic_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)

    def test_output_shapes_unchanged(self) -> None:
        config = _make_periodic_config()
        model = CurveCopilotModel(config)
        batch = 4
        inputs = _make_dummy_input(batch=batch)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (batch, 6)
        assert confidence.shape == (batch, 1)

    def test_within_budget(self) -> None:
        config = _make_periodic_config()
        model = CurveCopilotModel(config)
        params = count_parameters(model)
        assert 1_000_000 <= params <= 5_000_000, f"Parameter count {params} out of range"

    @pytest.mark.parametrize("seq_len", [1, 2, 4, 8])
    def test_variable_sequence_length(self, seq_len: int) -> None:
        config = _make_periodic_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input(batch=2, seq_len=seq_len)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)

    def test_causal_masking(self) -> None:
        config = _make_periodic_config(
            n_layers=2, d_model=64, d_ff=128, n_heads=2, dropout=0.0,
        )
        model = CurveCopilotModel(config)
        model.eval()

        base_input = _make_dummy_input(batch=1, seq_len=4)

        intermediate_outputs: list[torch.Tensor] = []

        def hook_fn(module: torch.nn.Module, input: object, output: torch.Tensor) -> None:
            intermediate_outputs.append(output.detach().clone())

        handle = model.blocks[0].register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**base_input)
        baseline_hidden = intermediate_outputs[0]

        modified_input = {k: v.clone() for k, v in base_input.items()}
        modified_input["context_keyframes"][:, -1, :] += 100.0

        intermediate_outputs.clear()
        with torch.no_grad():
            model(**modified_input)
        modified_hidden = intermediate_outputs[0]

        handle.remove()

        for pos in range(3):
            diff = (baseline_hidden[:, pos, :] - modified_hidden[:, pos, :]).abs().max().item()
            assert diff < 1e-5, f"Position {pos} changed by {diff} when modifying later token"

    def test_all_parameters_receive_gradients(self) -> None:
        config = _make_periodic_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        loss = prediction.sum() + confidence.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


def _make_expert_config(**overrides: object) -> CurveCopilotConfig:
    defaults: dict[str, object] = {"use_expert_mixing": True, "num_experts": 3}
    defaults.update(overrides)
    return CurveCopilotConfig(**defaults)  # type: ignore[arg-type]


@pytest.mark.unit
class TestGatingNetwork:
    def test_output_shape(self) -> None:
        num_experts = 3
        gating = GatingNetwork(num_property_types=9, d_model=128, num_experts=num_experts)
        property_type = torch.randint(0, 9, (4,))
        weights = gating(property_type)
        assert weights.shape == (4, num_experts)

    def test_weights_sum_to_one(self) -> None:
        gating = GatingNetwork(num_property_types=9, d_model=128, num_experts=3)
        property_type = torch.randint(0, 9, (8,))
        weights = gating(property_type)
        sums = weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), atol=1e-5)


@pytest.mark.unit
class TestPropertyAdaptiveBlock:
    def test_output_shape(self) -> None:
        block = PropertyAdaptiveBlock(
            d_model=64, n_heads=2, d_ff=128, dropout=0.0, num_experts=3,
        )
        x = torch.randn(2, 5, 64)
        mask = torch.triu(torch.ones(5, 5), diagonal=1).bool()
        gate_weights = torch.softmax(torch.randn(2, 3), dim=-1)
        out = block(x, mask, gate_weights)
        assert out.shape == (2, 5, 64)


@pytest.mark.unit
class TestExpertMixing:
    def test_forward_pass(self) -> None:
        config = _make_expert_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)

    def test_output_shapes_unchanged(self) -> None:
        config = _make_expert_config()
        model = CurveCopilotModel(config)
        batch = 4
        inputs = _make_dummy_input(batch=batch)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (batch, 6)
        assert confidence.shape == (batch, 1)

    def test_within_budget(self) -> None:
        config = _make_expert_config()
        model = CurveCopilotModel(config)
        params = count_parameters(model)
        assert 1_000_000 <= params <= 5_000_000, f"Parameter count {params} out of range"

    @pytest.mark.parametrize("seq_len", [1, 2, 4, 8])
    def test_variable_sequence_length(self, seq_len: int) -> None:
        config = _make_expert_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input(batch=2, seq_len=seq_len)
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)

    def test_causal_masking(self) -> None:
        config = _make_expert_config(
            n_layers=2, d_model=64, d_ff=128, n_heads=2, dropout=0.0,
        )
        model = CurveCopilotModel(config)
        model.eval()

        base_input = _make_dummy_input(batch=1, seq_len=4)

        intermediate_outputs: list[torch.Tensor] = []

        def hook_fn(module: torch.nn.Module, input: object, output: torch.Tensor) -> None:
            intermediate_outputs.append(output.detach().clone())

        handle = model.blocks[0].experts[0].register_forward_hook(hook_fn)

        with torch.no_grad():
            model(**base_input)
        baseline_hidden = intermediate_outputs[0]

        modified_input = {k: v.clone() for k, v in base_input.items()}
        modified_input["context_keyframes"][:, -1, :] += 100.0

        intermediate_outputs.clear()
        with torch.no_grad():
            model(**modified_input)
        modified_hidden = intermediate_outputs[0]

        handle.remove()

        for pos in range(3):
            diff = (baseline_hidden[:, pos, :] - modified_hidden[:, pos, :]).abs().max().item()
            assert diff < 1e-5, f"Position {pos} changed by {diff} when modifying later token"

    def test_all_parameters_receive_gradients(self) -> None:
        config = _make_expert_config()
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        loss = prediction.sum() + confidence.sum()
        loss.backward()

        for name, param in model.named_parameters():
            if param.requires_grad:
                assert param.grad is not None, f"No gradient for {name}"
                assert param.grad.abs().sum() > 0, f"Zero gradient for {name}"


@pytest.mark.unit
class TestExpertMixingWithPeriodic:
    def test_forward_pass(self) -> None:
        config = _make_expert_config(periodic_time_encoding=True)
        model = CurveCopilotModel(config)
        inputs = _make_dummy_input()
        prediction, confidence = model(**inputs)
        assert prediction.shape == (2, 6)
        assert confidence.shape == (2, 1)

    def test_within_budget(self) -> None:
        config = _make_expert_config(periodic_time_encoding=True)
        model = CurveCopilotModel(config)
        params = count_parameters(model)
        assert 1_000_000 <= params <= 5_000_000, f"Parameter count {params} out of range"
