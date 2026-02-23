from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from anim_ml.models.curve_copilot.export_onnx import export_to_onnx
from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel

if TYPE_CHECKING:
    from pathlib import Path


def _make_small_model(max_steps: int = 3) -> CurveCopilotModel:
    config = CurveCopilotConfig(
        d_model=64, n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        max_steps=max_steps,
    )
    model = CurveCopilotModel(config)
    model.eval()
    return model


def _export_small_model(tmp_path: Path) -> tuple[Path, CurveCopilotModel]:
    model = _make_small_model()
    onnx_path = tmp_path / "test_model.onnx"
    export_to_onnx(model, onnx_path)
    return onnx_path, model


@pytest.mark.unit
class TestOnnxChecker:
    def test_valid_model(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)


@pytest.mark.unit
class TestOnnxOpset:
    def test_opset_at_least_17(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        model = onnx.load(str(onnx_path))
        opset = model.opset_import[0].version
        assert opset >= 17


@pytest.mark.unit
class TestPytorchOrtParity:
    def test_output_matches(self, tmp_path: Path) -> None:
        onnx_path, pytorch_model = _export_small_model(tmp_path)
        max_steps = pytorch_model.config.max_steps

        context = torch.randn(2, 8, 6)
        prop_type = torch.tensor([0, 3], dtype=torch.long)
        topo_features = torch.rand(2, 6)
        bone_tokens = torch.randint(0, 64, (2, 32))
        query_times = torch.rand(2, max_steps)

        with torch.no_grad():
            pt_pred, pt_conf = pytorch_model(
                context, prop_type, topo_features, bone_tokens, query_times,
            )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "context_keyframes": context.numpy(),
            "property_type": prop_type.numpy(),
            "topology_features": topo_features.numpy(),
            "bone_name_tokens": bone_tokens.numpy(),
            "query_times": query_times.numpy(),
        }
        ort_pred, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_pred.numpy(), ort_pred, atol=1e-5)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-5)


@pytest.mark.unit
class TestDynamicBatch:
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_different_batch_sizes(self, tmp_path: Path, batch_size: int) -> None:
        onnx_path, model = _export_small_model(tmp_path)
        max_steps = model.config.max_steps
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(batch_size, 8, 6).astype(np.float32),
            "property_type": np.zeros(batch_size, dtype=np.int64),
            "topology_features": np.random.rand(batch_size, 6).astype(np.float32),
            "bone_name_tokens": np.zeros((batch_size, 32), dtype=np.int64),
            "query_times": np.random.rand(batch_size, max_steps).astype(np.float32),
        }
        pred, conf = session.run(None, inputs)

        assert pred.shape == (batch_size, max_steps, 6)
        assert conf.shape == (batch_size, max_steps, 1)


@pytest.mark.unit
class TestModelSize:
    def test_under_20mb(self, tmp_path: Path) -> None:
        config = CurveCopilotConfig()
        model = CurveCopilotModel(config)
        model.eval()

        onnx_path = tmp_path / "full_model.onnx"
        export_to_onnx(model, onnx_path)

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        assert size_mb < 20, f"Model size {size_mb:.1f} MB exceeds 20 MB limit"


def _make_small_expert_model() -> CurveCopilotModel:
    config = CurveCopilotConfig(
        d_model=64, n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        use_expert_mixing=True, num_experts=3, max_steps=3,
    )
    model = CurveCopilotModel(config)
    model.eval()
    return model


def _export_small_expert_model(tmp_path: Path) -> tuple[Path, CurveCopilotModel]:
    model = _make_small_expert_model()
    onnx_path = tmp_path / "test_expert_model.onnx"
    export_to_onnx(model, onnx_path)
    return onnx_path, model


@pytest.mark.unit
class TestExpertMixingOnnx:
    def test_valid_model(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_expert_model(tmp_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_output_matches(self, tmp_path: Path) -> None:
        onnx_path, pytorch_model = _export_small_expert_model(tmp_path)
        max_steps = pytorch_model.config.max_steps

        context = torch.randn(2, 8, 6)
        prop_type = torch.tensor([0, 3], dtype=torch.long)
        topo_features = torch.rand(2, 6)
        bone_tokens = torch.randint(0, 64, (2, 32))
        query_times = torch.rand(2, max_steps)

        with torch.no_grad():
            pt_pred, pt_conf = pytorch_model(
                context, prop_type, topo_features, bone_tokens, query_times,
            )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "context_keyframes": context.numpy(),
            "property_type": prop_type.numpy(),
            "topology_features": topo_features.numpy(),
            "bone_name_tokens": bone_tokens.numpy(),
            "query_times": query_times.numpy(),
        }
        ort_pred, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_pred.numpy(), ort_pred, atol=1e-5)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-5)

    def test_latency_under_5ms(self, tmp_path: Path) -> None:
        onnx_path, model = _export_small_expert_model(tmp_path)
        max_steps = model.config.max_steps
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(1, 8, 6).astype(np.float32),
            "property_type": np.zeros(1, dtype=np.int64),
            "topology_features": np.random.rand(1, 6).astype(np.float32),
            "bone_name_tokens": np.zeros((1, 32), dtype=np.int64),
            "query_times": np.random.rand(1, max_steps).astype(np.float32),
        }

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"


@pytest.mark.unit
class TestLatency:
    def test_batch1_under_5ms(self, tmp_path: Path) -> None:
        onnx_path, model = _export_small_model(tmp_path)
        max_steps = model.config.max_steps
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(1, 8, 6).astype(np.float32),
            "property_type": np.zeros(1, dtype=np.int64),
            "topology_features": np.random.rand(1, 6).astype(np.float32),
            "bone_name_tokens": np.zeros((1, 32), dtype=np.int64),
            "query_times": np.random.rand(1, max_steps).astype(np.float32),
        }

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"


def _make_small_pae_model() -> CurveCopilotModel:
    config = CurveCopilotConfig(
        d_model=64, n_heads=2, d_ff=128, n_layers=2, dropout=0.0,
        use_pae=True, pae_window_size=64, pae_latent_channels=5,
        max_steps=3,
    )
    model = CurveCopilotModel(config)
    model.eval()
    return model


def _export_small_pae_model(tmp_path: Path) -> tuple[Path, CurveCopilotModel]:
    model = _make_small_pae_model()
    onnx_path = tmp_path / "test_pae_model.onnx"
    export_to_onnx(model, onnx_path)
    return onnx_path, model


@pytest.mark.unit
class TestPAEOnnx:
    def test_valid_model(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_pae_model(tmp_path)
        model = onnx.load(str(onnx_path))
        onnx.checker.check_model(model)

    def test_output_matches(self, tmp_path: Path) -> None:
        onnx_path, pytorch_model = _export_small_pae_model(tmp_path)
        max_steps = pytorch_model.config.max_steps

        context = torch.randn(2, 8, 6)
        prop_type = torch.tensor([0, 3], dtype=torch.long)
        topo_features = torch.rand(2, 6)
        bone_tokens = torch.randint(0, 64, (2, 32))
        query_times = torch.rand(2, max_steps)
        curve_window = torch.randn(2, 64)

        with torch.no_grad():
            pt_pred, pt_conf = pytorch_model(
                context, prop_type, topo_features, bone_tokens, query_times,
                curve_window=curve_window,
            )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "context_keyframes": context.numpy(),
            "property_type": prop_type.numpy(),
            "topology_features": topo_features.numpy(),
            "bone_name_tokens": bone_tokens.numpy(),
            "query_times": query_times.numpy(),
            "curve_window": curve_window.numpy(),
        }
        ort_pred, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_pred.numpy(), ort_pred, atol=1e-5)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-5)

    def test_latency_under_5ms(self, tmp_path: Path) -> None:
        onnx_path, model = _export_small_pae_model(tmp_path)
        max_steps = model.config.max_steps
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(1, 8, 6).astype(np.float32),
            "property_type": np.zeros(1, dtype=np.int64),
            "topology_features": np.random.rand(1, 6).astype(np.float32),
            "bone_name_tokens": np.zeros((1, 32), dtype=np.int64),
            "query_times": np.random.rand(1, max_steps).astype(np.float32),
            "curve_window": np.random.randn(1, 64).astype(np.float32),
        }

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"
