from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from anim_ml.models.curve_copilot.export_onnx import export_to_onnx
from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel


def _make_small_model() -> CurveCopilotModel:
    config = CurveCopilotConfig(d_model=64, n_heads=2, d_ff=128, n_layers=2, dropout=0.0)
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

        context = torch.randn(2, 8, 6)
        prop_type = torch.tensor([0, 3], dtype=torch.long)
        joint_cat = torch.tensor([1, 2], dtype=torch.long)
        query_time = torch.tensor([0.3, 0.7])

        with torch.no_grad():
            pt_pred, pt_conf = pytorch_model(context, prop_type, joint_cat, query_time)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "context_keyframes": context.numpy(),
            "property_type": prop_type.numpy(),
            "joint_category": joint_cat.numpy(),
            "query_time": query_time.numpy(),
        }
        ort_pred, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_pred.numpy(), ort_pred, atol=1e-5)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-5)


@pytest.mark.unit
class TestDynamicBatch:
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_different_batch_sizes(self, tmp_path: Path, batch_size: int) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(batch_size, 8, 6).astype(np.float32),
            "property_type": np.zeros(batch_size, dtype=np.int64),
            "joint_category": np.zeros(batch_size, dtype=np.int64),
            "query_time": np.random.rand(batch_size).astype(np.float32),
        }
        pred, conf = session.run(None, inputs)

        assert pred.shape == (batch_size, 6)
        assert conf.shape == (batch_size, 1)


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


@pytest.mark.unit
class TestLatency:
    def test_batch1_under_5ms(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "context_keyframes": np.random.randn(1, 8, 6).astype(np.float32),
            "property_type": np.zeros(1, dtype=np.int64),
            "joint_category": np.zeros(1, dtype=np.int64),
            "query_time": np.array([0.5], dtype=np.float32),
        }

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"
