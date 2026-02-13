from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from anim_ml.models.rig_propagation.export_onnx import export_to_onnx
from anim_ml.models.rig_propagation.model import RigPropagationConfig, RigPropagationModel


def _make_small_model() -> RigPropagationModel:
    config = RigPropagationConfig(
        node_feature_dim=32,
        edge_feature_dim=8,
        hidden_dim=64,
        ffn_dim=128,
        num_message_passing_layers=2,
        dropout=0.0,
    )
    model = RigPropagationModel(config)
    model.eval()
    return model


def _export_small_model(tmp_path: Path) -> tuple[Path, RigPropagationModel]:
    model = _make_small_model()
    onnx_path = tmp_path / "test_rig_model.onnx"
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

        features = torch.randn(2, 22, 10)
        types = torch.randint(0, 13, (2, 22))

        with torch.no_grad():
            pt_deltas, pt_conf = pytorch_model(features, types)

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "joint_features": features.numpy(),
            "joint_types": types.numpy(),
        }
        ort_deltas, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_deltas.numpy(), ort_deltas, atol=1e-5)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-5)


@pytest.mark.unit
class TestDynamicBatch:
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_different_batch_sizes(self, tmp_path: Path, batch_size: int) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "joint_features": np.random.randn(batch_size, 22, 10).astype(np.float32),
            "joint_types": np.zeros((batch_size, 22), dtype=np.int64),
        }
        deltas, conf = session.run(None, inputs)

        assert deltas.shape == (batch_size, 22, 4)
        assert conf.shape == (batch_size, 22, 1)


@pytest.mark.unit
class TestFixedTopology:
    def test_no_adjacency_input(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = [inp.name for inp in session.get_inputs()]
        assert "adjacency" not in input_names
        assert set(input_names) == {"joint_features", "joint_types"}


@pytest.mark.unit
class TestOutputProperties:
    def test_unit_quaternion(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "joint_features": np.random.randn(4, 22, 10).astype(np.float32),
            "joint_types": np.zeros((4, 22), dtype=np.int64),
        }
        deltas, _ = session.run(None, inputs)
        norms = np.linalg.norm(deltas, axis=-1)
        np.testing.assert_allclose(norms, 1.0, atol=1e-5)

    def test_confidence_range(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "joint_features": np.random.randn(4, 22, 10).astype(np.float32),
            "joint_types": np.zeros((4, 22), dtype=np.int64),
        }
        _, conf = session.run(None, inputs)
        assert np.all(conf >= 0.0)
        assert np.all(conf <= 1.0)


@pytest.mark.unit
class TestModelSize:
    def test_under_20mb(self, tmp_path: Path) -> None:
        model = RigPropagationModel()
        model.eval()

        onnx_path = tmp_path / "full_rig_model.onnx"
        export_to_onnx(model, onnx_path)

        size_mb = onnx_path.stat().st_size / (1024 * 1024)
        assert size_mb < 20, f"Model size {size_mb:.1f} MB exceeds 20 MB limit"


@pytest.mark.unit
class TestLatency:
    def test_batch1_under_5ms(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = {
            "joint_features": np.random.randn(1, 22, 10).astype(np.float32),
            "joint_types": np.zeros((1, 22), dtype=np.int64),
        }

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"
