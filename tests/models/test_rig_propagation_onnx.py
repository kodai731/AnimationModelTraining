from __future__ import annotations

from pathlib import Path

import numpy as np
import onnx
import onnxruntime as ort
import pytest
import torch

from anim_ml.data.rig_data_generator import MAX_EDGES, MAX_JOINTS
from anim_ml.models.rig_propagation.export_onnx import export_to_onnx
from anim_ml.models.rig_propagation.model import RigPropagationConfig, RigPropagationModel

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


def _build_edge_arrays() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    src_list: list[int] = []
    tgt_list: list[int] = []
    for child, parent in enumerate(SAMPLE_PARENT_INDICES):
        if parent == -1:
            continue
        src_list.extend([parent, child])
        tgt_list.extend([child, parent])

    num_edges = len(src_list)
    source = np.zeros(MAX_EDGES, dtype=np.int64)
    target = np.zeros(MAX_EDGES, dtype=np.int64)
    direction = np.zeros(MAX_EDGES, dtype=np.int64)
    mask = np.zeros(MAX_EDGES, dtype=np.float32)

    source[:num_edges] = src_list
    target[:num_edges] = tgt_list
    for i in range(num_edges):
        direction[i] = 0 if SAMPLE_PARENT_INDICES[tgt_list[i]] == src_list[i] else 1
    mask[:num_edges] = 1.0

    return source, target, direction, mask


def _make_ort_inputs(batch_size: int) -> dict[str, np.ndarray]:
    source, target, direction, mask = _build_edge_arrays()

    joint_mask = np.zeros((batch_size, MAX_JOINTS), dtype=np.float32)
    joint_mask[:, :NUM_SAMPLE_JOINTS] = 1.0

    return {
        "joint_features": np.random.randn(batch_size, MAX_JOINTS, 9).astype(np.float32),
        "topology_features": np.random.randn(batch_size, MAX_JOINTS, 6).astype(np.float32),
        "bone_name_tokens": np.zeros((batch_size, MAX_JOINTS, 32), dtype=np.int64),
        "joint_mask": joint_mask,
        "source_indices": source,
        "target_indices": target,
        "edge_direction": direction,
        "edge_mask": mask,
    }


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

        source, target, direction, edge_mask = _build_edge_arrays()
        joint_mask_np = np.zeros((2, MAX_JOINTS), dtype=np.float32)
        joint_mask_np[:, :NUM_SAMPLE_JOINTS] = 1.0

        features = torch.randn(2, MAX_JOINTS, 9)
        topo = torch.randn(2, MAX_JOINTS, 6)
        tokens = torch.zeros(2, MAX_JOINTS, 32, dtype=torch.long)
        joint_mask = torch.from_numpy(joint_mask_np)
        src_t = torch.from_numpy(source)
        tgt_t = torch.from_numpy(target)
        dir_t = torch.from_numpy(direction)
        emask_t = torch.from_numpy(edge_mask)

        with torch.no_grad():
            pt_deltas, pt_conf = pytorch_model(
                features, topo, tokens, joint_mask,
                src_t, tgt_t, dir_t, emask_t,
            )

        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        ort_inputs = {
            "joint_features": features.numpy(),
            "topology_features": topo.numpy(),
            "bone_name_tokens": tokens.numpy(),
            "joint_mask": joint_mask.numpy(),
            "source_indices": source,
            "target_indices": target,
            "edge_direction": direction,
            "edge_mask": edge_mask,
        }
        ort_deltas, ort_conf = session.run(None, ort_inputs)

        np.testing.assert_allclose(pt_deltas.numpy(), ort_deltas, atol=1e-4)
        np.testing.assert_allclose(pt_conf.numpy(), ort_conf, atol=1e-4)


@pytest.mark.unit
class TestDynamicBatch:
    @pytest.mark.parametrize("batch_size", [1, 4, 32])
    def test_different_batch_sizes(self, tmp_path: Path, batch_size: int) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])

        inputs = _make_ort_inputs(batch_size)
        deltas, conf = session.run(None, inputs)

        assert deltas.shape == (batch_size, MAX_JOINTS, 4)
        assert conf.shape == (batch_size, MAX_JOINTS, 1)


@pytest.mark.unit
class TestInputNames:
    def test_expected_inputs(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        input_names = {inp.name for inp in session.get_inputs()}
        expected = {
            "joint_features", "topology_features", "bone_name_tokens", "joint_mask",
            "source_indices", "target_indices", "edge_direction", "edge_mask",
        }
        assert input_names == expected


@pytest.mark.unit
class TestOutputProperties:
    def test_confidence_range(self, tmp_path: Path) -> None:
        onnx_path, _ = _export_small_model(tmp_path)
        session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        inputs = _make_ort_inputs(4)
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

        inputs = _make_ort_inputs(1)

        from scripts.verify_onnx import measure_latency

        latency = measure_latency(session, inputs, num_runs=50)
        assert latency["p95_ms"] < 5.0, f"p95 latency {latency['p95_ms']:.2f} ms exceeds 5 ms"
