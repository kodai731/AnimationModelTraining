from __future__ import annotations

import math
import tempfile
from concurrent import futures
from pathlib import Path

import grpc
import pytest
import yaml

from anim_ml.server.proto import (
    BEZIER,
    LINEAR,
    VRM_HUMANOID,
    MotionRequest,
    MotionResponse,
    StatusRequest,
    StatusResponse,
    TextToMotionServiceStub,
    add_TextToMotionServiceServicer_to_server,
)
from anim_ml.server.service import (
    AnimationServicer,
    ServerConfig,
    create_model,
    load_server_config,
)
from anim_ml.utils.skeleton import SMPL_TO_VRM_MAPPING


@pytest.fixture
def grpc_channel():
    config = ServerConfig(port=0, model_name="mock")
    model = create_model(config)
    model.load("cpu")

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=2))
    servicer = AnimationServicer(model, config)
    add_TextToMotionServiceServicer_to_server(servicer, server)

    port = server.add_insecure_port("[::]:0")
    server.start()

    channel = grpc.insecure_channel(f"localhost:{port}")

    yield channel

    channel.close()
    server.stop(grace=0)


@pytest.mark.unit
class TestGetServerStatus:
    def test_ready_status(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        response: StatusResponse = stub.GetServerStatus(StatusRequest())

        assert response.ready is True
        assert response.active_model == "mock"
        assert response.gpu_memory_mb == 0


@pytest.mark.unit
class TestGenerateMotion:
    def test_basic_request(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(
            prompt="a person walking forward",
            duration_seconds=2.0,
            target_fps=30,
            skeleton_type=VRM_HUMANOID,
        )
        response: MotionResponse = stub.GenerateMotion(request)

        assert len(response.curves) > 0
        assert response.generation_time_ms > 0
        assert response.model_used == "mock"

    def test_curves_have_keyframes(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            assert len(curve.keyframes) >= 2
            assert curve.bone_name != ""

    def test_empty_prompt_error(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="", duration_seconds=2.0)

        with pytest.raises(grpc.RpcError) as exc_info:
            stub.GenerateMotion(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_large_duration_clamped(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=999.0, target_fps=30)

        response: MotionResponse = stub.GenerateMotion(request)
        assert len(response.curves) > 0

    def test_small_duration_clamped_to_minimum(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=0.1, target_fps=30)

        response: MotionResponse = stub.GenerateMotion(request)
        assert len(response.curves) > 0

    def test_invalid_fps_error(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=2.0, target_fps=24)

        with pytest.raises(grpc.RpcError) as exc_info:
            stub.GenerateMotion(request)

        assert exc_info.value.code() == grpc.StatusCode.INVALID_ARGUMENT

    def test_fps_60_accepted(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=2.0, target_fps=60)

        response: MotionResponse = stub.GenerateMotion(request)
        assert len(response.curves) > 0

    def test_generation_time_reported(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=1.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)
        assert response.generation_time_ms > 0

    def test_concurrent_requests(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk", duration_seconds=1.0, target_fps=30)

        response_futures = [
            stub.GenerateMotion.future(request) for _ in range(4)
        ]
        responses = [f.result(timeout=30) for f in response_futures]

        for resp in responses:
            assert len(resp.curves) > 0
            assert resp.model_used == "mock"


@pytest.mark.unit
class TestServerConfig:
    def test_load_config(self) -> None:
        config_data = {
            "server": {"port": 9999, "max_workers": 2},
            "model": {"name": "mock", "device": "cpu", "checkpoint": ""},
            "generation": {
                "default_fps": 60,
                "max_duration_seconds": 5.0,
                "default_duration_seconds": 2.0,
            },
        }

        with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
            yaml.dump(config_data, f)
            tmp_path = f.name

        config = load_server_config(tmp_path)

        assert config.port == 9999
        assert config.max_workers == 2
        assert config.model_name == "mock"
        assert config.model_device == "cpu"
        assert config.default_fps == 60
        assert config.max_duration_seconds == 5.0
        assert config.default_duration_seconds == 2.0

        Path(tmp_path).unlink()

    def test_load_missing_config_raises(self) -> None:
        with pytest.raises(FileNotFoundError):
            load_server_config("nonexistent.yaml")

    def test_create_mock_model(self) -> None:
        config = ServerConfig(model_name="mock")
        model = create_model(config)
        assert model.get_model_name() == "mock"

    def test_create_unknown_model_raises(self) -> None:
        config = ServerConfig(model_name="unknown")
        with pytest.raises(ValueError, match="Unknown model"):
            create_model(config)


@pytest.mark.unit
class TestResponseSpecCompliance:
    def test_all_bone_names_are_valid_vrm(self, grpc_channel: grpc.Channel) -> None:
        valid_vrm_names = set(SMPL_TO_VRM_MAPPING.values())
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            assert curve.bone_name in valid_vrm_names, f"Invalid bone: {curve.bone_name}"

    def test_property_types_in_valid_range(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            assert 0 <= curve.property_type <= 5

    def test_keyframes_sorted_ascending(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            times = [kf.time for kf in curve.keyframes]
            for i in range(1, len(times)):
                assert times[i] >= times[i - 1]

    def test_minimum_two_keyframes_per_curve(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            assert len(curve.keyframes) >= 2

    def test_no_nan_inf_in_keyframes(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            for kf in curve.keyframes:
                assert math.isfinite(kf.value), f"Non-finite value in {curve.bone_name}"
                assert math.isfinite(kf.time)
                assert math.isfinite(kf.tangent_in_dt)
                assert math.isfinite(kf.tangent_in_dv)
                assert math.isfinite(kf.tangent_out_dt)
                assert math.isfinite(kf.tangent_out_dv)

    def test_root_bone_has_translation_and_rotation(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        hips_props = {c.property_type for c in response.curves if c.bone_name == "hips"}
        assert 0 in hips_props
        assert 1 in hips_props
        assert 2 in hips_props
        assert 3 in hips_props
        assert 4 in hips_props
        assert 5 in hips_props

    def test_interpolation_type_valid(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(prompt="walk forward", duration_seconds=2.0, target_fps=30)
        response: MotionResponse = stub.GenerateMotion(request)

        for curve in response.curves:
            for kf in curve.keyframes:
                assert kf.interpolation in (LINEAR, BEZIER)

    def test_expected_curve_count(self, grpc_channel: grpc.Channel) -> None:
        stub = TextToMotionServiceStub(grpc_channel)
        request = MotionRequest(
            prompt="walk forward",
            duration_seconds=3.0,
            target_fps=30,
            skeleton_type=VRM_HUMANOID,
        )
        response: MotionResponse = stub.GenerateMotion(request)

        expected_curves = 22 * 3 + 3
        assert len(response.curves) == expected_curves
