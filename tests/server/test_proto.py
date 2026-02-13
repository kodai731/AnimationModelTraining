import pytest

from anim_ml.server.proto import (
    BEZIER,
    LINEAR,
    ROTATION_X,
    ROTATION_Y,
    ROTATION_Z,
    SMPL_22,
    TRANSLATION_X,
    TRANSLATION_Y,
    TRANSLATION_Z,
    VRM_HUMANOID,
    AnimationCurve,
    BoneMapping,
    CurveKeyframe,
    MotionRequest,
    MotionResponse,
    StatusRequest,
    StatusResponse,
    TextToMotionServiceServicer,
    TextToMotionServiceStub,
    add_TextToMotionServiceServicer_to_server,
)


@pytest.mark.unit
class TestProtoImports:
    def test_message_classes_importable(self) -> None:
        assert MotionRequest is not None
        assert MotionResponse is not None
        assert AnimationCurve is not None
        assert CurveKeyframe is not None
        assert BoneMapping is not None
        assert StatusRequest is not None
        assert StatusResponse is not None

    def test_service_classes_importable(self) -> None:
        assert TextToMotionServiceStub is not None
        assert TextToMotionServiceServicer is not None
        assert add_TextToMotionServiceServicer_to_server is not None


@pytest.mark.unit
class TestProtoEnums:
    def test_skeleton_type(self) -> None:
        assert SMPL_22 == 0
        assert VRM_HUMANOID == 1

    def test_property_type(self) -> None:
        assert TRANSLATION_X == 0
        assert TRANSLATION_Y == 1
        assert TRANSLATION_Z == 2
        assert ROTATION_X == 3
        assert ROTATION_Y == 4
        assert ROTATION_Z == 5

    def test_interpolation_type(self) -> None:
        assert LINEAR == 0
        assert BEZIER == 1


@pytest.mark.unit
class TestProtoSerialization:
    def test_motion_request_roundtrip(self) -> None:
        request = MotionRequest(
            prompt="a person walking forward",
            duration_seconds=3.0,
            target_fps=30,
            skeleton_type=VRM_HUMANOID,
        )

        data = request.SerializeToString()
        restored = MotionRequest()
        restored.ParseFromString(data)

        assert restored.prompt == "a person walking forward"
        assert restored.duration_seconds == pytest.approx(3.0, abs=1e-5)
        assert restored.target_fps == 30
        assert restored.skeleton_type == VRM_HUMANOID

    def test_motion_response_roundtrip(self) -> None:
        keyframe = CurveKeyframe(
            time=0.5,
            value=45.0,
            tangent_in_dt=-0.1,
            tangent_in_dv=-5.0,
            tangent_out_dt=0.1,
            tangent_out_dv=5.0,
            interpolation=BEZIER,
        )
        curve = AnimationCurve(
            bone_name="hips",
            property_type=ROTATION_X,
            keyframes=[keyframe],
        )
        response = MotionResponse(
            curves=[curve],
            generation_time_ms=42.5,
            model_used="mock",
        )

        data = response.SerializeToString()
        restored = MotionResponse()
        restored.ParseFromString(data)

        assert len(restored.curves) == 1
        assert restored.curves[0].bone_name == "hips"
        assert restored.curves[0].property_type == ROTATION_X
        assert len(restored.curves[0].keyframes) == 1
        assert restored.curves[0].keyframes[0].time == pytest.approx(0.5, abs=1e-5)
        assert restored.generation_time_ms == pytest.approx(42.5, abs=1e-2)
        assert restored.model_used == "mock"

    def test_bone_mapping_in_request(self) -> None:
        request = MotionRequest(
            prompt="walk",
            bone_mappings=[
                BoneMapping(source_joint_index=0, target_bone_name="hips"),
                BoneMapping(source_joint_index=1, target_bone_name="leftUpperLeg"),
            ],
        )

        data = request.SerializeToString()
        restored = MotionRequest()
        restored.ParseFromString(data)

        assert len(restored.bone_mappings) == 2
        assert restored.bone_mappings[0].target_bone_name == "hips"
        assert restored.bone_mappings[1].source_joint_index == 1

    def test_status_roundtrip(self) -> None:
        response = StatusResponse(ready=True, active_model="light_t2m", gpu_memory_mb=512)

        data = response.SerializeToString()
        restored = StatusResponse()
        restored.ParseFromString(data)

        assert restored.ready is True
        assert restored.active_model == "light_t2m"
        assert restored.gpu_memory_mb == 512
