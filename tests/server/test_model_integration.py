import numpy as np
import pytest

from anim_ml.server.model_base import TextToMotionModel
from anim_ml.server.model_mock import MockTextToMotionModel


@pytest.mark.unit
class TestMockModel:
    def test_load_and_ready(self) -> None:
        model = MockTextToMotionModel()
        assert not model.is_ready()
        model.load("cpu")
        assert model.is_ready()

    def test_generate_shape(self) -> None:
        model = MockTextToMotionModel()
        model.load("cpu")
        result = model.generate("a person walking", 3.0)
        assert result.motion_tensor.shape == (60, 263)
        assert result.num_frames == 60
        assert result.model_name == "mock"

    def test_generate_different_durations(self) -> None:
        model = MockTextToMotionModel()
        model.load("cpu")

        result_short = model.generate("walk", 1.0)
        result_long = model.generate("walk", 5.0)

        assert result_short.num_frames == 20
        assert result_long.num_frames == 100

    def test_generate_before_load_raises(self) -> None:
        model = MockTextToMotionModel()
        with pytest.raises(RuntimeError, match="Model not loaded"):
            model.generate("walk", 1.0)

    def test_different_prompts_different_output(self) -> None:
        model = MockTextToMotionModel()
        model.load("cpu")

        result_walk = model.generate("walk forward", 2.0)
        result_dance = model.generate("dance happily", 2.0)

        assert not np.array_equal(result_walk.motion_tensor, result_dance.motion_tensor)

    def test_same_prompt_same_output(self) -> None:
        model = MockTextToMotionModel()
        model.load("cpu")

        result1 = model.generate("walk forward", 2.0)
        result2 = model.generate("walk forward", 2.0)

        np.testing.assert_array_equal(result1.motion_tensor, result2.motion_tensor)

    def test_model_name(self) -> None:
        model = MockTextToMotionModel()
        assert model.get_model_name() == "mock"

    def test_gpu_memory(self) -> None:
        model = MockTextToMotionModel()
        assert model.get_gpu_memory_mb() == 0

    def test_rotation_6d_section_near_identity(self) -> None:
        model = MockTextToMotionModel()
        model.load("cpu")
        result = model.generate("walk", 1.0)

        rot_section = result.motion_tensor[:, 130:256]
        identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

        for frame in range(result.num_frames):
            for joint in range(21):
                offset = joint * 6
                joint_rot = rot_section[frame, offset:offset + 6]
                diff = np.linalg.norm(joint_rot - identity_6d)
                assert diff < 0.5


@pytest.mark.unit
class TestTextToMotionModelAbstract:
    def test_cannot_instantiate(self) -> None:
        with pytest.raises(TypeError):
            TextToMotionModel()  # type: ignore[abstract]
