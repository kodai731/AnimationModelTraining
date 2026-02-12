import numpy as np
import pytest

from anim_ml.utils.keyframe_reducer import reduce_keyframes


@pytest.mark.unit
class TestLinearSignal:
    def test_constant_value(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.full(60, 5.0)
        result = reduce_keyframes(times, values, epsilon=0.01)
        assert len(result) == 2
        assert result[0].time == pytest.approx(0.0)
        assert result[-1].time == pytest.approx(2.0)

    def test_linear_ramp(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.linspace(0, 10, 60)
        result = reduce_keyframes(times, values, epsilon=0.01)
        assert len(result) == 2


@pytest.mark.unit
class TestSineWave:
    def test_sine_reduces_to_reasonable_count(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.sin(2 * np.pi * times)
        result = reduce_keyframes(times, values, epsilon=0.05)
        assert 4 <= len(result) <= 20

    def test_reconstruction_within_epsilon(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.sin(2 * np.pi * times)
        epsilon = 0.05
        keyframes = reduce_keyframes(times, values, epsilon=epsilon)

        kf_times = np.array([kf.time for kf in keyframes])
        kf_values = np.array([kf.value for kf in keyframes])
        interpolated = np.interp(times, kf_times, kf_values)

        max_error = float(np.max(np.abs(interpolated - values)))
        assert max_error <= epsilon + 1e-10


@pytest.mark.unit
class TestStepFunction:
    def test_step_preserved(self) -> None:
        times = np.array([0.0, 0.5, 0.5001, 1.0])
        values = np.array([0.0, 0.0, 1.0, 1.0])
        result = reduce_keyframes(times, values, epsilon=0.01)
        kept_times = [kf.time for kf in result]
        assert any(abs(t - 0.5001) < 0.01 for t in kept_times)


@pytest.mark.unit
class TestEpsilonSensitivity:
    def test_lower_epsilon_more_keyframes(self) -> None:
        times = np.linspace(0, 2, 120)
        values = np.sin(2 * np.pi * times) + 0.3 * np.sin(6 * np.pi * times)

        kf_loose = reduce_keyframes(times, values, epsilon=0.1)
        kf_tight = reduce_keyframes(times, values, epsilon=0.01)

        assert len(kf_tight) >= len(kf_loose)


@pytest.mark.unit
class TestFirstLastPreserved:
    def test_endpoints_always_kept(self) -> None:
        times = np.linspace(0, 3, 90)
        values = np.sin(times)
        result = reduce_keyframes(times, values, epsilon=0.01)
        assert result[0].time == pytest.approx(0.0)
        assert result[-1].time == pytest.approx(3.0)


@pytest.mark.unit
class TestEdgeCases:
    def test_empty_input(self) -> None:
        result = reduce_keyframes(np.array([]), np.array([]), epsilon=0.01)
        assert len(result) == 0

    def test_single_point(self) -> None:
        result = reduce_keyframes(np.array([1.0]), np.array([5.0]), epsilon=0.01)
        assert len(result) == 1

    def test_two_points(self) -> None:
        result = reduce_keyframes(np.array([0.0, 1.0]), np.array([0.0, 1.0]), epsilon=0.01)
        assert len(result) == 2
