import numpy as np
import pytest
from numpy.testing import assert_allclose

from anim_ml.utils.bezier_fitter import fit_bezier_segments
from anim_ml.utils.keyframe_reducer import KeyframePoint


def _evaluate_cubic_bezier(p0: float, p1: float, p2: float, p3: float, t: np.ndarray) -> np.ndarray:
    t1 = 1.0 - t
    return t1**3 * p0 + 3 * t1**2 * t * p1 + 3 * t1 * t**2 * p2 + t**3 * p3


@pytest.mark.unit
class TestKnownCubic:
    def test_fit_recovers_cubic(self) -> None:
        p0, p1, p2, p3 = 0.0, 2.0, -1.0, 1.0
        t_start, t_end = 0.0, 1.0

        times = np.linspace(t_start, t_end, 50)
        t_norm = (times - t_start) / (t_end - t_start)
        values = _evaluate_cubic_bezier(p0, p1, p2, p3, t_norm)

        keyframes = [KeyframePoint(t_start, p0), KeyframePoint(t_end, p3)]
        result = fit_bezier_segments(keyframes, times, values, max_error=0.01)

        assert len(result) == 2

        dv_out = result[0].tangent_out[1]
        fitted_p1 = p0 + dv_out
        assert_allclose(fitted_p1, p1, atol=0.1)

        dv_in = result[1].tangent_in[1]
        fitted_p2 = p3 + dv_in
        assert_allclose(fitted_p2, p2, atol=0.1)


@pytest.mark.unit
class TestReconstructionError:
    def test_sine_fit_error(self) -> None:
        times = np.linspace(0, 1, 100)
        values = np.sin(np.pi * times)

        keyframes = [KeyframePoint(0.0, 0.0), KeyframePoint(1.0, 0.0)]
        result = fit_bezier_segments(keyframes, times, values, max_error=0.05)

        p0 = result[0].value
        p1 = p0 + result[0].tangent_out[1]
        p2 = result[1].value + result[1].tangent_in[1]
        p3 = result[1].value

        t_norm = times
        fitted = _evaluate_cubic_bezier(p0, p1, p2, p3, t_norm)
        mean_error = float(np.mean(np.abs(fitted - values)))
        assert mean_error < 0.05


@pytest.mark.unit
class TestTangentFormat:
    def test_tangent_out_dt_positive(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.sin(times)
        keyframes = [KeyframePoint(0.0, 0.0), KeyframePoint(2.0, float(np.sin(2.0)))]
        result = fit_bezier_segments(keyframes, times, values)
        assert result[0].tangent_out[0] > 0

    def test_tangent_in_dt_negative(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.sin(times)
        keyframes = [KeyframePoint(0.0, 0.0), KeyframePoint(2.0, float(np.sin(2.0)))]
        result = fit_bezier_segments(keyframes, times, values)
        assert result[1].tangent_in[0] < 0


@pytest.mark.unit
class TestLinearSegment:
    def test_linear_has_proportional_tangents(self) -> None:
        times = np.linspace(0, 1, 30)
        values = times * 3.0
        keyframes = [KeyframePoint(0.0, 0.0), KeyframePoint(1.0, 3.0)]
        result = fit_bezier_segments(keyframes, times, values, max_error=0.01)

        dv_out = result[0].tangent_out[1]
        dv_in = result[1].tangent_in[1]

        assert dv_out == pytest.approx(1.0, abs=0.2)
        assert dv_in == pytest.approx(-1.0, abs=0.2)


@pytest.mark.unit
class TestMultipleSegments:
    def test_three_keyframes(self) -> None:
        times = np.linspace(0, 2, 60)
        values = np.sin(np.pi * times)
        keyframes = [
            KeyframePoint(0.0, 0.0),
            KeyframePoint(1.0, 0.0),
            KeyframePoint(2.0, 0.0),
        ]
        result = fit_bezier_segments(keyframes, times, values)
        assert len(result) == 3


@pytest.mark.unit
class TestEdgeCases:
    def test_single_keyframe(self) -> None:
        result = fit_bezier_segments(
            [KeyframePoint(0.0, 1.0)],
            np.array([0.0]),
            np.array([1.0]),
        )
        assert len(result) == 1
        assert result[0].tangent_in == (0.0, 0.0)
        assert result[0].tangent_out == (0.0, 0.0)

    def test_empty_keyframes(self) -> None:
        result = fit_bezier_segments([], np.array([]), np.array([]))
        assert len(result) == 0
