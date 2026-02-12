from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from anim_ml.utils.keyframe_reducer import KeyframePoint


@dataclass
class BezierKeyframe:
    time: float
    value: float
    tangent_in: tuple[float, float]
    tangent_out: tuple[float, float]


def fit_bezier_segments(
    keyframes: list[KeyframePoint],
    original_times: np.ndarray,
    original_values: np.ndarray,
    max_error: float = 0.005,
) -> list[BezierKeyframe]:
    if len(keyframes) < 2:
        return [
            BezierKeyframe(kf.time, kf.value, (0.0, 0.0), (0.0, 0.0))
            for kf in keyframes
        ]

    segment_tangents: list[tuple[tuple[float, float], tuple[float, float]]] = []

    for seg_idx in range(len(keyframes) - 1):
        kf_start = keyframes[seg_idx]
        kf_end = keyframes[seg_idx + 1]

        segment_mask = (original_times >= kf_start.time) & (original_times <= kf_end.time)
        seg_times = original_times[segment_mask]
        seg_values = original_values[segment_mask]

        tangent_out, tangent_in = _fit_single_segment(
            kf_start, kf_end, seg_times, seg_values, max_error,
        )
        segment_tangents.append((tangent_out, tangent_in))

    result: list[BezierKeyframe] = []
    for kf_idx, kf in enumerate(keyframes):
        tan_in = segment_tangents[kf_idx - 1][1] if kf_idx > 0 else (0.0, 0.0)
        tan_out = segment_tangents[kf_idx][0] if kf_idx < len(segment_tangents) else (0.0, 0.0)
        result.append(BezierKeyframe(kf.time, kf.value, tan_in, tan_out))

    return result


def _fit_single_segment(
    kf_start: KeyframePoint,
    kf_end: KeyframePoint,
    seg_times: np.ndarray,
    seg_values: np.ndarray,
    max_error: float,
) -> tuple[tuple[float, float], tuple[float, float]]:
    duration = kf_end.time - kf_start.time
    if duration <= 0 or len(seg_times) < 3:
        return _linear_tangents(duration)

    t_normalized = (seg_times - kf_start.time) / duration

    p1v, p2v = _solve_bezier_control_values(t_normalized, seg_values, kf_start.value, kf_end.value)

    error = _compute_segment_error(t_normalized, seg_values, kf_start.value, kf_end.value, p1v, p2v)
    if error > max_error and len(seg_times) > 4:
        return _linear_tangents(duration)

    dt_out = duration / 3.0
    dt_in = -duration / 3.0
    dv_out = p1v - kf_start.value
    dv_in = p2v - kf_end.value

    return (dt_out, dv_out), (dt_in, dv_in)


def _solve_bezier_control_values(
    t_normalized: np.ndarray,
    values: np.ndarray,
    v_start: float,
    v_end: float,
) -> tuple[float, float]:
    t = t_normalized
    t1 = 1.0 - t

    col0 = 3.0 * t1 * t1 * t
    col1 = 3.0 * t1 * t * t

    rhs = values - (t1 ** 3) * v_start - (t ** 3) * v_end

    a_matrix = np.column_stack([col0, col1])
    result, _, _, _ = np.linalg.lstsq(a_matrix, rhs, rcond=None)

    return float(result[0]), float(result[1])


def _compute_segment_error(
    t_normalized: np.ndarray,
    values: np.ndarray,
    v_start: float,
    v_end: float,
    p1v: float,
    p2v: float,
) -> float:
    t = t_normalized
    t1 = 1.0 - t

    b0 = (t1 ** 3) * v_start
    b1 = 3.0 * (t1 ** 2) * t * p1v
    b2 = 3.0 * t1 * (t ** 2) * p2v
    b3 = (t ** 3) * v_end
    fitted = b0 + b1 + b2 + b3

    return float(np.max(np.abs(fitted - values)))


def _linear_tangents(duration: float) -> tuple[tuple[float, float], tuple[float, float]]:
    dt = duration / 3.0
    return (dt, 0.0), (-dt, 0.0)
