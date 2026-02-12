from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class KeyframePoint:
    time: float
    value: float


def reduce_keyframes(
    times: np.ndarray,
    values: np.ndarray,
    epsilon: float = 0.01,
) -> list[KeyframePoint]:
    if len(times) <= 2:
        return [KeyframePoint(float(times[i]), float(values[i])) for i in range(len(times))]

    keep_mask = np.zeros(len(times), dtype=bool)
    keep_mask[0] = True
    keep_mask[-1] = True

    _rdp_mark(times, values, 0, len(times) - 1, epsilon, keep_mask)

    indices = np.where(keep_mask)[0]
    return [KeyframePoint(float(times[i]), float(values[i])) for i in indices]


def _rdp_mark(
    times: np.ndarray,
    values: np.ndarray,
    start: int,
    end: int,
    epsilon: float,
    keep_mask: np.ndarray,
) -> None:
    if end - start <= 1:
        return

    max_dist = 0.0
    max_index = start

    t_start, t_end = times[start], times[end]
    v_start, v_end = values[start], values[end]
    dt = t_end - t_start

    for i in range(start + 1, end):
        if dt == 0.0:
            dist = abs(float(values[i] - v_start))
        else:
            t_ratio = (times[i] - t_start) / dt
            interpolated = v_start + t_ratio * (v_end - v_start)
            dist = abs(float(values[i] - interpolated))

        if dist > max_dist:
            max_dist = dist
            max_index = i

    if max_dist > epsilon:
        keep_mask[max_index] = True
        _rdp_mark(times, values, start, max_index, epsilon, keep_mask)
        _rdp_mark(times, values, max_index, end, epsilon, keep_mask)
