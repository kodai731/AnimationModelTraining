from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anim_ml.data.bvh_parser import parse_bvh
from anim_ml.data.curve_extractor import (
    CONTEXT_LENGTH,
    CurveSample,
    extract_curve_samples,
)

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"


@pytest.mark.unit
class TestExtractCurveSamples:
    def test_returns_samples(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        assert len(samples) > 0

    def test_sample_types(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert isinstance(s, CurveSample)

    def test_context_shape(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.context_keyframes.shape == (CONTEXT_LENGTH, 6)
            assert s.context_keyframes.dtype == np.float32

    def test_target_shape(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.target_keyframe.shape == (6,)
            assert s.target_keyframe.dtype == np.float32

    def test_property_type_range(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        prop_types = {s.property_type for s in samples}
        for pt in prop_types:
            assert 0 <= pt <= 5

    def test_joint_category_range(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert 0 <= s.joint_category <= 6

    def test_query_time_normalized(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert 0.0 <= s.query_time <= 1.01

    def test_clip_duration_positive(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.clip_duration > 0

    def test_joint_depth_nonnegative(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.joint_depth >= 0


@pytest.mark.unit
class TestContextPadding:
    def test_early_keyframes_zero_padded(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        first_sample = samples[0]
        has_zero_rows = np.any(np.all(first_sample.context_keyframes == 0.0, axis=1))
        assert has_zero_rows or first_sample.context_keyframes.shape[0] == CONTEXT_LENGTH


@pytest.mark.unit
class TestZUpExtraction:
    def test_z_up_produces_samples(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple_zup.bvh", z_up=True)
        samples = extract_curve_samples(motion)
        assert len(samples) > 0


@pytest.mark.unit
class TestEdgeCases:
    def test_short_motion_returns_empty(self) -> None:
        from anim_ml.data.bvh_parser import MotionData
        motion = MotionData(
            joint_names=["root"],
            parent_indices=[-1],
            frame_time=0.033,
            positions=np.zeros((2, 1, 3)),
            rotations=np.zeros((2, 1, 3)),
        )
        samples = extract_curve_samples(motion)
        assert samples == []


@pytest.mark.unit
class TestPerCurveNormalization:
    def test_curve_mean_std_present(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert hasattr(s, "curve_mean")
            assert hasattr(s, "curve_std")
            assert s.curve_std > 0

    def test_normalized_values_centered(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        for s in samples:
            active_rows = s.context_keyframes[s.context_keyframes[:, 0] > 0]
            if len(active_rows) < 2:
                continue
            values = active_rows[:, 1]
            assert abs(float(np.mean(values))) < 5.0
