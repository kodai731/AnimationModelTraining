from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anim_ml.data.bvh_parser import MotionData, parse_bvh
from anim_ml.data.curve_extractor import (
    CONTEXT_LENGTH,
    MIN_CURVE_STD,
    CurveSample,
    _generate_sliding_window_samples,
    extract_curve_samples,
)
from anim_ml.utils.bezier_fitter import BezierKeyframe

FIXTURES_DIR = Path(__file__).parent.parent / "fixtures"

DUMMY_TOPO = np.zeros(6, dtype=np.float32)
DUMMY_TOKENS = np.zeros(32, dtype=np.int64)


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

    def test_topology_features_shape(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.topology_features.shape == (6,)
            assert s.topology_features.dtype == np.float32

    def test_bone_name_tokens_shape(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        for s in samples:
            assert s.bone_name_tokens.shape == (32,)
            assert s.bone_name_tokens.dtype == np.int64

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


VALUE_BOUND = 50.0


def _make_bezier_keyframes(
    times: list[float], values: list[float],
) -> list[BezierKeyframe]:
    return [
        BezierKeyframe(t, v, (-0.1, 0.0), (0.1, 0.0))
        for t, v in zip(times, values, strict=True)
    ]


def _make_dense_timeline(
    times: list[float], values: list[float], num_points: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    dense_times = np.linspace(times[0], times[-1], num_points)
    dense_values = np.interp(dense_times, times, values)
    return dense_times, dense_values


@pytest.mark.unit
class TestNearConstantCurveSkipped:
    def test_identical_values_produce_no_samples(self) -> None:
        kf_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        kf_values = [0.5, 0.5, 0.5, 0.5, 0.5]
        keyframes = _make_bezier_keyframes(times=kf_times, values=kf_values)
        dense_t, dense_v = _make_dense_timeline(kf_times, kf_values)
        samples = _generate_sliding_window_samples(
            keyframes, property_type=0,
            topology_features=DUMMY_TOPO, bone_name_tokens=DUMMY_TOKENS,
            clip_duration=2.0, joint_depth=0, scale=1.0,
            times=dense_t, normalized_values=dense_v,
        )
        assert len(samples) == 0

    def test_tiny_variation_below_threshold_skipped(self) -> None:
        kf_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        kf_values = [0.5, 0.5001, 0.5, 0.4999, 0.5]
        keyframes = _make_bezier_keyframes(times=kf_times, values=kf_values)
        dense_t, dense_v = _make_dense_timeline(kf_times, kf_values)
        samples = _generate_sliding_window_samples(
            keyframes, property_type=0,
            topology_features=DUMMY_TOPO, bone_name_tokens=DUMMY_TOKENS,
            clip_duration=2.0, joint_depth=0, scale=1.0,
            times=dense_t, normalized_values=dense_v,
        )
        assert len(samples) == 0

    def test_sufficient_variation_produces_samples(self) -> None:
        kf_times = [0.0, 0.5, 1.0, 1.5, 2.0]
        kf_values = [0.0, 0.5, 1.0, 0.3, 0.8]
        keyframes = _make_bezier_keyframes(times=kf_times, values=kf_values)
        dense_t, dense_v = _make_dense_timeline(kf_times, kf_values)
        samples = _generate_sliding_window_samples(
            keyframes, property_type=0,
            topology_features=DUMMY_TOPO, bone_name_tokens=DUMMY_TOKENS,
            clip_duration=2.0, joint_depth=0, scale=1.0,
            times=dense_t, normalized_values=dense_v,
        )
        assert len(samples) > 0


@pytest.mark.unit
class TestNormalizedValueBound:
    def test_context_values_bounded(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        for s in samples:
            assert np.all(np.abs(s.context_keyframes[:, 1]) < VALUE_BOUND), (
                f"context value out of bound: max={np.max(np.abs(s.context_keyframes[:, 1]))}"
            )

    def test_target_values_bounded(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        for s in samples:
            assert np.all(np.abs(s.target_keyframe[1:]) < VALUE_BOUND), (
                f"target value out of bound: {s.target_keyframe}"
            )

    def test_context_tangents_bounded(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        for s in samples:
            tangent_cols = s.context_keyframes[:, 2:6]
            assert np.all(np.abs(tangent_cols) < VALUE_BOUND), (
                f"context tangent out of bound: max={np.max(np.abs(tangent_cols))}"
            )

    def test_synthetic_varying_curve_bounded(self) -> None:
        kf_times = [0.0, 0.3, 0.7, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        kf_values = [0.0, 0.2, -0.1, 0.5, 0.3, 0.8, 0.1, 0.6, 0.4, 0.9]
        keyframes = _make_bezier_keyframes(times=kf_times, values=kf_values)
        dense_t, dense_v = _make_dense_timeline(kf_times, kf_values)
        samples = _generate_sliding_window_samples(
            keyframes, property_type=3,
            topology_features=DUMMY_TOPO, bone_name_tokens=DUMMY_TOKENS,
            clip_duration=4.0, joint_depth=2, scale=90.0,
            times=dense_t, normalized_values=dense_v,
        )

        for s in samples:
            assert np.all(np.isfinite(s.context_keyframes))
            assert np.all(np.isfinite(s.target_keyframe))
            assert np.all(np.abs(s.context_keyframes) < VALUE_BOUND)
            assert np.all(np.abs(s.target_keyframe) < VALUE_BOUND)


@pytest.mark.unit
class TestConstantChannelMotion:
    def _make_motion_with_constant_position(self) -> MotionData:
        num_frames = 90
        positions = np.zeros((num_frames, 2, 3), dtype=np.float64)
        rotations = np.zeros((num_frames, 2, 3), dtype=np.float64)

        positions[:, 0, 1] = 90.0

        t = np.linspace(0, 2 * np.pi, num_frames)
        rotations[:, 1, 0] = np.sin(t) * 45.0
        rotations[:, 1, 2] = np.cos(t) * 30.0

        return MotionData(
            joint_names=["pelvis", "left_knee"],
            parent_indices=[-1, 0],
            frame_time=1.0 / 30.0,
            positions=positions,
            rotations=rotations,
        )

    def test_constant_position_channels_excluded(self) -> None:
        motion = self._make_motion_with_constant_position()
        samples = extract_curve_samples(motion)

        constant_channel_samples = [
            s for s in samples
            if s.property_type < 3
            and np.all(s.context_keyframes[:, 1] == 0.0)
        ]
        assert len(constant_channel_samples) == 0

    def test_varying_rotation_channels_included(self) -> None:
        motion = self._make_motion_with_constant_position()
        samples = extract_curve_samples(motion)

        rotation_samples = [s for s in samples if s.property_type >= 3]
        assert len(rotation_samples) > 0

    def test_all_samples_have_finite_values(self) -> None:
        motion = self._make_motion_with_constant_position()
        samples = extract_curve_samples(motion)

        for s in samples:
            assert np.all(np.isfinite(s.context_keyframes))
            assert np.all(np.isfinite(s.target_keyframe))
            assert np.all(np.abs(s.context_keyframes) < VALUE_BOUND)
            assert np.all(np.abs(s.target_keyframe) < VALUE_BOUND)


@pytest.mark.unit
class TestCurveStdThreshold:
    def test_curve_std_above_minimum(self) -> None:
        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)

        for s in samples:
            scale = 180.0 if s.property_type >= 3 else max(s.curve_std, 1.0)
            raw_std = s.curve_std / scale if scale > 0 else s.curve_std
            assert raw_std >= MIN_CURVE_STD, (
                f"curve_std too small: {raw_std} (property_type={s.property_type})"
            )

    def test_gradual_increase_just_above_threshold(self) -> None:
        kf_values = [0.0 + i * (MIN_CURVE_STD * 0.6) for i in range(10)]
        kf_times = [i * 0.5 for i in range(10)]
        keyframes = _make_bezier_keyframes(times=kf_times, values=kf_values)
        dense_t, dense_v = _make_dense_timeline(kf_times, kf_values)

        samples = _generate_sliding_window_samples(
            keyframes, property_type=0,
            topology_features=DUMMY_TOPO, bone_name_tokens=DUMMY_TOKENS,
            clip_duration=4.5, joint_depth=0, scale=1.0,
            times=dense_t, normalized_values=dense_v,
        )

        for s in samples:
            assert np.all(np.abs(s.context_keyframes) < VALUE_BOUND)
            assert np.all(np.abs(s.target_keyframe) < VALUE_BOUND)


@pytest.mark.unit
class TestLossReasonable:
    def test_initial_loss_bounded(self) -> None:
        import torch

        from anim_ml.models.curve_copilot.model import CurveCopilotConfig, CurveCopilotModel
        from anim_ml.models.curve_copilot.train import LossWeights, compute_loss

        motion = parse_bvh(FIXTURES_DIR / "simple.bvh")
        samples = extract_curve_samples(motion)
        assert len(samples) > 0

        context = torch.from_numpy(np.stack([s.context_keyframes for s in samples[:64]]))
        target = torch.from_numpy(np.stack([s.target_keyframe for s in samples[:64]]))
        prop_type = torch.tensor([s.property_type for s in samples[:64]], dtype=torch.long)
        topo_features = torch.from_numpy(np.stack([s.topology_features for s in samples[:64]]))
        bone_tokens = torch.from_numpy(np.stack([s.bone_name_tokens for s in samples[:64]]))
        query_time = torch.tensor([s.query_time for s in samples[:64]], dtype=torch.float32)

        config = CurveCopilotConfig(d_model=32, n_heads=2, d_ff=64, n_layers=2, dropout=0.0)
        model = CurveCopilotModel(config)
        model.eval()

        with torch.no_grad():
            prediction, confidence = model(
                context, prop_type, topo_features, bone_tokens, query_time,
            )

            last_context_value = context[:, -1, 1]
            target_value = target[:, 1]
            value_distance = torch.abs(target_value - last_context_value)
            confidence_targets = torch.exp(-value_distance * 5.0).unsqueeze(-1)

            loss, metrics = compute_loss(
                prediction, confidence, target, LossWeights(), confidence_targets,
            )

        assert loss.item() < 1000.0, f"Initial loss too high: {loss.item()}"
        assert metrics["loss/value"] < 1000.0
        assert metrics["loss/tangent"] < 1000.0
