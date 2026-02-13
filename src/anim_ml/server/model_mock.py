from __future__ import annotations

import hashlib

import numpy as np

from anim_ml.server.model_base import GenerationResult, TextToMotionModel

HUMANML3D_DIM = 263
SOURCE_FPS = 20


class MockTextToMotionModel(TextToMotionModel):
    def __init__(self) -> None:
        self._ready = False

    def load(self, device: str) -> None:
        self._ready = True

    def generate(self, prompt: str, duration_seconds: float) -> GenerationResult:
        if not self._ready:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        num_frames = max(1, int(duration_seconds * SOURCE_FPS))
        motion_tensor = _generate_synthetic_motion(num_frames, prompt)

        return GenerationResult(
            motion_tensor=motion_tensor,
            model_name=self.get_model_name(),
            num_frames=num_frames,
        )

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "mock"

    def get_gpu_memory_mb(self) -> int:
        return 0


def _generate_synthetic_motion(num_frames: int, prompt: str) -> np.ndarray:
    seed = int(hashlib.sha256(prompt.encode()).hexdigest()[:8], 16)
    rng = np.random.default_rng(seed)

    tensor = np.zeros((num_frames, HUMANML3D_DIM), dtype=np.float32)

    amplitude = 0.1 if "walk" in prompt.lower() else 0.05
    frequency = 2.0 if "walk" in prompt.lower() else 1.0

    t = np.linspace(0, duration_from_frames(num_frames), num_frames)

    tensor[:, 0] = amplitude * np.sin(2.0 * np.pi * frequency * t)
    tensor[:, 1] = amplitude * 0.5 * np.cos(2.0 * np.pi * frequency * t)
    tensor[:, 2] = 0.0
    tensor[:, 3] = 0.9 + 0.02 * np.sin(2.0 * np.pi * frequency * t)

    _fill_identity_rotations(tensor, rng, amplitude, frequency, t)

    tensor[:, 256:260] = rng.standard_normal((num_frames, 4)).astype(np.float32) * 0.01
    tensor[:, 260:263] = 0.0

    return tensor


def _fill_identity_rotations(
    tensor: np.ndarray,
    rng: np.random.Generator,
    amplitude: float,
    frequency: float,
    t: np.ndarray,
) -> None:
    num_frames = tensor.shape[0]
    identity_6d = np.array([1, 0, 0, 0, 1, 0], dtype=np.float32)

    for joint_idx in range(21):
        offset = 130 + joint_idx * 6
        base = np.tile(identity_6d, (num_frames, 1))
        perturbation = rng.standard_normal((num_frames, 6)).astype(np.float32) * 0.01

        motion_signal = amplitude * 0.3 * np.sin(
            2.0 * np.pi * frequency * t + joint_idx * 0.5
        )
        perturbation[:, 0] += motion_signal.astype(np.float32)

        tensor[:, offset:offset + 6] = base + perturbation

    tensor[:, 4:67] = rng.standard_normal((num_frames, 63)).astype(np.float32) * 0.1
    tensor[:, 67:130] = rng.standard_normal((num_frames, 63)).astype(np.float32) * 0.01


def duration_from_frames(num_frames: int) -> float:
    return num_frames / SOURCE_FPS
