from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

from anim_ml.server.model_base import GenerationResult, TextToMotionModel

if TYPE_CHECKING:
    import numpy as np

HUMANML3D_DIM = 263
SOURCE_FPS = 20


class LightT2MModel(TextToMotionModel):
    def __init__(self, checkpoint_path: str) -> None:
        self._checkpoint_path = Path(checkpoint_path)
        self._ready = False
        self._model: object = None

    def load(self, device: str) -> None:
        if not self._checkpoint_path.exists():
            msg = f"Checkpoint not found: {self._checkpoint_path}"
            raise FileNotFoundError(msg)

        self._model = _load_model(self._checkpoint_path, device)
        _warmup(self._model)
        self._ready = True

    def generate(self, prompt: str, duration_seconds: float) -> GenerationResult:
        if not self._ready or self._model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        num_frames = max(1, int(duration_seconds * SOURCE_FPS))
        motion_tensor = _run_inference(self._model, prompt, num_frames)

        return GenerationResult(
            motion_tensor=motion_tensor,
            model_name=self.get_model_name(),
            num_frames=num_frames,
        )

    def is_ready(self) -> bool:
        return self._ready

    def get_model_name(self) -> str:
        return "light_t2m"

    def get_gpu_memory_mb(self) -> int:
        return 512


def _load_model(checkpoint_path: Path, device: str) -> object:
    _ = checkpoint_path, device
    raise NotImplementedError("Light-T2M model loading not yet implemented")


def _warmup(model: object) -> None:
    _ = model
    raise NotImplementedError("Light-T2M warmup not yet implemented")


def _run_inference(model: object, prompt: str, num_frames: int) -> np.ndarray:
    _ = model, prompt, num_frames
    raise NotImplementedError("Light-T2M inference not yet implemented")
