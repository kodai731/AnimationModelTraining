from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np


@dataclass
class GenerationResult:
    motion_tensor: np.ndarray
    model_name: str
    num_frames: int


class TextToMotionModel(ABC):
    @abstractmethod
    def load(self, device: str) -> None: ...

    @abstractmethod
    def generate(self, prompt: str, duration_seconds: float) -> GenerationResult: ...

    @abstractmethod
    def is_ready(self) -> bool: ...

    @abstractmethod
    def get_model_name(self) -> str: ...

    @abstractmethod
    def get_gpu_memory_mb(self) -> int: ...
