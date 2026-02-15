from __future__ import annotations

import argparse
import logging
import time
from concurrent import futures
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import grpc
import yaml

from anim_ml.server.model_mock import MockTextToMotionModel
from anim_ml.server.motion_converter import (
    ConversionConfig,
    convert_humanml3d_to_curves,
)
from anim_ml.server.proto import (
    BEZIER,
    LINEAR,
    AnimationCurve,
    CurveKeyframe,
    MotionRequest,
    MotionResponse,
    StatusRequest,
    StatusResponse,
    TextToMotionServiceServicer,
    add_TextToMotionServiceServicer_to_server,  # pyright: ignore[reportUnknownVariableType]
)

if TYPE_CHECKING:
    from anim_ml.server.model_base import TextToMotionModel
    from anim_ml.server.motion_converter import MotionCurveData

logger = logging.getLogger(__name__)


@dataclass
class ServerConfig:
    port: int = 50051
    max_workers: int = 4
    model_name: str = "mock"
    model_device: str = "cuda"
    model_checkpoint: str = ""
    default_fps: int = 30
    max_duration_seconds: float = 10.0
    default_duration_seconds: float = 3.0


def load_server_config(config_path: str | Path) -> ServerConfig:
    path = Path(config_path)
    if not path.exists():
        msg = f"Config file not found: {path}"
        raise FileNotFoundError(msg)

    with path.open() as f:
        raw = yaml.safe_load(f)

    server_section = raw.get("server", {})
    model_section = raw.get("model", {})
    gen_section = raw.get("generation", {})

    return ServerConfig(
        port=server_section.get("port", 50051),
        max_workers=server_section.get("max_workers", 4),
        model_name=model_section.get("name", "mock"),
        model_device=model_section.get("device", "cuda"),
        model_checkpoint=model_section.get("checkpoint", ""),
        default_fps=gen_section.get("default_fps", 30),
        max_duration_seconds=gen_section.get("max_duration_seconds", 10.0),
        default_duration_seconds=gen_section.get("default_duration_seconds", 3.0),
    )


def create_model(config: ServerConfig) -> TextToMotionModel:
    if config.model_name == "mock":
        return MockTextToMotionModel()

    if config.model_name == "light_t2m":
        from anim_ml.paths import resolve_data_path
        from anim_ml.server.model_light_t2m import LightT2MModel
        return LightT2MModel(str(resolve_data_path(config.model_checkpoint)))

    msg = f"Unknown model: {config.model_name}"
    raise ValueError(msg)


class AnimationServicer(TextToMotionServiceServicer):
    def __init__(self, model: TextToMotionModel, config: ServerConfig) -> None:
        self._model = model
        self._config = config

    def GenerateMotion(
        self,
        request: MotionRequest,
        context: grpc.ServicerContext,
    ) -> MotionResponse:
        if not request.prompt:
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details("prompt must not be empty")
            return MotionResponse()

        duration = request.duration_seconds or self._config.default_duration_seconds
        duration = max(0.5, min(duration, self._config.max_duration_seconds))

        target_fps = request.target_fps or self._config.default_fps
        if target_fps not in (30, 60):
            context.set_code(grpc.StatusCode.INVALID_ARGUMENT)
            context.set_details(f"target_fps must be 30 or 60, got {target_fps}")
            return MotionResponse()

        if not self._model.is_ready():
            context.set_code(grpc.StatusCode.UNAVAILABLE)
            context.set_details("Model not loaded")
            return MotionResponse()

        bone_mappings = _parse_bone_mappings(request) or None

        try:
            return self._generate_motion(request.prompt, duration, target_fps, bone_mappings)
        except MemoryError:
            context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
            context.set_details("Out of memory")
            return MotionResponse()
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                context.set_code(grpc.StatusCode.RESOURCE_EXHAUSTED)
                context.set_details("GPU out of memory")
                return MotionResponse()
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(str(e))
            return MotionResponse()
        except Exception as e:
            logger.exception("Unexpected error in GenerateMotion")
            context.set_code(grpc.StatusCode.INTERNAL)
            context.set_details(f"Internal error: {e}")
            return MotionResponse()

    def _generate_motion(
        self,
        prompt: str,
        duration: float,
        target_fps: int,
        bone_mappings: list[tuple[int, str]] | None,
    ) -> MotionResponse:
        start_time = time.monotonic()

        result = self._model.generate(prompt, duration)

        conversion_config = ConversionConfig(
            target_fps=target_fps,
            keyframe_epsilon=0.5,
            bezier_max_error=0.01,
        )
        curves = convert_humanml3d_to_curves(
            result.motion_tensor, conversion_config, bone_mappings,
        )

        elapsed_ms = (time.monotonic() - start_time) * 1000.0

        return MotionResponse(
            curves=_build_proto_curves(curves),
            generation_time_ms=elapsed_ms,
            model_used=result.model_name,
        )

    def GetServerStatus(
        self,
        request: StatusRequest,
        context: grpc.ServicerContext,
    ) -> StatusResponse:
        return StatusResponse(
            ready=self._model.is_ready(),
            active_model=self._model.get_model_name(),
            gpu_memory_mb=self._model.get_gpu_memory_mb(),
        )


def serve(config: ServerConfig) -> grpc.Server:
    model = create_model(config)
    model.load(config.model_device)

    server = grpc.server(futures.ThreadPoolExecutor(max_workers=config.max_workers))
    servicer = AnimationServicer(model, config)
    add_TextToMotionServiceServicer_to_server(servicer, server)

    server.add_insecure_port(f"[::]:{config.port}")
    server.start()

    logger.info("Server started on port %d with model '%s'", config.port, config.model_name)
    return server


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    parser = argparse.ArgumentParser(description="Text-to-Motion gRPC server")
    parser.add_argument("--config", type=str, default="configs/server.yaml")
    parser.add_argument("--port", type=int, default=None)
    args = parser.parse_args()

    config = load_server_config(args.config)
    if args.port is not None:
        config.port = args.port

    server = serve(config)
    try:
        server.wait_for_termination()
    except KeyboardInterrupt:
        server.stop(grace=5)
        logger.info("Server stopped")


def _parse_bone_mappings(request: MotionRequest) -> list[tuple[int, str]]:
    return [
        (bm.source_joint_index, bm.target_bone_name)
        for bm in request.bone_mappings
    ]


def _build_proto_curves(curves: list[MotionCurveData]) -> list[AnimationCurve]:
    proto_curves: list[AnimationCurve] = []

    for curve in curves:
        keyframes: list[CurveKeyframe] = []
        for i in range(len(curve.times)):
            interpolation = _determine_interpolation(
                curve.tangent_in[i], curve.tangent_out[i],
            )

            keyframes.append(CurveKeyframe(
                time=float(curve.times[i]),
                value=float(curve.values[i]),
                tangent_in_dt=float(curve.tangent_in[i][0]),
                tangent_in_dv=float(curve.tangent_in[i][1]),
                tangent_out_dt=float(curve.tangent_out[i][0]),
                tangent_out_dv=float(curve.tangent_out[i][1]),
                interpolation=interpolation,  # type: ignore[arg-type]
            ))

        proto_curves.append(AnimationCurve(
            bone_name=curve.bone_name,
            property_type=curve.property_type,  # type: ignore[arg-type]
            keyframes=keyframes,
        ))

    return proto_curves


def _determine_interpolation(
    tangent_in: tuple[float, float],
    tangent_out: tuple[float, float],
) -> int:
    has_bezier_curve = abs(tangent_in[1]) > 1e-6 or abs(tangent_out[1]) > 1e-6
    return BEZIER if has_bezier_curve else LINEAR


if __name__ == "__main__":
    main()
