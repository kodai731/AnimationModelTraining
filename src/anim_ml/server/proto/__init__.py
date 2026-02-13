from anim_ml.server.proto.animation_ml_pb2 import (  # noqa: I001
    BEZIER as BEZIER,
    LINEAR as LINEAR,
    ROTATION_X as ROTATION_X,
    ROTATION_Y as ROTATION_Y,
    ROTATION_Z as ROTATION_Z,
    SMPL_22 as SMPL_22,
    TRANSLATION_X as TRANSLATION_X,
    TRANSLATION_Y as TRANSLATION_Y,
    TRANSLATION_Z as TRANSLATION_Z,
    VRM_HUMANOID as VRM_HUMANOID,
    AnimationCurve as AnimationCurve,
    BoneMapping as BoneMapping,
    CurveKeyframe as CurveKeyframe,
    InterpolationType as InterpolationType,
    MotionRequest as MotionRequest,
    MotionResponse as MotionResponse,
    PropertyType as PropertyType,
    SkeletonType as SkeletonType,
    StatusRequest as StatusRequest,
    StatusResponse as StatusResponse,
)
from anim_ml.server.proto.animation_ml_pb2_grpc import (
    TextToMotionServiceServicer as TextToMotionServiceServicer,
    TextToMotionServiceStub as TextToMotionServiceStub,
    add_TextToMotionServiceServicer_to_server as add_TextToMotionServiceServicer_to_server,  # pyright: ignore[reportUnknownVariableType]
)
