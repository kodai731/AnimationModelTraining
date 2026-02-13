from google.protobuf.internal import containers as _containers
from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from collections.abc import Iterable as _Iterable, Mapping as _Mapping
from typing import ClassVar as _ClassVar, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class SkeletonType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    SMPL_22: _ClassVar[SkeletonType]
    VRM_HUMANOID: _ClassVar[SkeletonType]

class PropertyType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    TRANSLATION_X: _ClassVar[PropertyType]
    TRANSLATION_Y: _ClassVar[PropertyType]
    TRANSLATION_Z: _ClassVar[PropertyType]
    ROTATION_X: _ClassVar[PropertyType]
    ROTATION_Y: _ClassVar[PropertyType]
    ROTATION_Z: _ClassVar[PropertyType]

class InterpolationType(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    LINEAR: _ClassVar[InterpolationType]
    BEZIER: _ClassVar[InterpolationType]
SMPL_22: SkeletonType
VRM_HUMANOID: SkeletonType
TRANSLATION_X: PropertyType
TRANSLATION_Y: PropertyType
TRANSLATION_Z: PropertyType
ROTATION_X: PropertyType
ROTATION_Y: PropertyType
ROTATION_Z: PropertyType
LINEAR: InterpolationType
BEZIER: InterpolationType

class MotionRequest(_message.Message):
    __slots__ = ("prompt", "duration_seconds", "target_fps", "skeleton_type", "bone_mappings")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    DURATION_SECONDS_FIELD_NUMBER: _ClassVar[int]
    TARGET_FPS_FIELD_NUMBER: _ClassVar[int]
    SKELETON_TYPE_FIELD_NUMBER: _ClassVar[int]
    BONE_MAPPINGS_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    duration_seconds: float
    target_fps: int
    skeleton_type: SkeletonType
    bone_mappings: _containers.RepeatedCompositeFieldContainer[BoneMapping]
    def __init__(self, prompt: _Optional[str] = ..., duration_seconds: _Optional[float] = ..., target_fps: _Optional[int] = ..., skeleton_type: _Optional[_Union[SkeletonType, str]] = ..., bone_mappings: _Optional[_Iterable[_Union[BoneMapping, _Mapping]]] = ...) -> None: ...

class BoneMapping(_message.Message):
    __slots__ = ("source_joint_index", "target_bone_name")
    SOURCE_JOINT_INDEX_FIELD_NUMBER: _ClassVar[int]
    TARGET_BONE_NAME_FIELD_NUMBER: _ClassVar[int]
    source_joint_index: int
    target_bone_name: str
    def __init__(self, source_joint_index: _Optional[int] = ..., target_bone_name: _Optional[str] = ...) -> None: ...

class MotionResponse(_message.Message):
    __slots__ = ("curves", "generation_time_ms", "model_used")
    CURVES_FIELD_NUMBER: _ClassVar[int]
    GENERATION_TIME_MS_FIELD_NUMBER: _ClassVar[int]
    MODEL_USED_FIELD_NUMBER: _ClassVar[int]
    curves: _containers.RepeatedCompositeFieldContainer[AnimationCurve]
    generation_time_ms: float
    model_used: str
    def __init__(self, curves: _Optional[_Iterable[_Union[AnimationCurve, _Mapping]]] = ..., generation_time_ms: _Optional[float] = ..., model_used: _Optional[str] = ...) -> None: ...

class AnimationCurve(_message.Message):
    __slots__ = ("bone_name", "property_type", "keyframes")
    BONE_NAME_FIELD_NUMBER: _ClassVar[int]
    PROPERTY_TYPE_FIELD_NUMBER: _ClassVar[int]
    KEYFRAMES_FIELD_NUMBER: _ClassVar[int]
    bone_name: str
    property_type: PropertyType
    keyframes: _containers.RepeatedCompositeFieldContainer[CurveKeyframe]
    def __init__(self, bone_name: _Optional[str] = ..., property_type: _Optional[_Union[PropertyType, str]] = ..., keyframes: _Optional[_Iterable[_Union[CurveKeyframe, _Mapping]]] = ...) -> None: ...

class CurveKeyframe(_message.Message):
    __slots__ = ("time", "value", "tangent_in_dt", "tangent_in_dv", "tangent_out_dt", "tangent_out_dv", "interpolation")
    TIME_FIELD_NUMBER: _ClassVar[int]
    VALUE_FIELD_NUMBER: _ClassVar[int]
    TANGENT_IN_DT_FIELD_NUMBER: _ClassVar[int]
    TANGENT_IN_DV_FIELD_NUMBER: _ClassVar[int]
    TANGENT_OUT_DT_FIELD_NUMBER: _ClassVar[int]
    TANGENT_OUT_DV_FIELD_NUMBER: _ClassVar[int]
    INTERPOLATION_FIELD_NUMBER: _ClassVar[int]
    time: float
    value: float
    tangent_in_dt: float
    tangent_in_dv: float
    tangent_out_dt: float
    tangent_out_dv: float
    interpolation: InterpolationType
    def __init__(self, time: _Optional[float] = ..., value: _Optional[float] = ..., tangent_in_dt: _Optional[float] = ..., tangent_in_dv: _Optional[float] = ..., tangent_out_dt: _Optional[float] = ..., tangent_out_dv: _Optional[float] = ..., interpolation: _Optional[_Union[InterpolationType, str]] = ...) -> None: ...

class StatusRequest(_message.Message):
    __slots__ = ()
    def __init__(self) -> None: ...

class StatusResponse(_message.Message):
    __slots__ = ("ready", "active_model", "gpu_memory_mb")
    READY_FIELD_NUMBER: _ClassVar[int]
    ACTIVE_MODEL_FIELD_NUMBER: _ClassVar[int]
    GPU_MEMORY_MB_FIELD_NUMBER: _ClassVar[int]
    ready: bool
    active_model: str
    gpu_memory_mb: int
    def __init__(self, ready: bool = ..., active_model: _Optional[str] = ..., gpu_memory_mb: _Optional[int] = ...) -> None: ...
