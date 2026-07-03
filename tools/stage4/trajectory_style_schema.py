from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Mapping, Optional, Sequence


SUPPORTED_SCENE_CLASSES = frozenset(
    {
        "perpendicular_enough_space",
        "perpendicular_limited_space",
        "parallel_standard",
        "parallel_tight",
        "unsupported",
    }
)

STYLE_LABELS = frozenset(
    {
        "human_like",
        "acceptable_variant",
        "style_mismatch",
        "unsafe_or_too_close",
        "unsupported",
    }
)


def finite_float(value: Any, field_name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"{field_name} must be a finite number")

    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{field_name} must be a finite number")
    return result


def _json_safe(value: Any, field_name: str = "value") -> Any:
    if hasattr(value, "to_dict"):
        return _json_safe(value.to_dict(), field_name)
    if isinstance(value, Mapping):
        return {
            str(key): _json_safe(value[key], f"{field_name}.{key}")
            for key in sorted(value, key=str)
        }
    if isinstance(value, (tuple, list)):
        return [_json_safe(item, field_name) for item in value]
    if isinstance(value, (set, frozenset)):
        return [_json_safe(item, field_name) for item in sorted(value, key=repr)]
    if isinstance(value, os.PathLike):
        return os.fspath(value)
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return finite_float(value, field_name)
    if isinstance(value, Real) and not isinstance(value, bool):
        return finite_float(value, field_name)
    raise TypeError(f"{field_name} contains unsupported non JSON-safe value: {type(value).__name__}")


def _validate_known(value: str, known_values: frozenset[str], field_name: str) -> None:
    if value not in known_values:
        raise ValueError(f"{field_name} must be one of {sorted(known_values)}")


def _string_sequence(values: Sequence[Any], field_name: str) -> list[str]:
    result: list[str] = []
    for index, value in enumerate(values):
        if not isinstance(value, str):
            raise TypeError(f"{field_name}[{index}] must be a string")
        result.append(value)
    return result


def _bool_sequence(values: Sequence[Any], field_name: str) -> list[bool]:
    result: list[bool] = []
    for index, value in enumerate(values):
        if not isinstance(value, bool):
            raise TypeError(f"{field_name}[{index}] must be a bool")
        result.append(value)
    return result


@dataclass(frozen=True)
class Pose2D:
    x: float
    y: float
    heading: float
    speed: Optional[float] = None
    steering: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "x": finite_float(self.x, "x"),
            "y": finite_float(self.y, "y"),
            "heading": finite_float(self.heading, "heading"),
        }
        if self.speed is not None:
            payload["speed"] = finite_float(self.speed, "speed")
        if self.steering is not None:
            payload["steering"] = finite_float(self.steering, "steering")
        return payload


@dataclass(frozen=True)
class ClearancePreferences:
    min_clearance_m: float
    preferred_side: str
    target_clearance_m: Optional[float] = None

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "min_clearance_m": finite_float(self.min_clearance_m, "min_clearance_m"),
            "preferred_side": self.preferred_side,
        }
        if self.target_clearance_m is not None:
            payload["target_clearance_m"] = finite_float(self.target_clearance_m, "target_clearance_m")
        return payload


@dataclass(frozen=True)
class TrajectoryTrace:
    case_uid: str
    source_trace_path: str
    scene_type: str
    slot_type: str
    start_pose: Pose2D
    target_pose: Pose2D
    poses: Sequence[Pose2D]
    actions: Sequence[Sequence[float]]
    action_sources: Sequence[str]
    planner_route_active: Sequence[bool]
    obstacles: Sequence[Sequence[Sequence[float]]] = field(default_factory=list)
    outcome: Mapping[str, Any] = field(default_factory=dict)
    raw_metrics: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "case_uid": self.case_uid,
            "source_trace_path": self.source_trace_path,
            "scene_type": self.scene_type,
            "slot_type": self.slot_type,
            "start_pose": self.start_pose.to_dict(),
            "target_pose": self.target_pose.to_dict(),
            "poses": [pose.to_dict() for pose in self.poses],
            "actions": [[finite_float(value, "action") for value in action] for action in self.actions],
            "action_sources": _string_sequence(self.action_sources, "action_sources"),
            "planner_route_active": _bool_sequence(self.planner_route_active, "planner_route_active"),
            "obstacles": _json_safe(list(self.obstacles)),
            "outcome": _json_safe(self.outcome),
            "raw_metrics": _json_safe(self.raw_metrics),
        }


@dataclass(frozen=True)
class ReferenceFamily:
    family_id: str
    scene_class: str
    route_family: str
    waypoints: Sequence[Pose2D]
    corridor: Mapping[str, Any]
    phase_labels: Sequence[str]
    expected_cusp_count: tuple[int, int]
    expected_gear_shift_count: tuple[int, int]
    slot_mouth_pose_window: Mapping[str, Any]
    clearance_preferences: ClearancePreferences
    generation_method: str

    def to_dict(self) -> dict[str, Any]:
        _validate_known(self.scene_class, SUPPORTED_SCENE_CLASSES, "scene_class")
        return {
            "family_id": self.family_id,
            "scene_class": self.scene_class,
            "route_family": self.route_family,
            "waypoints": [pose.to_dict() for pose in self.waypoints],
            "corridor": _json_safe(self.corridor),
            "phase_labels": _string_sequence(self.phase_labels, "phase_labels"),
            "expected_cusp_count": list(self.expected_cusp_count),
            "expected_gear_shift_count": list(self.expected_gear_shift_count),
            "slot_mouth_pose_window": _json_safe(self.slot_mouth_pose_window),
            "clearance_preferences": self.clearance_preferences.to_dict(),
            "generation_method": self.generation_method,
        }


@dataclass(frozen=True)
class SceneClassification:
    scene_class: str
    reason: str
    confidence: float
    case_uid: Optional[str] = None
    slot_type: Optional[str] = None
    attributes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        _validate_known(self.scene_class, SUPPORTED_SCENE_CLASSES, "scene_class")
        payload: dict[str, Any] = {
            "scene_class": self.scene_class,
            "reason": self.reason,
            "confidence": finite_float(self.confidence, "confidence"),
            "attributes": _json_safe(self.attributes, "attributes"),
        }
        if self.case_uid is not None:
            payload["case_uid"] = self.case_uid
        if self.slot_type is not None:
            payload["slot_type"] = self.slot_type
        return payload


@dataclass(frozen=True)
class StyleCaseReport:
    case_uid: str
    scene_class: str
    classification_reason: str
    selected_reference_family: Optional[str]
    shape_metrics: Mapping[str, Any]
    behavior_metrics: Mapping[str, Any]
    clearance_metrics: Mapping[str, Any]
    segment_metrics: Mapping[str, Any]
    style_label: str
    style_score: float
    diagnosis: Sequence[str]
    unsupported_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        _validate_known(self.style_label, STYLE_LABELS, "style_label")
        _validate_known(self.scene_class, SUPPORTED_SCENE_CLASSES, "scene_class")
        return {
            "case_uid": self.case_uid,
            "scene_class": self.scene_class,
            "classification_reason": self.classification_reason,
            "selected_reference_family": self.selected_reference_family,
            "shape_metrics": _json_safe(self.shape_metrics),
            "behavior_metrics": _json_safe(self.behavior_metrics),
            "clearance_metrics": _json_safe(self.clearance_metrics),
            "segment_metrics": _json_safe(self.segment_metrics),
            "style_label": self.style_label,
            "style_score": finite_float(self.style_score, "style_score"),
            "diagnosis": _string_sequence(self.diagnosis, "diagnosis"),
            "unsupported_reason": self.unsupported_reason,
        }


def pose_from_mapping(mapping: Mapping[str, Any]) -> Pose2D:
    pose_kwargs: dict[str, Any] = {
        "x": finite_float(mapping["x"], "x"),
        "y": finite_float(mapping["y"], "y"),
        "heading": finite_float(mapping["heading"], "heading"),
    }
    if "speed" in mapping and mapping["speed"] is not None:
        pose_kwargs["speed"] = finite_float(mapping["speed"], "speed")
    if "steering" in mapping and mapping["steering"] is not None:
        pose_kwargs["steering"] = finite_float(mapping["steering"], "steering")
    return Pose2D(**pose_kwargs)


def poses_to_xy(points: Sequence[Pose2D]) -> list[list[float]]:
    return [[finite_float(point.x, "x"), finite_float(point.y, "y")] for point in points]
