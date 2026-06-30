from __future__ import annotations

import math
from typing import Any, Mapping, Optional, Sequence

from tools.stage4.trajectory_style_schema import (
    ClearancePreferences,
    Pose2D,
    ReferenceFamily,
    SceneClassification,
    SUPPORTED_SCENE_CLASSES,
    TrajectoryTrace,
)


SUPPORTED_REFERENCE_CLASSES = frozenset(
    {
        "perpendicular_enough_space",
        "perpendicular_limited_space",
        "parallel_standard",
        "parallel_tight",
    }
)


def classify_scene(
    trace: TrajectoryTrace,
    label_overrides: Optional[Mapping[str, Any]] = None,
) -> SceneClassification:
    """Classify a trace into the coarse offline style-reference scene classes."""
    override = _lookup_override(trace.case_uid, label_overrides)
    if override is not None:
        return SceneClassification(
            override,
            f"manual override for {trace.case_uid}",
            1.0,
            case_uid=trace.case_uid,
            slot_type=trace.slot_type,
            attributes={"approach_distance_m": _approach_distance(trace)},
        )

    approach_distance = _approach_distance(trace)
    slot_type = trace.slot_type.lower()
    attributes = {
        "approach_distance_m": approach_distance,
        "obstacle_count": len(trace.obstacles),
    }

    if slot_type == "perpendicular":
        if approach_distance >= 5.5 and not trace.obstacles:
            return SceneClassification(
                "perpendicular_enough_space",
                "perpendicular slot with at least 5.5 m approach distance and no obstacles",
                0.85,
                case_uid=trace.case_uid,
                slot_type=trace.slot_type,
                attributes=attributes,
            )
        return SceneClassification(
            "perpendicular_limited_space",
            "perpendicular slot with limited approach distance or nearby obstacles",
            0.8,
            case_uid=trace.case_uid,
            slot_type=trace.slot_type,
            attributes=attributes,
        )

    if slot_type == "parallel":
        if approach_distance >= 6.5:
            return SceneClassification(
                "parallel_standard",
                "parallel slot with at least 6.5 m approach distance",
                0.85,
                case_uid=trace.case_uid,
                slot_type=trace.slot_type,
                attributes=attributes,
            )
        return SceneClassification(
            "parallel_tight",
            "parallel slot with less than 6.5 m approach distance",
            0.8,
            case_uid=trace.case_uid,
            slot_type=trace.slot_type,
            attributes=attributes,
        )

    return SceneClassification(
        "unsupported",
        f"unsupported slot type: {trace.slot_type}",
        1.0,
        case_uid=trace.case_uid,
        slot_type=trace.slot_type,
        attributes=attributes,
    )


def generate_reference_families(
    trace: TrajectoryTrace,
    classification: SceneClassification,
) -> list[ReferenceFamily]:
    if classification.scene_class not in SUPPORTED_REFERENCE_CLASSES:
        return []

    spec = _REFERENCE_SPECS[classification.scene_class]
    waypoints = _generate_waypoints(trace, spec["lateral_offsets"], spec["heading_offsets"])

    return [
        ReferenceFamily(
            family_id=f"{classification.scene_class}:{spec['route_family']}",
            scene_class=classification.scene_class,
            route_family=spec["route_family"],
            waypoints=waypoints,
            corridor=_corridor(trace, classification.scene_class),
            phase_labels=spec["phase_labels"],
            expected_cusp_count=spec["expected_cusp_count"],
            expected_gear_shift_count=spec["expected_gear_shift_count"],
            slot_mouth_pose_window=_slot_mouth_pose_window(trace),
            clearance_preferences=spec["clearance_preferences"],
            generation_method="analytic_p0_scene_reference",
        )
    ]


def _lookup_override(case_uid: str, label_overrides: Optional[Mapping[str, Any]]) -> Optional[str]:
    if not label_overrides or case_uid not in label_overrides:
        return None

    override = label_overrides[case_uid]
    if isinstance(override, str):
        return _validate_override_scene_class(case_uid, override)
    if isinstance(override, Mapping):
        scene_class = override.get("scene_class")
        if isinstance(scene_class, str):
            return _validate_override_scene_class(case_uid, scene_class)
    raise TypeError(f"label override for {case_uid} must be a scene class string or mapping")


def _validate_override_scene_class(case_uid: str, scene_class: str) -> str:
    if scene_class not in SUPPORTED_SCENE_CLASSES:
        raise ValueError(
            f"invalid scene class override {scene_class!r} for case {case_uid!r}; "
            f"expected one of {sorted(SUPPORTED_SCENE_CLASSES)}"
        )
    return scene_class


def _approach_distance(trace: TrajectoryTrace) -> float:
    dx = trace.target_pose.x - trace.start_pose.x
    dy = trace.target_pose.y - trace.start_pose.y
    return math.hypot(dx, dy)


def _generate_waypoints(
    trace: TrajectoryTrace,
    lateral_offsets: Sequence[float],
    heading_offsets: Sequence[float],
) -> list[Pose2D]:
    start = trace.start_pose
    target = trace.target_pose
    dx = target.x - start.x
    dy = target.y - start.y
    length = math.hypot(dx, dy)
    if length > 1e-9:
        normal_x = -dy / length
        normal_y = dx / length
    else:
        normal_x = -math.sin(target.heading)
        normal_y = math.cos(target.heading)

    count = len(lateral_offsets)
    waypoints: list[Pose2D] = []
    for index, lateral in enumerate(lateral_offsets):
        ratio = index / (count - 1) if count > 1 else 0.0
        heading_offset = heading_offsets[index]
        waypoints.append(
            Pose2D(
                x=start.x + dx * ratio + normal_x * lateral,
                y=start.y + dy * ratio + normal_y * lateral,
                heading=start.heading + (target.heading - start.heading) * ratio + heading_offset,
            )
        )
    return waypoints


def _corridor(trace: TrajectoryTrace, scene_class: str) -> dict[str, Any]:
    half_width = 1.2 if scene_class in {"parallel_tight", "perpendicular_limited_space"} else 1.6
    return {
        "type": "centerline_tube",
        "half_width_m": half_width,
        "approach_distance_m": _approach_distance(trace),
        "obstacle_count": len(trace.obstacles),
    }


def _slot_mouth_pose_window(trace: TrajectoryTrace) -> dict[str, Any]:
    return {
        "center": trace.target_pose.to_dict(),
        "position_tolerance_m": 0.75,
        "heading_tolerance_rad": 0.35,
    }


_REFERENCE_SPECS: dict[str, dict[str, Any]] = {
    "perpendicular_enough_space": {
        "route_family": "one_shot_reverse_sweep",
        "phase_labels": ["approach", "reverse_entry", "late_straighten"],
        "expected_cusp_count": (0, 0),
        "expected_gear_shift_count": (0, 1),
        "clearance_preferences": ClearancePreferences(
            min_clearance_m=0.35,
            preferred_side="centered",
            target_clearance_m=0.6,
        ),
        "lateral_offsets": [0.0, -0.35, -0.55, -0.25, 0.0],
        "heading_offsets": [0.0, -0.12, -0.18, -0.08, 0.0],
    },
    "perpendicular_limited_space": {
        "route_family": "planned_cusp_reverse",
        "phase_labels": ["approach", "reverse_entry", "planned_correction", "straighten"],
        "expected_cusp_count": (1, 2),
        "expected_gear_shift_count": (1, 3),
        "clearance_preferences": ClearancePreferences(
            min_clearance_m=0.25,
            preferred_side="open_side",
            target_clearance_m=0.45,
        ),
        "lateral_offsets": [0.0, -0.45, -0.7, 0.25, 0.0],
        "heading_offsets": [0.0, -0.18, -0.24, 0.14, 0.0],
    },
    "parallel_standard": {
        "route_family": "reverse_s_curve",
        "phase_labels": ["approach", "reverse_entry", "counter_steer", "align"],
        "expected_cusp_count": (0, 1),
        "expected_gear_shift_count": (1, 2),
        "clearance_preferences": ClearancePreferences(
            min_clearance_m=0.3,
            preferred_side="curb_side",
            target_clearance_m=0.5,
        ),
        "lateral_offsets": [0.0, -0.45, -0.75, -0.3, 0.0],
        "heading_offsets": [0.0, -0.1, -0.18, 0.08, 0.0],
    },
    "parallel_tight": {
        "route_family": "reverse_s_curve_with_forward_correction",
        "phase_labels": [
            "approach",
            "reverse_entry",
            "counter_steer",
            "planned_forward_correction",
            "align",
        ],
        "expected_cusp_count": (1, 2),
        "expected_gear_shift_count": (2, 4),
        "clearance_preferences": ClearancePreferences(
            min_clearance_m=0.2,
            preferred_side="curb_side",
            target_clearance_m=0.4,
        ),
        "lateral_offsets": [0.0, -0.55, -0.85, 0.35, -0.2, 0.0],
        "heading_offsets": [0.0, -0.14, -0.22, 0.16, 0.06, 0.0],
    },
}
