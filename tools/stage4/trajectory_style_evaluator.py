from __future__ import annotations

import math
from collections import Counter
from typing import Any, Iterable, Sequence

from tools.stage4.trajectory_style_metrics import (
    count_cusps,
    count_gear_shifts,
    count_low_speed_chatter,
    curvature_stats,
    fourier_descriptor_distance,
    hausdorff_distance,
    mean_l2_distance,
    min_obstacle_clearance,
    steering_sign_changes,
)
from tools.stage4.trajectory_style_schema import (
    ReferenceFamily,
    SceneClassification,
    StyleCaseReport,
    TrajectoryTrace,
    finite_float,
    poses_to_xy,
)


def evaluate_trace_style(
    trace: TrajectoryTrace,
    classification: SceneClassification,
    references: Sequence[ReferenceFamily],
) -> StyleCaseReport:
    if classification.scene_class == "unsupported" or not references:
        return _unsupported_report(trace, classification)

    path_points = poses_to_xy(trace.poses)
    speeds = _series_from_actions_or_poses(trace.actions, trace.poses, action_index=1, pose_attr="speed")
    steering = _series_from_actions_or_poses(trace.actions, trace.poses, action_index=0, pose_attr="steering")
    best_reference, shape_metrics = _best_reference(path_points, references)

    behavior_metrics = _behavior_metrics(path_points, speeds, steering)
    clearance_metrics = _clearance_metrics(path_points, trace.obstacles, best_reference)
    segment_metrics = _segment_metrics(trace)
    score = _score(shape_metrics, behavior_metrics, best_reference)
    diagnosis = _diagnosis(best_reference, behavior_metrics, clearance_metrics)
    style_label = _style_label(score, clearance_metrics)

    return StyleCaseReport(
        case_uid=trace.case_uid,
        scene_class=classification.scene_class,
        classification_reason=classification.reason,
        selected_reference_family=best_reference.family_id,
        shape_metrics=shape_metrics,
        behavior_metrics=behavior_metrics,
        clearance_metrics=clearance_metrics,
        segment_metrics=segment_metrics,
        style_label=style_label,
        style_score=score,
        diagnosis=diagnosis,
    )


def _unsupported_report(trace: TrajectoryTrace, classification: SceneClassification) -> StyleCaseReport:
    reason = _normalize_unsupported_reason(classification)
    return StyleCaseReport(
        case_uid=trace.case_uid,
        scene_class="unsupported",
        classification_reason=classification.reason,
        selected_reference_family=None,
        shape_metrics={},
        behavior_metrics={},
        clearance_metrics={},
        segment_metrics=_segment_metrics(trace),
        style_label="unsupported",
        style_score=0.0,
        diagnosis=["route_family_mismatch"],
        unsupported_reason=reason,
    )


def _normalize_unsupported_reason(classification: SceneClassification) -> str:
    reason = classification.reason
    prefix = "unsupported slot type:"
    if reason.startswith(prefix):
        slot_type = reason[len(prefix) :].strip()
        return f"unsupported parking_type={slot_type}"
    return reason


def _best_reference(
    path_points: Sequence[Sequence[float]],
    references: Sequence[ReferenceFamily],
) -> tuple[ReferenceFamily, dict[str, Any]]:
    candidates: list[tuple[float, ReferenceFamily, dict[str, Any]]] = []
    for reference in references:
        reference_points = poses_to_xy(reference.waypoints)
        l2 = mean_l2_distance(path_points, reference_points)
        hausdorff = hausdorff_distance(path_points, reference_points)
        fourier = fourier_descriptor_distance(path_points, reference_points)
        score = _shape_score(l2, hausdorff, fourier)
        candidates.append(
            (
                score,
                reference,
                {
                    "l2": l2,
                    "hausdorff": hausdorff,
                    "fourier": fourier,
                    "shape_score": score,
                    "reference_family": reference.family_id,
                    "route_family": reference.route_family,
                },
            )
        )

    _, reference, metrics = max(candidates, key=lambda item: item[0])
    return reference, metrics


def _shape_score(l2: float, hausdorff: float, fourier: float) -> float:
    distance = finite_float(l2, "l2") + 0.5 * finite_float(hausdorff, "hausdorff") + finite_float(fourier, "fourier")
    return 1.0 / (1.0 + distance)


def _score(
    shape_metrics: dict[str, Any],
    behavior_metrics: dict[str, Any],
    reference: ReferenceFamily,
) -> float:
    score = finite_float(shape_metrics["shape_score"], "shape_score")
    extra_cusps = _excess_over_range(behavior_metrics["cusp_count"], reference.expected_cusp_count)
    extra_shifts = _excess_over_range(behavior_metrics["gear_shifts"], reference.expected_gear_shift_count)
    chatter = int(behavior_metrics["low_speed_chatter"])
    penalty = 0.08 * extra_cusps + 0.04 * extra_shifts + 0.05 * chatter
    return max(0.0, min(1.0, score - penalty))


def _behavior_metrics(
    path_points: Sequence[Sequence[float]],
    speeds: Sequence[float],
    steering: Sequence[float],
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "gear_shifts": count_gear_shifts(speeds),
        "cusp_count": count_cusps(speeds),
        "low_speed_chatter": count_low_speed_chatter(speeds),
        "steering_sign_changes": steering_sign_changes(steering),
    }
    metrics.update(curvature_stats(path_points))
    return metrics


def _clearance_metrics(
    path_points: Sequence[Sequence[float]],
    obstacles: Iterable[Iterable[Iterable[float]]],
    reference: ReferenceFamily,
) -> dict[str, Any]:
    clearance = min_obstacle_clearance(path_points, obstacles)
    return {
        "min_clearance_m": clearance,
        "preferred_side": reference.clearance_preferences.preferred_side,
    }


def _segment_metrics(trace: TrajectoryTrace) -> dict[str, Any]:
    source_counts = Counter(str(source).upper() for source in trace.action_sources)
    planner_active = sum(1 for active in trace.planner_route_active if active)
    return {
        "rl_count": source_counts.get("RL", 0),
        "rs_count": source_counts.get("RS", 0),
        "planner_active_count": planner_active,
        "total_count": len(trace.poses),
    }


def _diagnosis(
    reference: ReferenceFamily,
    behavior_metrics: dict[str, Any],
    clearance_metrics: dict[str, Any],
) -> list[str]:
    diagnosis: list[str] = []
    if behavior_metrics["low_speed_chatter"] > 0:
        diagnosis.append("excessive_chatter")
    if _excess_over_range(behavior_metrics["cusp_count"], reference.expected_cusp_count) > 0:
        diagnosis.append("unplanned_extra_cusp")
    clearance = clearance_metrics["min_clearance_m"]
    if clearance is not None and clearance < 0.15:
        diagnosis.append("object_side_clearance_too_close")
    return diagnosis


def _style_label(score: float, clearance_metrics: dict[str, Any]) -> str:
    clearance = clearance_metrics["min_clearance_m"]
    if clearance is not None and clearance < 0.15:
        return "unsafe_or_too_close"
    if score >= 0.8:
        return "human_like"
    if score >= 0.55:
        return "acceptable_variant"
    return "style_mismatch"


def _series_from_actions_or_poses(
    actions: Sequence[Sequence[float]],
    poses: Sequence[Any],
    action_index: int,
    pose_attr: str,
) -> list[float]:
    values: list[float] = []
    for index, pose in enumerate(poses):
        value = None
        if index < len(actions) and len(actions[index]) > action_index:
            value = actions[index][action_index]
        if value is None:
            value = getattr(pose, pose_attr)
        if value is None:
            value = 0.0
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{pose_attr}[{index}] must be finite")
        values.append(value)
    return values


def _excess_over_range(value: int, expected_range: tuple[int, int]) -> int:
    return max(0, int(value) - int(expected_range[1]))
