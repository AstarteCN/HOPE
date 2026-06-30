from __future__ import annotations

import json
from json import JSONDecodeError
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.stage4.trajectory_style_schema import (
    Pose2D,
    TrajectoryTrace,
    finite_float,
    pose_from_mapping,
)


NUMERIC_RAW_METRIC_KEYS = (
    "path_length_m",
    "step_count",
    "gear_shifts",
    "total_reward",
)

OUTCOME_KEYS = ("success", "terminal_status", "truncated_by_max_steps")
ACTION_KEYS = ("applied_model", "raw_model", "model", "env")


def load_trajectory_traces(path: str | Path) -> list[TrajectoryTrace]:
    trace_path = Path(path)
    if not trace_path.exists():
        raise ValueError(f"trace file does not exist: {trace_path}")

    try:
        payload = json.loads(trace_path.read_text(encoding="utf-8"))
    except JSONDecodeError as exc:
        raise ValueError(f"trace file contains invalid JSON: {trace_path}") from exc

    if not isinstance(payload, Mapping):
        raise ValueError("trace file must contain a JSON object")

    raw_traces = _select_trace_payloads(payload)
    traces: list[TrajectoryTrace] = []
    seen_case_uids: set[str] = set()
    for index, raw_trace in enumerate(raw_traces):
        trace = _normalize_trace(raw_trace, trace_path, index)
        if trace.case_uid in seen_case_uids:
            raise ValueError(f"duplicate case_uid in trace file: {trace.case_uid}")
        seen_case_uids.add(trace.case_uid)
        traces.append(trace)
    return traces


def _select_trace_payloads(payload: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    if "case_traces" in payload:
        raw_traces = payload["case_traces"]
        if not isinstance(raw_traces, Sequence) or isinstance(raw_traces, (str, bytes)):
            raise ValueError("case_traces must be a list")
        traces: list[Mapping[str, Any]] = []
        for index, raw_trace in enumerate(raw_traces):
            if not isinstance(raw_trace, Mapping):
                raise ValueError(f"case_traces[{index}] must be an object")
            traces.append(raw_trace)
        return traces

    if isinstance(payload.get("case"), Mapping) and isinstance(payload.get("frames"), Sequence):
        return [payload]

    raise ValueError("trace file must contain case_traces or a single trace with case and frames")


def _normalize_trace(raw_trace: Mapping[str, Any], source_path: Path, trace_index: int) -> TrajectoryTrace:
    case = _mapping(raw_trace.get("case"), f"trace[{trace_index}].case")
    static_geometry = _mapping(raw_trace.get("static_geometry", {}), f"trace[{trace_index}].static_geometry")
    frames = _sequence(raw_trace.get("frames"), f"trace[{trace_index}].frames")
    summary = _mapping(raw_trace.get("summary", {}), f"trace[{trace_index}].summary")

    case_uid = _case_uid(case)
    poses = [_pose_from_frame(frame, trace_index, frame_index) for frame_index, frame in enumerate(frames)]
    start_pose = poses[0] if poses else Pose2D(0.0, 0.0, 0.0)
    final_pose = poses[-1] if poses else start_pose

    actions: list[list[float]] = []
    action_sources: list[str] = []
    planner_route_active: list[bool] = []
    for frame_index, frame in enumerate(frames):
        frame_mapping = _mapping(frame, f"trace[{trace_index}].frames[{frame_index}]")
        action = _mapping(frame_mapping.get("action", {}), f"trace[{trace_index}].frames[{frame_index}].action")
        status = _mapping(frame_mapping.get("status", {}), f"trace[{trace_index}].frames[{frame_index}].status")
        actions.append(_action_values(action, trace_index, frame_index))
        action_sources.append(str(action.get("source", "unknown")))
        planner_route_active.append(
            _planner_route_active(status.get("planner_route_active", False), trace_index, frame_index)
        )

    return TrajectoryTrace(
        case_uid=case_uid,
        source_trace_path=str(source_path),
        scene_type=str(case.get("split", "unknown")),
        slot_type=str(case.get("parking_type", "unknown")).lower(),
        start_pose=start_pose,
        target_pose=_target_pose(static_geometry, final_pose),
        poses=poses,
        actions=actions,
        action_sources=action_sources,
        planner_route_active=planner_route_active,
        obstacles=_obstacles(static_geometry),
        outcome=_outcome(summary),
        raw_metrics=_raw_metrics(summary),
    )


def _case_uid(case: Mapping[str, Any]) -> str:
    value = case.get("case_uid")
    if not isinstance(value, str) or not value.strip():
        raise ValueError("case_uid must be a non-empty string")
    return value


def _mapping(value: Any, field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be an object")
    return value


def _sequence(value: Any, field_name: str) -> Sequence[Any]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValueError(f"{field_name} must be a list")
    return value


def _pose_from_frame(frame: Any, trace_index: int, frame_index: int) -> Pose2D:
    frame_mapping = _mapping(frame, f"trace[{trace_index}].frames[{frame_index}]")
    ego_state = _mapping(frame_mapping.get("ego_state"), f"trace[{trace_index}].frames[{frame_index}].ego_state")
    return pose_from_mapping(ego_state)


def _action_values(action: Mapping[str, Any], trace_index: int, frame_index: int) -> list[float]:
    for key in ACTION_KEYS:
        if key in action:
            values = _sequence(action[key], f"trace[{trace_index}].frames[{frame_index}].action.{key}")
            return [finite_float(value, f"trace[{trace_index}].frames[{frame_index}].action.{key}") for value in values]
    raise ValueError(f"trace[{trace_index}].frames[{frame_index}].action must include a model or env action")


def _planner_route_active(value: Any, trace_index: int, frame_index: int) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"trace[{trace_index}].frames[{frame_index}].status.planner_route_active must be a bool")
    return value


def _outcome(summary: Mapping[str, Any]) -> dict[str, Any]:
    outcome: dict[str, Any] = {}
    if "success" in summary:
        outcome["success"] = _summary_bool(summary["success"], "success")
    if "terminal_status" in summary:
        value = summary["terminal_status"]
        if value is not None and not isinstance(value, str):
            raise ValueError("terminal_status must be a string or None")
        outcome["terminal_status"] = value
    if "truncated_by_max_steps" in summary:
        outcome["truncated_by_max_steps"] = _summary_bool(
            summary["truncated_by_max_steps"],
            "truncated_by_max_steps",
        )
    return outcome


def _summary_bool(value: Any, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _raw_metrics(summary: Mapping[str, Any]) -> dict[str, Any]:
    metrics: dict[str, Any] = {}
    for key in NUMERIC_RAW_METRIC_KEYS:
        if key in summary:
            metrics[key] = finite_float(summary[key], key)
    if "action_selection" in summary:
        metrics["action_selection"] = summary["action_selection"]
    return metrics


def _obstacles(static_geometry: Mapping[str, Any]) -> list[list[list[float]]]:
    raw_obstacles = static_geometry.get("obstacle_polygons", [])
    obstacles: list[list[list[float]]] = []
    for polygon_index, raw_polygon in enumerate(_sequence(raw_obstacles, "static_geometry.obstacle_polygons")):
        polygon: list[list[float]] = []
        for point_index, raw_point in enumerate(_sequence(raw_polygon, f"static_geometry.obstacle_polygons[{polygon_index}]")):
            point = _sequence(raw_point, f"static_geometry.obstacle_polygons[{polygon_index}][{point_index}]")
            if len(point) < 2:
                raise ValueError(f"static_geometry.obstacle_polygons[{polygon_index}][{point_index}] must have x and y")
            polygon.append(
                [
                    finite_float(point[0], "obstacle.x"),
                    finite_float(point[1], "obstacle.y"),
                ]
            )
        obstacles.append(polygon)
    return obstacles


def _target_pose(static_geometry: Mapping[str, Any], final_pose: Pose2D) -> Pose2D:
    raw_polygon = static_geometry.get("target_polygon")
    if raw_polygon is None:
        return final_pose

    points = _sequence(raw_polygon, "static_geometry.target_polygon")
    if not points:
        return final_pose

    points_xy: list[tuple[float, float]] = []
    for point_index, raw_point in enumerate(points):
        point = _sequence(raw_point, f"static_geometry.target_polygon[{point_index}]")
        if len(point) < 2:
            raise ValueError(f"static_geometry.target_polygon[{point_index}] must have x and y")
        points_xy.append((finite_float(point[0], "target.x"), finite_float(point[1], "target.y")))

    centroid = _polygon_centroid(points_xy)
    if centroid is None:
        centroid = _average_point(_without_repeated_closure(points_xy))
    if centroid is None:
        return final_pose
    return Pose2D(centroid[0], centroid[1], final_pose.heading)


def _polygon_centroid(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if len(points) < 3:
        return None

    signed_area_twice = 0.0
    centroid_x_numerator = 0.0
    centroid_y_numerator = 0.0
    for index, (x0, y0) in enumerate(points):
        x1, y1 = points[(index + 1) % len(points)]
        cross = x0 * y1 - x1 * y0
        signed_area_twice += cross
        centroid_x_numerator += (x0 + x1) * cross
        centroid_y_numerator += (y0 + y1) * cross

    if signed_area_twice == 0.0:
        return None
    return (
        centroid_x_numerator / (3.0 * signed_area_twice),
        centroid_y_numerator / (3.0 * signed_area_twice),
    )


def _without_repeated_closure(points: Sequence[tuple[float, float]]) -> Sequence[tuple[float, float]]:
    if len(points) > 1 and points[0] == points[-1]:
        return points[:-1]
    return points


def _average_point(points: Sequence[tuple[float, float]]) -> tuple[float, float] | None:
    if not points:
        return None
    return (
        sum(point[0] for point in points) / len(points),
        sum(point[1] for point in points) / len(points),
    )
