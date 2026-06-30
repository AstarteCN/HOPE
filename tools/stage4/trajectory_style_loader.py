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


RAW_METRIC_KEYS = (
    "path_length_m",
    "step_count",
    "gear_shifts",
    "total_reward",
    "action_selection",
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

    case_uid = str(case["case_uid"])
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
        planner_route_active.append(_bool_value(status.get("planner_route_active", False)))

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
        outcome={key: summary[key] for key in OUTCOME_KEYS if key in summary},
        raw_metrics={key: summary[key] for key in RAW_METRIC_KEYS if key in summary},
    )


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


def _bool_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return bool(value)


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

    xs: list[float] = []
    ys: list[float] = []
    for point_index, raw_point in enumerate(points):
        point = _sequence(raw_point, f"static_geometry.target_polygon[{point_index}]")
        if len(point) < 2:
            raise ValueError(f"static_geometry.target_polygon[{point_index}] must have x and y")
        xs.append(finite_float(point[0], "target.x"))
        ys.append(finite_float(point[1], "target.y"))
    return Pose2D(sum(xs) / len(xs), sum(ys) / len(ys), final_pose.heading)
