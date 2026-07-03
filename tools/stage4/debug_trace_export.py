from __future__ import annotations

import argparse
import json
import os
import random
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.stage4.debug_trace_schema import (
    SCHEMA_VERSION,
    default_layer_groups,
    densify_polyline,
    ogm_default_grid,
    project_lidar_hits,
    sparsify_ogm,
    vehicle_body_corners_local,
    vehicle_params_from_configs,
)
from tools.stage4.stage4_target_transform import (
    TARGET_MODE_AUTO,
    TARGET_MODE_LEGACY_COSCOS,
    TARGET_MODE_REQUEST_CHOICES,
    resolve_checkpoint_target_mode,
    validate_target_mode,
)
from tools.stage4.eval_stage4_ogm_checkpoint import (
    MANEUVER_STABILITY_MODE_AUTO,
    MANEUVER_STABILITY_MODE_REQUEST_CHOICES,
    get_eval_action_with_maneuver_stability,
    resolve_checkpoint_maneuver_stability_config,
)
from tools.stage4.stage4_maneuver_stability import (
    APPLY_TO_MODES,
    MODE_OFF,
    ManeuverStabilityConfig,
    ManeuverStabilityResult,
    ManeuverStabilityTracker,
)


def _now_iso() -> str:
    return datetime.now().astimezone().replace(microsecond=0).isoformat()


def _git_commit() -> str | None:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=REPO_ROOT,
            check=False,
            text=True,
            capture_output=True,
        )
    except OSError:
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


def _json_default(value: Any) -> Any:
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    raise TypeError(f"Object of type {value.__class__.__name__} is not JSON serializable")


def build_action_trace_payload(
    *,
    source: str,
    raw_action: Any,
    applied_action: Any,
    env_action: Any,
    action_result: ManeuverStabilityResult | None = None,
) -> dict[str, Any]:
    return {
        "source": str(source),
        "raw_model": [float(value) for value in np.asarray(raw_action, dtype=float).reshape(-1).tolist()],
        "applied_model": [float(value) for value in np.asarray(applied_action, dtype=float).reshape(-1).tolist()],
        "env": [float(value) for value in np.asarray(env_action, dtype=float).reshape(-1).tolist()],
        "intervention_reason": action_result.reason if action_result is not None else "off",
        "intervention_changed": bool(action_result.changed) if action_result is not None else False,
    }


def build_trace(
    source: Mapping[str, Any],
    case: Mapping[str, Any],
    static_geometry: Mapping[str, Any],
    frames: Sequence[Mapping[str, Any]],
    summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    vehicle = vehicle_params_from_configs()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": dict(source),
        "vehicle": {
            **vehicle,
            "body_corners_local": vehicle_body_corners_local(vehicle),
        },
        "grid": ogm_default_grid(),
        "layer_groups": default_layer_groups(),
        "case": dict(case),
        "static_geometry": dict(static_geometry),
        "frames": list(frames),
        "summary": dict(summary or {}),
    }


def build_trace_collection(source: Mapping[str, Any], traces: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vehicle = vehicle_params_from_configs()
    return {
        "schema_version": SCHEMA_VERSION,
        "source": dict(source),
        "vehicle": {
            **vehicle,
            "body_corners_local": vehicle_body_corners_local(vehicle),
        },
        "grid": ogm_default_grid(),
        "layer_groups": default_layer_groups(),
        "case_count": len(traces),
        "case_traces": list(traces),
    }


def write_trace_json(trace: Mapping[str, Any], output_path: str | Path) -> None:
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(trace, indent=2, ensure_ascii=False, allow_nan=False, default=_json_default) + "\n",
        encoding="utf-8",
    )


def select_cases_by_uid(cases: Sequence[Mapping[str, Any]], case_uids: Sequence[str]) -> list[Mapping[str, Any]]:
    by_uid = {str(case["case_uid"]): case for case in cases}
    selected: list[Mapping[str, Any]] = []
    for case_uid in case_uids:
        if case_uid not in by_uid:
            raise ValueError(f"Requested case_uid is missing from cases: {case_uid}")
        selected.append(by_uid[case_uid])
    return selected


def _fixture_frame(step: int, target_mode: str = TARGET_MODE_LEGACY_COSCOS) -> dict[str, Any]:
    x = step * 0.4
    rl_path = [[idx * 0.4, 0.1 * idx, 0.02 * idx] for idx in range(step + 1)]
    rs_path = [[idx * 0.4, -0.15 * idx, -0.01 * idx] for idx in range(step + 1)]
    target = [5.0, 1.0, 0.0, 1.0, 1.0]
    if validate_target_mode(target_mode) != TARGET_MODE_LEGACY_COSCOS:
        target = [5.0, 1.0, 0.0, 1.0, 0.0]
    return {
        "step_index": step,
        "sim_time_s": step * 0.5,
        "ego_state": {"x": x, "y": 0.1 * step, "heading": 0.02 * step, "speed": 0.5, "steering": 0.1},
        "drive_direction": "forward",
        "action": build_action_trace_payload(
            source="RL",
            raw_action=[0.1, 0.2],
            applied_action=[0.1, 0.2],
            env_action=[0.075, 0.5],
        ),
        "action_mask": {"valid_count": 22, "raw": [1] * 22},
        "target": target,
        "ogm": {
            "size": 64,
            "sparse_cells": [
                {"channel": "obstacle", "row": 20, "col": 25, "value": 1.0},
                {"channel": "target", "row": 32, "col": 38, "value": 1.0},
            ],
            "truncated": False,
        },
        "lidar": {
            "ray_count": 120,
            "min_distance_m": 5.0,
            "hit_points": [[5.0, 0.0], [0.0, 5.0]],
        },
        "rl_trajectory": densify_polyline(rl_path),
        "rs_trajectory": densify_polyline(rs_path),
        "reward": {"value": 0.1 * step, "components": {}},
        "status": {"terminal": False, "name": "CONTINUE"},
    }


def build_fixture_trace(frame_count: int = 6, target_mode: str = TARGET_MODE_LEGACY_COSCOS) -> dict[str, Any]:
    mode = validate_target_mode(target_mode)
    frames = [_fixture_frame(step, mode) for step in range(frame_count)]
    return build_trace(
        source={
            "checkpoint": "fixture",
            "target_mode": mode,
            "created_by": "tools/stage4/debug_trace_export.py",
            "created_at": _now_iso(),
        },
        case={
            "case_uid": "fixture",
            "split": "fixture",
            "parking_type": "fixture",
        },
        static_geometry={
            "obstacle_polygons": [[[2.0, -1.0], [3.0, -1.0], [3.0, 1.0], [2.0, 1.0]]],
            "target_polygon": [[5.0, -1.0], [6.0, -1.0], [6.0, 1.0], [5.0, 1.0]],
            "map_bounds": [-10.0, -10.0, 10.0, 10.0],
        },
        frames=frames,
        summary={"success": False, "step_count": frame_count, "total_reward": sum(frame["reward"]["value"] for frame in frames)},
    )


def _split_case_uids(values: Sequence[str] | None) -> list[str]:
    if not values:
        return []
    result: list[str] = []
    for value in values:
        result.extend(part.strip() for part in value.split(",") if part.strip())
    return result


def _coords_from_ring(ring: Any) -> list[list[float]]:
    return [[float(x), float(y)] for x, y in ring.coords]


def _static_geometry(raw_env: Any) -> dict[str, Any]:
    return {
        "obstacle_polygons": [_coords_from_ring(obstacle.shape) for obstacle in raw_env.map.obstacles],
        "target_polygon": _coords_from_ring(raw_env.map.dest_box),
        "start_polygon": _coords_from_ring(raw_env.map.start_box),
        "map_bounds": [
            float(raw_env.map.xmin),
            float(raw_env.map.ymin),
            float(raw_env.map.xmax),
            float(raw_env.map.ymax),
        ],
    }


def _state_dict(state: Any) -> dict[str, float]:
    return {
        "x": float(state.loc.x),
        "y": float(state.loc.y),
        "heading": float(state.heading),
        "speed": float(state.speed),
        "steering": float(state.steering),
    }


def _trajectory_from_states(states: Iterable[Any]) -> list[list[float]]:
    return [[float(state.loc.x), float(state.loc.y), float(state.heading)] for state in states]


def _path_to_points(path: Any | None) -> list[list[float]]:
    if path is None:
        return []
    xs = getattr(path, "x", [])
    ys = getattr(path, "y", [])
    yaws = getattr(path, "yaw", [])
    return [[float(x), float(y), float(yaw)] for x, y, yaw in zip(xs, ys, yaws)]


def _action_mask_summary(action_mask: Any) -> dict[str, Any]:
    if action_mask is None:
        return {"valid_count": 0, "raw": []}
    raw = np.asarray(action_mask, dtype=float).reshape(-1)
    valid_indices = [int(index) for index, value in enumerate(raw) if value > 0.5]
    return {
        "valid_count": len(valid_indices),
        "valid_indices": valid_indices,
        "raw": [float(value) for value in raw.tolist()],
    }


def _ogm_payload(ogm: Any, max_cells: int | None) -> dict[str, Any]:
    if ogm is None:
        return {"size": ogm_default_grid()["size"], "sparse_cells": [], "truncated": False}
    all_nonzero = int(np.count_nonzero(np.asarray(ogm)))
    cells = sparsify_ogm(np.asarray(ogm), max_cells=max_cells)
    return {
        "size": ogm_default_grid()["size"],
        "nonzero_count": all_nonzero,
        "sparse_cells": cells,
        "truncated": max_cells is not None and all_nonzero > len(cells),
    }


def _lidar_payload(lidar: Any) -> dict[str, Any]:
    if lidar is None:
        return {"ray_count": 0, "min_distance_m": None, "hit_points": []}
    distances = np.asarray(lidar, dtype=float).reshape(-1)
    return {
        "ray_count": int(len(distances)),
        "min_distance_m": float(np.min(distances)) if len(distances) else None,
        "hit_points": project_lidar_hits(distances),
    }


def _case_source(case: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _json_default(value) if isinstance(value, (np.generic, np.ndarray, Path)) else value for key, value in case.items()}


def _load_stage4_runtime() -> dict[str, Any]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from configs import VALID_SPEED
    from env.env_wrapper import action_rescale
    from env.vehicle import Status
    from model.agent.parking_agent import ParkingAgent, RsPlanner
    from model.agent.sac_agent import SACAgent
    from tools.stage4.eval_stage4_ogm_checkpoint import build_ogm_agent_config, count_gear_shifts
    from tools.stage4.stage4_ogm_cases import load_cases
    from tools.stage4.train_HOPE_sac_ogm import build_stage4_env

    return {
        "torch": torch,
        "VALID_SPEED": VALID_SPEED,
        "build_stage4_env": build_stage4_env,
        "Status": Status,
        "ParkingAgent": ParkingAgent,
        "RsPlanner": RsPlanner,
        "SACAgent": SACAgent,
        "action_rescale": action_rescale,
        "build_ogm_agent_config": build_ogm_agent_config,
        "count_gear_shifts": count_gear_shifts,
        "load_cases": load_cases,
    }


def _select_cases(
    cases: Sequence[Mapping[str, Any]],
    case_uids: Sequence[str],
    case_indices: Sequence[int] | None,
) -> list[Mapping[str, Any]]:
    selected: list[Mapping[str, Any]] = []
    if case_uids:
        selected.extend(select_cases_by_uid(cases, case_uids))
    if case_indices:
        for index in case_indices:
            if index < 0 or index >= len(cases):
                raise ValueError(f"case-index out of range: {index}")
            selected.append(cases[index])
    if not selected:
        selected.append(cases[0])
    seen: set[str] = set()
    deduped: list[Mapping[str, Any]] = []
    for case in selected:
        case_uid = str(case["case_uid"])
        if case_uid not in seen:
            deduped.append(case)
            seen.add(case_uid)
    return deduped


def export_checkpoint_traces(
    checkpoint_path: str | Path,
    cases_path: str | Path,
    case_uids: Sequence[str] | None,
    case_indices: Sequence[int] | None,
    action_selection: str,
    max_steps: int | None,
    max_ogm_cells_per_frame: int | None,
    target_mode: str = TARGET_MODE_AUTO,
    maneuver_stability_config: ManeuverStabilityConfig | Mapping[str, object] | None = None,
) -> dict[str, Any]:
    runtime = _load_stage4_runtime()
    torch = runtime["torch"]
    build_stage4_env = runtime["build_stage4_env"]
    SACAgent = runtime["SACAgent"]
    ParkingAgent = runtime["ParkingAgent"]
    RsPlanner = runtime["RsPlanner"]
    build_ogm_agent_config = runtime["build_ogm_agent_config"]
    load_cases = runtime["load_cases"]
    count_gear_shifts = runtime["count_gear_shifts"]
    action_rescale = runtime["action_rescale"]
    Status = runtime["Status"]

    checkpoint = Path(checkpoint_path)
    cases_file = Path(cases_path)
    mode, target_mode_source = resolve_checkpoint_target_mode(checkpoint, target_mode)
    maneuver_config, maneuver_config_source = resolve_checkpoint_maneuver_stability_config(
        checkpoint,
        requested_config=maneuver_stability_config,
    )
    env = build_stage4_env(visualize=False, verbose=False, target_mode=mode)
    try:
        rl_agent = SACAgent(build_ogm_agent_config(env))
        rl_agent.load(str(checkpoint), params_only=True)
        step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * runtime["VALID_SPEED"][1]
        parking_agent = ParkingAgent(rl_agent, RsPlanner(step_ratio))
        all_cases = list(load_cases(cases_file))
        selected_cases = _select_cases(all_cases, _split_case_uids(case_uids), case_indices)

        source = {
            "checkpoint": str(checkpoint),
            "cases_json": str(cases_file),
            "target_mode": mode,
            "target_mode_source": target_mode_source,
            "maneuver_stability_mode": maneuver_config.mode,
            "created_by": "tools/stage4/debug_trace_export.py",
            "created_at": _now_iso(),
            "git_commit": _git_commit(),
            "action_selection": action_selection,
            "max_steps": max_steps,
        }

        traces: list[dict[str, Any]] = []
        with torch.no_grad():
            for case in selected_cases:
                seed = int(case["seed"])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                maneuver_tracker = ManeuverStabilityTracker(maneuver_config)
                obs = env.reset(int(case["hope_case_id"]), None, str(case["hope_level"]))
                parking_agent.reset()
                done = False
                frames: list[dict[str, Any]] = []
                path_length = 0.0
                total_reward = 0.0
                model_speeds: list[float] = []
                last_xy = np.array([env.vehicle.state.loc.x, env.vehicle.state.loc.y], dtype=float)
                info: Mapping[str, Any] = {"status": Status.CONTINUE, "path_to_dest": None, "reward_info": {}}
                step_index = 0

                while not done:
                    if max_steps is not None and step_index >= max_steps:
                        break

                    if action_selection == "choose_action":
                        was_executing_rs = bool(parking_agent.executing_rs)
                        raw_model_action, _ = parking_agent.choose_action(obs)
                        action_source = "RS" if was_executing_rs else "RL"
                        action_result = maneuver_tracker.apply(raw_model_action, source=action_source)
                        model_action = action_result.applied_action
                    else:
                        model_action, action_result = get_eval_action_with_maneuver_stability(
                            parking_agent=parking_agent,
                            tracker=maneuver_tracker,
                            obs=obs,
                        )
                        raw_model_action = action_result.raw_action
                        action_source = action_result.source
                    model_action = np.asarray(model_action, dtype=float).reshape(-1)
                    raw_model_action = np.asarray(raw_model_action, dtype=float).reshape(-1)
                    model_speeds.append(float(model_action[1]))
                    env_action = action_rescale(model_action, env.action_space, explore=False)
                    pre_ego_state = _state_dict(env.vehicle.state)
                    pre_rl_trajectory = densify_polyline(_trajectory_from_states(env.vehicle.trajectory))
                    pre_planner_route = getattr(getattr(parking_agent, "planner", None), "route", None)
                    pre_rs_points = _path_to_points(pre_planner_route)
                    target_payload = [
                        float(value) for value in np.asarray(obs.get("target"), dtype=float).reshape(-1).tolist()
                    ]
                    action_mask_payload = _action_mask_summary(obs.get("action_mask"))
                    ogm_payload = _ogm_payload(obs.get("ogm"), max_ogm_cells_per_frame)
                    lidar_payload = _lidar_payload(obs.get("lidar"))

                    next_obs, reward, done, info = env.step(model_action)
                    current_xy = np.array([env.vehicle.state.loc.x, env.vehicle.state.loc.y], dtype=float)
                    path_length += float(np.linalg.norm(last_xy - current_xy))
                    last_xy = current_xy
                    total_reward += float(reward)

                    rs_path = info.get("path_to_dest")

                    frames.append(
                        {
                            "step_index": step_index,
                            "sim_time_s": float(step_index * 0.5),
                            "timing": {
                                "observation": "pre_step",
                                "ego_state": "pre_step",
                                "action": "pre_step",
                                "reward_status": "post_step",
                            },
                            "ego_state": pre_ego_state,
                            "drive_direction": "forward"
                            if float(env_action[1]) > 0
                            else "reverse"
                            if float(env_action[1]) < 0
                            else "stopped",
                            "action": {
                                **build_action_trace_payload(
                                    source=action_source,
                                    raw_action=raw_model_action,
                                    applied_action=model_action,
                                    env_action=env_action,
                                    action_result=action_result,
                                ),
                            },
                            "target": target_payload,
                            "action_mask": action_mask_payload,
                            "ogm": ogm_payload,
                            "lidar": lidar_payload,
                            "rl_trajectory": pre_rl_trajectory,
                            "rs_trajectory": pre_rs_points,
                            "reward": {
                                "value": float(reward),
                                "components": {
                                    key: float(value) for key, value in dict(info.get("reward_info", {})).items()
                                },
                            },
                            "status": {
                                "terminal": bool(done),
                                "name": getattr(info.get("status"), "name", str(info.get("status"))),
                                "path_to_dest_available": rs_path is not None,
                                "planner_route_active": pre_planner_route is not None,
                            },
                        }
                    )

                    if rs_path is not None:
                        parking_agent.set_planner_path(rs_path)
                    obs = next_obs
                    step_index += 1

                status = info.get("status")
                summary = {
                    "success": status == Status.ARRIVED,
                    "terminal_status": getattr(status, "name", str(status)),
                    "step_count": len(frames),
                    "path_length_m": path_length,
                    "gear_shifts": count_gear_shifts(model_speeds),
                    "total_reward": total_reward,
                    "truncated_by_max_steps": bool(max_steps is not None and len(frames) >= max_steps and not done),
                    "action_selection": action_selection,
                }
                if maneuver_config.mode != MODE_OFF:
                    summary["maneuver_stability"] = {
                        "config": maneuver_config.to_dict(),
                        "config_source": maneuver_config_source,
                        "counters": maneuver_tracker.snapshot()["counters"],
                    }
                traces.append(
                    build_trace(
                        source=source,
                        case=_case_source(case),
                        static_geometry=_static_geometry(env.env),
                        frames=frames,
                        summary=summary,
                    )
                )

        if len(traces) == 1:
            return traces[0]
        return build_trace_collection(source, traces)
    finally:
        env.close()


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export Stage 4 OGM replay traces for offline debugging.")
    parser.add_argument("--fixture", action="store_true", help="Write a small deterministic trace without loading torch.")
    parser.add_argument("--checkpoint", type=Path, help="Stage 4 SAC checkpoint to replay.")
    parser.add_argument(
        "--cases-json",
        type=Path,
        default=REPO_ROOT / "docs" / "research" / "stage4_ogm_fixed_eval_cases_20260620.json",
    )
    parser.add_argument("--case-uid", action="append", help="Case uid to replay. Can be repeated or comma-separated.")
    parser.add_argument("--case-index", action="append", type=int, help="Case index to replay.")
    parser.add_argument("--action-selection", choices=("get_action", "choose_action"), default="get_action")
    parser.add_argument("--max-steps", type=int)
    parser.add_argument("--max-ogm-cells-per-frame", type=int, default=2048)
    parser.add_argument("--target-mode", choices=TARGET_MODE_REQUEST_CHOICES, default=TARGET_MODE_AUTO)
    parser.add_argument(
        "--maneuver_stability_mode",
        choices=MANEUVER_STABILITY_MODE_REQUEST_CHOICES,
        default=MANEUVER_STABILITY_MODE_AUTO,
    )
    parser.add_argument("--maneuver_stability_apply_to", choices=APPLY_TO_MODES, default="rl")
    parser.add_argument("--maneuver_stability_min_speed", type=float, default=1e-6)
    parser.add_argument("--maneuver_stability_hold_steps", type=int, default=1)
    parser.add_argument("--maneuver_stability_max_hold_speed", type=float, default=None)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.fixture:
        fixture_target_mode = TARGET_MODE_LEGACY_COSCOS if args.target_mode == TARGET_MODE_AUTO else args.target_mode
        trace = build_fixture_trace(target_mode=fixture_target_mode)
    else:
        if args.checkpoint is None:
            raise SystemExit("--checkpoint is required unless --fixture is set.")
        maneuver_config = None
        if args.maneuver_stability_mode != MANEUVER_STABILITY_MODE_AUTO:
            maneuver_config = ManeuverStabilityConfig(
                mode=args.maneuver_stability_mode,
                apply_to=args.maneuver_stability_apply_to,
                min_speed=args.maneuver_stability_min_speed,
                hold_steps=args.maneuver_stability_hold_steps,
                max_hold_speed=args.maneuver_stability_max_hold_speed,
            )
        trace = export_checkpoint_traces(
            checkpoint_path=args.checkpoint,
            cases_path=args.cases_json,
            case_uids=args.case_uid,
            case_indices=args.case_index,
            action_selection=args.action_selection,
            max_steps=args.max_steps,
            max_ogm_cells_per_frame=args.max_ogm_cells_per_frame,
            target_mode=args.target_mode,
            maneuver_stability_config=maneuver_config,
        )
    write_trace_json(trace, args.output)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
