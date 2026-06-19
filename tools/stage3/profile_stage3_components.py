from __future__ import annotations

import argparse
from contextlib import ExitStack, contextmanager
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import platform
import sys
import time
from typing import Any, Callable, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ORIGINAL_CWD = Path.cwd()
SCENES = ("Normal", "Complex", "Extrem", "dlp")
NOTE = "measurement_only_not_quality_evidence"
PROFILE_MODES = {
    "original": {
        "verbose": True,
        "render_mode": None,
        "expected_effective_render_mode": "human",
        "description": "Original argparse bool semantics: visualize truthy, verbose truthy.",
    },
    "command-only": {
        "verbose": False,
        "render_mode": "rgb_array",
        "expected_effective_render_mode": "rgb_array",
        "description": "Validated command-only flags: --visualize= --verbose=.",
    },
}
PROFILE_DETAILS = {
    "basic": (),
    "env-step": (
        "env.raw_step.total",
        "env.render.total",
        "env.render.draw",
        "env.render.img_capture",
        "env.render.img_process",
        "env.render.lidar_total",
        "env.render.lidar_get_observation",
        "env.render.lidar_rotate_filter",
        "env.render.lidar_fast_calc",
        "env.render.action_mask",
        "env.render.action_mask_post_process",
        "env.render.target",
        "env.sim.vehicle_step",
        "env.sim.kinematic_step",
        "env.sim.vehicle_retreat",
        "env.status.total",
        "env.status.detect_collision",
        "env.status.detect_outbound",
        "env.status.check_arrived",
        "env.status.check_time_exceeded",
        "env.reward.total",
        "env.reward.components",
        "env.rs.find_path",
        "env.rs.calc_all_paths",
        "env.rs.calc_optimal_path",
        "env.rs.traj_valid",
        "env.wrapper.action_rescale",
        "env.wrapper.reward_shaping",
        "env.wrapper.observation_rescale",
    ),
}

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


class TimerTable:
    def __init__(self):
        self.rows = {}

    def record(self, name, seconds):
        row = self.rows.setdefault(name, {"seconds": 0.0, "calls": 0})
        row["seconds"] += seconds
        row["calls"] += 1

    def summary(self):
        total = sum(row["seconds"] for row in self.rows.values())
        return {
            name: {
                "seconds": row["seconds"],
                "calls": row["calls"],
                "avg_ms": (row["seconds"] / row["calls"]) * 1000.0 if row["calls"] else 0.0,
                "percent_measured_wall": (row["seconds"] / total) * 100.0 if total else 0.0,
            }
            for name, row in sorted(self.rows.items())
        }


@contextmanager
def timed_local_method(target: Any, method_name: str, timers: TimerTable, component_name: str) -> Iterator[None]:
    original = getattr(target, method_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timers.record(component_name, time.perf_counter() - start)

    setattr(target, method_name, wrapped)
    try:
        yield
    finally:
        setattr(target, method_name, original)


@contextmanager
def timed_callable_attr(target: Any, attr_name: str, timers: TimerTable, component_name: str) -> Iterator[None]:
    original = getattr(target, attr_name)

    def wrapped(*args: Any, **kwargs: Any) -> Any:
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            timers.record(component_name, time.perf_counter() - start)

    setattr(target, attr_name, wrapped)
    try:
        yield
    finally:
        setattr(target, attr_name, original)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be greater than 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be greater than or equal to 0")
    return parsed


def profile_mode_settings(profile_mode: str) -> dict[str, Any]:
    try:
        return dict(PROFILE_MODES[profile_mode])
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_MODES))
        raise ValueError(f"unknown profile mode {profile_mode!r}; expected one of: {choices}") from exc


def profile_detail_components(detail: str) -> tuple[str, ...]:
    try:
        return PROFILE_DETAILS[detail]
    except KeyError as exc:
        choices = ", ".join(sorted(PROFILE_DETAILS))
        raise ValueError(f"unknown profile detail {detail!r}; expected one of: {choices}") from exc


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure bounded HOPE Stage 3 component timings without producing training-quality evidence."
    )
    parser.add_argument("--episodes", type=_positive_int, default=3)
    parser.add_argument("--max-steps", type=_positive_int, default=50)
    parser.add_argument("--updates", type=_non_negative_int, default=5)
    parser.add_argument("--profile-mode", choices=sorted(PROFILE_MODES), default="command-only")
    parser.add_argument("--detail", choices=sorted(PROFILE_DETAILS), default="basic")
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ORIGINAL_CWD / path


def _load_hope_objects() -> dict[str, Any]:
    from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, SEED  # noqa: PLC0415
    from env.car_parking_base import CarParking  # noqa: PLC0415
    from env.env_wrapper import CarParkingWrapper  # noqa: PLC0415
    from env.vehicle import VALID_SPEED  # noqa: PLC0415
    from model.agent.parking_agent import ParkingAgent, RsPlanner  # noqa: PLC0415
    from model.agent.sac_agent import SACAgent  # noqa: PLC0415
    from model.replay_memory import ReplayMemory  # noqa: PLC0415
    from train.train_HOPE_sac import DlpCaseChoose, SceneChoose  # noqa: PLC0415

    return {
        "ACTOR_CONFIGS": ACTOR_CONFIGS,
        "CRITIC_CONFIGS": CRITIC_CONFIGS,
        "CarParking": CarParking,
        "CarParkingWrapper": CarParkingWrapper,
        "ParkingAgent": ParkingAgent,
        "ReplayMemory": ReplayMemory,
        "RsPlanner": RsPlanner,
        "SACAgent": SACAgent,
        "SEED": SEED,
        "DlpCaseChoose": DlpCaseChoose,
        "SceneChoose": SceneChoose,
        "VALID_SPEED": VALID_SPEED,
    }


def _make_env(hope: dict[str, Any], profile_mode: str) -> Any:
    settings = profile_mode_settings(profile_mode)
    return hope["CarParkingWrapper"](
        hope["CarParking"](
            fps=100,
            verbose=bool(settings["verbose"]),
            render_mode=settings["render_mode"],
        )
    )


def _make_agent(hope: dict[str, Any], env: Any) -> Any:
    configs = {
        "discrete": False,
        "observation_shape": env.observation_shape,
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": hope["ACTOR_CONFIGS"],
        "critic_layers": hope["CRITIC_CONFIGS"],
    }
    rl_agent = hope["SACAgent"](configs)
    step_ratio = (
        env.vehicle.kinetic_model.step_len
        * env.vehicle.kinetic_model.n_step
        * hope["VALID_SPEED"][1]
    )
    return hope["ParkingAgent"](rl_agent, hope["RsPlanner"](step_ratio))


def _install_env_step_detail_timers(env: Any, timers: TimerTable) -> ExitStack:
    import env.reeds_shepp as rs_curve  # noqa: PLC0415

    stack = ExitStack()
    raw_env = env.env

    stack.enter_context(timed_callable_attr(env, "action_func", timers, "env.wrapper.action_rescale"))
    stack.enter_context(timed_callable_attr(env, "reward_func", timers, "env.wrapper.reward_shaping"))
    stack.enter_context(timed_callable_attr(env, "obs_func", timers, "env.wrapper.observation_rescale"))

    stack.enter_context(timed_local_method(raw_env, "step", timers, "env.raw_step.total"))
    stack.enter_context(timed_local_method(raw_env, "render", timers, "env.render.total"))
    stack.enter_context(timed_local_method(raw_env, "_render", timers, "env.render.draw"))
    stack.enter_context(timed_local_method(raw_env, "_get_img_observation", timers, "env.render.img_capture"))
    stack.enter_context(timed_local_method(raw_env, "_process_img_observation", timers, "env.render.img_process"))
    stack.enter_context(timed_local_method(raw_env, "_get_lidar_observation", timers, "env.render.lidar_total"))
    stack.enter_context(timed_local_method(raw_env, "_get_targt_repr", timers, "env.render.target"))

    if raw_env.use_action_mask:
        stack.enter_context(timed_local_method(raw_env.action_filter, "get_steps", timers, "env.render.action_mask"))
        stack.enter_context(
            timed_local_method(raw_env.action_filter, "post_process", timers, "env.render.action_mask_post_process")
        )
    if raw_env.use_lidar_observation:
        stack.enter_context(timed_local_method(raw_env.lidar, "get_observation", timers, "env.render.lidar_get_observation"))
        stack.enter_context(
            timed_local_method(raw_env.lidar, "_rotate_and_filter_obstacles", timers, "env.render.lidar_rotate_filter")
        )
        stack.enter_context(timed_local_method(raw_env.lidar, "_fast_calc_lidar_obs", timers, "env.render.lidar_fast_calc"))

    stack.enter_context(timed_local_method(raw_env.vehicle, "step", timers, "env.sim.vehicle_step"))
    stack.enter_context(timed_local_method(raw_env.vehicle.kinetic_model, "step", timers, "env.sim.kinematic_step"))
    stack.enter_context(timed_local_method(raw_env.vehicle, "retreat", timers, "env.sim.vehicle_retreat"))

    stack.enter_context(timed_local_method(raw_env, "_check_status", timers, "env.status.total"))
    stack.enter_context(timed_local_method(raw_env, "_detect_collision", timers, "env.status.detect_collision"))
    stack.enter_context(timed_local_method(raw_env, "_detect_outbound", timers, "env.status.detect_outbound"))
    stack.enter_context(timed_local_method(raw_env, "_check_arrived", timers, "env.status.check_arrived"))
    stack.enter_context(timed_local_method(raw_env, "_check_time_exceeded", timers, "env.status.check_time_exceeded"))

    stack.enter_context(timed_local_method(raw_env, "get_reward", timers, "env.reward.total"))
    stack.enter_context(timed_local_method(raw_env, "_get_reward", timers, "env.reward.components"))
    stack.enter_context(timed_local_method(raw_env, "find_rs_path", timers, "env.rs.find_path"))
    stack.enter_context(timed_local_method(raw_env, "is_traj_valid", timers, "env.rs.traj_valid"))
    stack.enter_context(timed_callable_attr(rs_curve, "calc_all_paths", timers, "env.rs.calc_all_paths"))
    stack.enter_context(timed_callable_attr(rs_curve, "calc_optimal_path", timers, "env.rs.calc_optimal_path"))

    return stack


def _reset_env(env: Any, episode_index: int) -> Any:
    scene = SCENES[episode_index % len(SCENES)]
    if scene == "dlp":
        return scene, episode_index, env.reset(episode_index, None, scene)
    return scene, None, env.reset(None, None, scene)


def _run_episodes(
    *,
    env: Any,
    parking_agent: Any,
    episodes: int,
    max_steps: int,
    timers: TimerTable,
) -> dict[str, Any]:
    transitions = 0
    episode_summaries: list[dict[str, Any]] = []

    with (
        timed_local_method(env, "reset", timers, "env.reset"),
        timed_local_method(env, "step", timers, "env.step"),
        timed_local_method(parking_agent, "get_action", timers, "ParkingAgent.get_action"),
        timed_local_method(parking_agent, "push_memory", timers, "replay.push"),
    ):
        for episode_index in range(episodes):
            parking_agent.reset()
            scene, case_id, obs = _reset_env(env, episode_index)
            done = False
            steps = 0
            total_reward = 0.0
            status = None

            while not done and steps < max_steps:
                action, log_prob = parking_agent.get_action(obs)
                next_obs, reward, done, info = env.step(action)
                parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
                transitions += 1
                steps += 1
                total_reward += float(reward)
                status = str(info.get("status"))

                path_to_dest = info.get("path_to_dest")
                if path_to_dest is not None:
                    parking_agent.set_planner_path(path_to_dest)

                obs = next_obs

            episode_summaries.append(
                {
                    "episode": episode_index + 1,
                    "scene": scene,
                    "case_id": case_id,
                    "steps": steps,
                    "done": bool(done),
                    "status": status,
                    "total_reward": total_reward,
                }
            )

    return {
        "episodes_completed": len(episode_summaries),
        "transitions_collected": transitions,
        "episode_summaries": episode_summaries,
    }


def _time_replay_sample(parking_agent: Any, timers: TimerTable, skipped: list[dict[str, str]]) -> None:
    memory_len = len(parking_agent.memory)
    if memory_len <= 0:
        skipped.append(
            {
                "component": "replay.sample",
                "reason": "replay memory had no transitions after bounded diagnostic",
            }
        )
        return

    sample_size = min(int(parking_agent.configs.batch_size), memory_len)
    with timed_local_method(parking_agent.memory, "sample", timers, "replay.sample"):
        parking_agent.memory.sample(sample_size)


def _time_updates(
    *,
    parking_agent: Any,
    updates: int,
    timers: TimerTable,
    skipped: list[dict[str, str]],
) -> int:
    if updates <= 0:
        skipped.append({"component": "SACAgent.update", "reason": "--updates was 0"})
        return 0

    memory_len = len(parking_agent.memory)
    batch_size = int(parking_agent.configs.batch_size)
    if memory_len < batch_size:
        skipped.append(
            {
                "component": "SACAgent.update",
                "reason": f"replay memory length {memory_len} is below SAC batch size {batch_size}",
            }
        )
        return 0

    completed = 0
    with (
        timed_local_method(parking_agent.memory, "sample", timers, "replay.sample"),
        timed_local_method(parking_agent, "update", timers, "SACAgent.update"),
    ):
        for _ in range(updates):
            try:
                parking_agent.update()
            except Exception as exc:  # Keep the diagnostic measurement-only and report why update stopped.
                skipped.append(
                    {
                        "component": "SACAgent.update",
                        "reason": f"stopped after {completed} updates due to {type(exc).__name__}: {exc}",
                    }
                )
                break
            completed += 1
    return completed


def _environment_report() -> dict[str, Any]:
    report: dict[str, Any] = {
        "python": sys.version.replace("\n", " "),
        "platform": platform.platform(),
        "sdl_videodriver": os.environ.get("SDL_VIDEODRIVER"),
        "torch_force_no_weights_only_load": os.environ.get("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"),
    }
    try:
        import torch  # noqa: PLC0415

        report.update(
            {
                "torch_version": torch.__version__,
                "torch_cuda_available": bool(torch.cuda.is_available()),
                "torch_device_count": int(torch.cuda.device_count()),
            }
        )
        if torch.cuda.is_available():
            report["torch_cuda_device_name"] = torch.cuda.get_device_name(0)
    except Exception as exc:
        report["torch_import_error"] = f"{type(exc).__name__}: {exc}"
    return report


def run_diagnostic(episodes: int, max_steps: int, updates: int, profile_mode: str, detail: str) -> dict[str, Any]:
    timers = TimerTable()
    skipped: list[dict[str, str]] = []
    notes: list[str] = [
        NOTE,
        "bounded diagnostic only; do not use as Stage 3 training-quality evidence",
        "timing wrappers are installed only on objects created by this diagnostic process",
    ]
    if detail != "basic":
        notes.append("detail timings include nested calls; do not sum them as exclusive wall-clock percentages")
    started_at = time.perf_counter()
    created_at_utc = datetime.now(timezone.utc).isoformat(timespec="seconds")
    env = None
    status = "pass"

    previous_cwd = Path.cwd()
    os.chdir(SRC_ROOT)
    try:
        hope = _load_hope_objects()
        np.random.seed(int(hope["SEED"]))
        env = _make_env(hope, profile_mode)
        env.action_space.seed(int(hope["SEED"]))
        parking_agent = _make_agent(hope, env)

        with _install_env_step_detail_timers(env, timers) if detail == "env-step" else ExitStack():
            run_summary = _run_episodes(
                env=env,
                parking_agent=parking_agent,
                episodes=episodes,
                max_steps=max_steps,
                timers=timers,
            )
        _time_replay_sample(parking_agent, timers, skipped)
        updates_completed = _time_updates(
            parking_agent=parking_agent,
            updates=updates,
            timers=timers,
            skipped=skipped,
        )
    except Exception as exc:
        status = "fail"
        run_summary = {
            "episodes_completed": 0,
            "transitions_collected": 0,
            "episode_summaries": [],
        }
        updates_completed = 0
        notes.append(f"diagnostic_failed: {type(exc).__name__}: {exc}")
    finally:
        if env is not None:
            env.close()
        os.chdir(previous_cwd)

    total_seconds = time.perf_counter() - started_at
    components = timers.summary()
    measured_component_seconds = sum(row["seconds"] for row in components.values())
    if skipped and status == "pass":
        status = "pass_with_skipped_components"

    return {
        "schema_version": 1,
        "status": status,
        "note": NOTE,
        "created_at_utc": created_at_utc,
        "original_hope_imports": [
            "train.train_HOPE_sac.SceneChoose",
            "train.train_HOPE_sac.DlpCaseChoose",
            "env.car_parking_base.CarParking",
            "env.env_wrapper.CarParkingWrapper",
            "model.agent.parking_agent.ParkingAgent",
            "model.agent.parking_agent.RsPlanner",
            "model.agent.sac_agent.SACAgent",
            "model.replay_memory.ReplayMemory",
        ],
        "cli_args": {
            "episodes": episodes,
            "max_steps": max_steps,
            "updates": updates,
            "profile_mode": profile_mode,
            "detail": detail,
        },
        "profile_mode": {
            "name": profile_mode,
            **profile_mode_settings(profile_mode),
            "effective_render_mode": getattr(env, "render_mode", None) if env is not None else None,
        },
        "profile_detail": {
            "name": detail,
            "components": list(profile_detail_components(detail)),
            "nested_timings": bool(detail != "basic"),
        },
        "environment": _environment_report(),
        "total_seconds": total_seconds,
        "measured_component_seconds": measured_component_seconds,
        "components": components,
        "episodes_completed": run_summary["episodes_completed"],
        "transitions_collected": run_summary["transitions_collected"],
        "updates_requested": updates,
        "updates_completed": updates_completed,
        "episode_summaries": run_summary["episode_summaries"],
        "skipped_components": skipped,
        "notes": notes,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage 3 Component Profile",
        "",
        f"- Status: `{report['status']}`",
        f"- Note: `{report['note']}`",
        f"- Profile mode: `{report['cli_args']['profile_mode']}`",
        f"- Detail: `{report['cli_args']['detail']}`",
        f"- Render mode: `{report['profile_mode']['effective_render_mode']}`",
        f"- Verbose env: `{report['profile_mode']['verbose']}`",
        f"- Episodes: `{report['cli_args']['episodes']}`",
        f"- Max steps per episode: `{report['cli_args']['max_steps']}`",
        f"- Updates requested/completed: `{report['updates_requested']}` / `{report['updates_completed']}`",
        f"- Total seconds: `{report['total_seconds']:.6f}`",
        f"- Measured component seconds: `{report['measured_component_seconds']:.6f}`",
        f"- Transitions collected: `{report['transitions_collected']}`",
        "",
        "This is a bounded measurement-only diagnostic, not training-quality evidence.",
        "",
        "## Components",
        "",
        "| Component | Seconds | Calls | Avg ms | Percent measured wall |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, row in report["components"].items():
        lines.append(
            f"| `{name}` | {row['seconds']:.6f} | {row['calls']} | "
            f"{row['avg_ms']:.3f} | {row['percent_measured_wall']:.2f}% |"
        )

    lines.extend(["", "## Skipped Components", ""])
    if report["skipped_components"]:
        for item in report["skipped_components"]:
            lines.append(f"- `{item['component']}`: {item['reason']}")
    else:
        lines.append("- None")

    lines.extend(["", "## Episodes", ""])
    for item in report["episode_summaries"]:
        lines.append(
            f"- Episode {item['episode']} `{item['scene']}`: steps=`{item['steps']}`, "
            f"done=`{item['done']}`, status=`{item['status']}`, "
            f"total_reward=`{item['total_reward']:.6f}`"
        )

    lines.extend(["", "## Notes", ""])
    for note in report["notes"]:
        lines.append(f"- {note}")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_json(report: dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def main() -> int:
    args = _parse_args()
    json_path = _resolve_output_path(args.json)
    markdown_path = _resolve_output_path(args.markdown)
    report = run_diagnostic(args.episodes, args.max_steps, args.updates, args.profile_mode, args.detail)
    write_json(report, json_path)
    write_markdown(report, markdown_path)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
