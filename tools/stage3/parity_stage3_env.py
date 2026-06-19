from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
import sys
from pathlib import Path
from typing import Any, Iterator

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
ORIGINAL_CWD = Path.cwd()

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, str(SRC_ROOT))

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
import env.parking_map_dlp as parking_map_dlp  # noqa: E402


SCENES = ("Normal", "Complex", "Extrem", "dlp")
OBS_KEYS = ("img", "lidar", "target", "action_mask")
SCRIPTED_ACTIONS = (
    np.array([0.0, 0.4], dtype=np.float32),
    np.array([0.25, 0.3], dtype=np.float32),
    np.array([-0.25, 0.3], dtype=np.float32),
    np.array([0.5, -0.2], dtype=np.float32),
    np.array([-0.5, -0.2], dtype=np.float32),
)


@contextmanager
def _dlp_false_path_compat() -> Iterator[None]:
    # The Task 6 plan passes False as the DLP data-dir sentinel; HOPE's DLP map
    # code otherwise treats bool False as file descriptor 0 on current Python.
    had_module_open = hasattr(parking_map_dlp, "open")
    previous_open = getattr(parking_map_dlp, "open", None)

    def compat_open(file: Any, *args: Any, **kwargs: Any):
        if file is False:
            file = parking_map_dlp.ParkingMapDLP.default["path"]
        return open(file, *args, **kwargs)

    parking_map_dlp.open = compat_open
    try:
        yield
    finally:
        if had_module_open:
            parking_map_dlp.open = previous_open
        else:
            delattr(parking_map_dlp, "open")


def _make_env(render_mode: str | None) -> CarParkingWrapper:
    kwargs: dict[str, Any] = {"fps": 100, "verbose": False}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return CarParkingWrapper(CarParking(**kwargs))


def _reset(env: CarParkingWrapper, scene: str, case_index: int) -> dict[str, Any]:
    if scene == "dlp":
        with _dlp_false_path_compat():
            return env.reset(case_index, False, "dlp")
    return env.reset(None, None, scene)


def _reset_pair(
    left_env: CarParkingWrapper,
    right_env: CarParkingWrapper,
    scene: str,
    case_index: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    seed = 30_000 + (SCENES.index(scene) * 1_000) + case_index
    np.random.seed(seed)
    left_obs = _reset(left_env, scene, case_index)
    np.random.seed(seed)
    right_obs = _reset(right_env, scene, case_index)
    return left_obs, right_obs


def _compare_array(key: str, left: Any, right: Any, context: str) -> list[str]:
    if left is None or right is None:
        if left is None and right is None:
            return []
        return [f"{context}.{key}_none_mismatch"]

    left_array = np.asarray(left)
    right_array = np.asarray(right)
    if left_array.shape != right_array.shape:
        return [f"{context}.{key}_shape_mismatch:{left_array.shape}!={right_array.shape}"]

    if key == "action_mask":
        if not np.array_equal(left_array, right_array):
            return [f"{context}.action_mask_mismatch"]
        return []

    if not np.allclose(left_array, right_array, rtol=1e-6, atol=1e-6):
        diff = np.max(np.abs(left_array.astype(np.float64) - right_array.astype(np.float64)))
        return [f"{context}.{key}_mismatch:max_abs_diff={diff:.12g}"]
    return []


def _compare_obs(left: dict[str, Any], right: dict[str, Any], context: str) -> list[str]:
    failures: list[str] = []
    for key in OBS_KEYS:
        if key not in left or key not in right:
            failures.append(f"{context}.{key}_missing")
            continue
        failures.extend(_compare_array(key, left[key], right[key], context))
    return failures


def _path_to_dest_present(info: dict[str, Any]) -> bool:
    return info.get("path_to_dest") is not None


def _compare_step(
    context: str,
    left_obs: dict[str, Any],
    left_reward: float,
    left_done: bool,
    left_info: dict[str, Any],
    right_obs: dict[str, Any],
    right_reward: float,
    right_done: bool,
    right_info: dict[str, Any],
) -> list[str]:
    failures = _compare_obs(left_obs, right_obs, context)
    if not np.allclose(float(left_reward), float(right_reward), rtol=1e-6, atol=1e-6):
        failures.append(f"{context}.reward_mismatch:{left_reward!r}!={right_reward!r}")
    if bool(left_done) != bool(right_done):
        failures.append(f"{context}.done_mismatch:{bool(left_done)}!={bool(right_done)}")
    if left_info.get("status") != right_info.get("status"):
        failures.append(f"{context}.status_mismatch:{left_info.get('status')}!={right_info.get('status')}")
    if _path_to_dest_present(left_info) != _path_to_dest_present(right_info):
        failures.append(
            f"{context}.path_to_dest_presence_mismatch:"
            f"{_path_to_dest_present(left_info)}!={_path_to_dest_present(right_info)}"
        )
    return failures


def _case_result(scene: str, case_index: int, terminal_step: int | None, failures: list[str]) -> dict[str, Any]:
    return {
        "scene": scene,
        "case": case_index,
        "terminal_step": terminal_step,
        "failures": sorted(set(failures)),
    }


def run_parity(episodes_per_scene: int, max_steps: int) -> dict[str, Any]:
    left_env = _make_env(None)
    right_env = _make_env("rgb_array")
    results: list[dict[str, Any]] = []

    try:
        for scene in SCENES:
            for case_index in range(episodes_per_scene):
                failures: list[str] = []
                terminal_step = None
                try:
                    left_obs, right_obs = _reset_pair(left_env, right_env, scene, case_index)
                    failures.extend(_compare_obs(left_obs, right_obs, "reset"))

                    for step_index in range(max_steps):
                        action = SCRIPTED_ACTIONS[step_index % len(SCRIPTED_ACTIONS)]
                        left_obs, left_reward, left_done, left_info = left_env.step(action)
                        right_obs, right_reward, right_done, right_info = right_env.step(action)
                        context = f"step_{step_index + 1:03d}"
                        failures.extend(
                            _compare_step(
                                context,
                                left_obs,
                                left_reward,
                                left_done,
                                left_info,
                                right_obs,
                                right_reward,
                                right_done,
                                right_info,
                            )
                        )
                        if left_done or right_done:
                            terminal_step = step_index + 1
                            break
                except Exception as exc:  # Preserve report evidence for environment/runtime failures.
                    failures.append(f"exception:{type(exc).__name__}:{exc}")

                results.append(_case_result(scene, case_index, terminal_step, failures))
    finally:
        left_env.close()
        right_env.close()

    failed_cases = [result for result in results if result["failures"]]
    return {
        "schema_version": 1,
        "status": "fail" if failed_cases else "pass",
        "episodes_per_scene": episodes_per_scene,
        "max_steps": max_steps,
        "render_modes": {"left": "default", "right": "rgb_array"},
        "observation_keys": list(OBS_KEYS),
        "results": results,
    }


def write_markdown(report: dict[str, Any], path: Path) -> None:
    lines = [
        "# Stage 3 Environment Parity",
        "",
        f"- Status: `{report['status']}`",
        f"- Episodes per scene: `{report['episodes_per_scene']}`",
        f"- Max steps: `{report['max_steps']}`",
        f"- Render modes: `default` vs `rgb_array`",
        "",
        "## Cases",
        "",
    ]
    for item in report["results"]:
        failures = item["failures"]
        failure_text = "pass" if not failures else "; ".join(failures)
        lines.append(
            f"- {item['scene']} case {item['case']}: terminal_step="
            f"`{item['terminal_step']}`, failures=`{failure_text}`"
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check HOPE Stage 3 render-mode environment parity.")
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    return parser.parse_args()


def _resolve_output_path(path: Path) -> Path:
    if path.is_absolute():
        return path
    return ORIGINAL_CWD / path


def _run_from_src(episodes_per_scene: int, max_steps: int) -> dict[str, Any]:
    previous_cwd = Path.cwd()
    os.chdir(SRC_ROOT)
    try:
        return run_parity(episodes_per_scene, max_steps)
    finally:
        os.chdir(previous_cwd)


def main() -> int:
    args = _parse_args()
    json_path = _resolve_output_path(args.json)
    markdown_path = _resolve_output_path(args.markdown)

    try:
        report = _run_from_src(args.episodes_per_scene, args.max_steps)
    except Exception as exc:
        report = {
            "schema_version": 1,
            "status": "fail",
            "episodes_per_scene": args.episodes_per_scene,
            "max_steps": args.max_steps,
            "render_modes": {"left": "default", "right": "rgb_array"},
            "observation_keys": list(OBS_KEYS),
            "results": [],
            "error": f"{type(exc).__name__}: {exc}",
        }

    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, markdown_path)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
