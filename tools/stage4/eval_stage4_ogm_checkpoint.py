from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


EXPECTED_SPLITS = ("Sim-Normal", "Sim-Complex")
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from tools.stage4.stage4_target_transform import (  # noqa: E402
    TARGET_MODE_AUTO,
    TARGET_MODE_REQUEST_CHOICES,
    resolve_checkpoint_target_mode,
    validate_target_mode,
)
from tools.stage4.stage4_maneuver_stability import (  # noqa: E402
    APPLY_TO_MODES,
    MODE_OFF,
    MODES as MANEUVER_STABILITY_MODES,
    ManeuverStabilityConfig,
    ManeuverStabilityResult,
    ManeuverStabilityTracker,
)

MANEUVER_STABILITY_MODE_AUTO = "auto"
MANEUVER_STABILITY_MODE_REQUEST_CHOICES = (MANEUVER_STABILITY_MODE_AUTO,) + MANEUVER_STABILITY_MODES


def count_gear_shifts(speeds: Iterable[float]) -> int:
    """Count gear shifts as commanded-speed sign changes, ignoring zero speeds."""
    last_sign = 0
    shifts = 0
    for speed in speeds:
        sign = 1 if speed > 0 else -1 if speed < 0 else 0
        if sign == 0:
            continue
        if last_sign != 0 and sign != last_sign:
            shifts += 1
        last_sign = sign
    return shifts


def summarize_eval_records(
    records: Iterable[Mapping[str, object]],
    expected_splits: Iterable[str] = EXPECTED_SPLITS,
) -> dict[str, dict[str, float | None]]:
    """Summarize PSR over all cases and ANGS/PL over successful cases by expected split."""
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split"])].append(record)

    missing_splits = [split for split in expected_splits if not grouped.get(split)]
    if len(missing_splits) == 1:
        raise ValueError(f"Missing eval records for expected split: {missing_splits[0]}")
    if missing_splits:
        raise ValueError(f"Missing eval records for expected splits: {', '.join(missing_splits)}")

    summary: dict[str, dict[str, float | None]] = {}
    for split, split_records in grouped.items():
        successes = [record for record in split_records if bool(record["success"])]
        psr = len(successes) / len(split_records) if split_records else 0.0
        if successes:
            angs = sum(float(record["gear_shifts"]) for record in successes) / len(successes)
            pl = sum(float(record["path_length"]) for record in successes) / len(successes)
        else:
            angs = None
            pl = None
        summary[split] = {"psr": psr, "angs": angs, "pl": pl}
    return summary


def build_eval_result(
    records: Iterable[Mapping[str, object]],
    target_mode: str,
    target_mode_source: Mapping[str, object] | None = None,
    maneuver_stability_config: ManeuverStabilityConfig | Mapping[str, object] | None = None,
    maneuver_stability_summary: Mapping[str, object] | None = None,
) -> dict[str, object]:
    records_list = list(records)
    mode = validate_target_mode(target_mode)
    result = {
        "target_mode": mode,
        "records": records_list,
        "summary": summarize_eval_records(records_list),
    }
    if target_mode_source is not None:
        result["target_mode_source"] = dict(target_mode_source)
    maneuver_config = _coerce_maneuver_stability_config(maneuver_stability_config)
    if maneuver_config.mode != MODE_OFF:
        result["maneuver_stability_config"] = maneuver_config.to_dict()
    if maneuver_config.mode != MODE_OFF and maneuver_stability_summary is not None:
        result["maneuver_stability_summary"] = dict(maneuver_stability_summary)
    return result


def _coerce_maneuver_stability_config(
    config: ManeuverStabilityConfig | Mapping[str, object] | None,
) -> ManeuverStabilityConfig:
    if isinstance(config, ManeuverStabilityConfig):
        return config
    return ManeuverStabilityConfig.from_mapping(config)


def _candidate_state_paths_for_checkpoint(checkpoint_path: Path) -> list[Path]:
    candidates: list[Path] = []
    match = re.fullmatch(r"SAC_(\d+)\.pt", checkpoint_path.name)
    if match:
        candidates.append(checkpoint_path.with_name("stage4_state_%s.pt" % match.group(1)))
    candidates.append(checkpoint_path.with_name("stage4_state_latest.pt"))
    return candidates


def resolve_checkpoint_maneuver_stability_config(
    checkpoint_path: str | Path,
    requested_config: ManeuverStabilityConfig | Mapping[str, object] | None = None,
) -> tuple[ManeuverStabilityConfig, dict[str, str]]:
    if requested_config is not None:
        config = _coerce_maneuver_stability_config(requested_config)
        return config, {"type": "explicit", "value": config.mode}

    import torch

    checkpoint = Path(checkpoint_path)
    for state_path in _candidate_state_paths_for_checkpoint(checkpoint):
        if not state_path.exists():
            continue
        state = torch.load(state_path, map_location="cpu", weights_only=False)
        if "maneuver_stability_config" in state:
            config = ManeuverStabilityConfig.from_mapping(state["maneuver_stability_config"])
            return config, {"type": "stage4_state", "path": str(state_path)}
        return ManeuverStabilityConfig(), {"type": "stage4_state_legacy_default", "path": str(state_path)}
    return ManeuverStabilityConfig(), {"type": "legacy_default"}


def apply_maneuver_stability_to_eval_action(
    *,
    tracker: ManeuverStabilityTracker,
    action: Any,
    source: str,
) -> ManeuverStabilityResult:
    return tracker.apply(action, source=source)


def get_eval_action_with_maneuver_stability(
    *,
    parking_agent: Any,
    tracker: ManeuverStabilityTracker,
    obs: Any,
) -> tuple[np.ndarray, ManeuverStabilityResult]:
    was_executing_rs = bool(getattr(parking_agent, "executing_rs", False))
    action, _ = parking_agent.get_action(obs)
    action_source = "RS" if was_executing_rs else "RL"
    action_result = apply_maneuver_stability_to_eval_action(
        tracker=tracker,
        action=action,
        source=action_source,
    )
    return action_result.applied_action, action_result


def build_ogm_agent_config(env: Any) -> dict[str, object]:
    from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, N_DISCRETE_ACTION

    actor_layers = deepcopy(ACTOR_CONFIGS)
    actor_layers.update(
        {
            "n_modal": 3,
            "lidar_shape": None,
            "target_shape": 5,
            "action_mask_shape": N_DISCRETE_ACTION,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
        }
    )

    critic_layers = deepcopy(CRITIC_CONFIGS)
    critic_layers.update(
        {
            "n_modal": 4,
            "lidar_shape": None,
            "target_shape": 5,
            "action_mask_shape": N_DISCRETE_ACTION,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
        }
    )

    return {
        "discrete": False,
        "observation_shape": {
            "target": env.observation_shape["target"],
            "action_mask": env.observation_shape["action_mask"],
            "ogm": env.observation_shape["ogm"],
        },
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_layers,
        "critic_layers": critic_layers,
    }


def _case_status_name(status: object) -> str:
    return getattr(status, "name", str(status))


def evaluate_checkpoint(
    checkpoint_path: str | Path,
    cases_path: str | Path,
    output_json: str | Path,
    target_mode: str = TARGET_MODE_AUTO,
    maneuver_stability_config: ManeuverStabilityConfig | Mapping[str, object] | None = None,
) -> dict[str, object]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from configs import VALID_SPEED
    from env.vehicle import Status
    from model.agent.parking_agent import ParkingAgent, RsPlanner
    from model.agent.sac_agent import SACAgent
    from tools.stage4.stage4_ogm_cases import load_cases
    from tools.stage4.train_HOPE_sac_ogm import build_stage4_env

    checkpoint_path = Path(checkpoint_path)
    cases_path = Path(cases_path)
    output_json = Path(output_json)
    mode, target_mode_source = resolve_checkpoint_target_mode(checkpoint_path, target_mode)
    maneuver_config, maneuver_config_source = resolve_checkpoint_maneuver_stability_config(
        checkpoint_path,
        requested_config=maneuver_stability_config,
    )
    maneuver_tracker = ManeuverStabilityTracker(maneuver_config)

    env = build_stage4_env(visualize=False, verbose=False, target_mode=mode)
    try:
        rl_agent = SACAgent(build_ogm_agent_config(env))
        rl_agent.load(str(checkpoint_path), params_only=True)
        step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
        parking_agent = ParkingAgent(rl_agent, RsPlanner(step_ratio))

        records: list[dict[str, object]] = []
        with torch.no_grad():
            for case in load_cases(cases_path):
                seed = int(case["seed"])
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)

                obs = env.reset(int(case["hope_case_id"]), None, str(case["hope_level"]))
                parking_agent.reset()
                maneuver_tracker.reset_episode()
                done = False
                step_num = 0
                path_length = 0.0
                speeds: list[float] = []
                info: Mapping[str, object] = {"status": Status.CONTINUE, "path_to_dest": None}
                last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)

                while not done:
                    step_num += 1
                    action, _action_result = get_eval_action_with_maneuver_stability(
                        parking_agent=parking_agent,
                        tracker=maneuver_tracker,
                        obs=obs,
                    )
                    speeds.append(float(action[1]))
                    next_obs, _reward, done, info = env.step(action)
                    obs = next_obs
                    current_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
                    path_length += float(np.linalg.norm(np.array(last_xy) - np.array(current_xy)))
                    last_xy = current_xy
                    if info["path_to_dest"] is not None:
                        parking_agent.set_planner_path(info["path_to_dest"])

                status = info["status"]
                records.append(
                    {
                        "case_uid": str(case["case_uid"]),
                        "split": str(case["split"]),
                        "success": status == Status.ARRIVED,
                        "status": _case_status_name(status),
                        "step_num": step_num,
                        "gear_shifts": count_gear_shifts(speeds),
                        "path_length": path_length,
                    }
                )

        maneuver_snapshot = maneuver_tracker.snapshot()
        if maneuver_config.mode == MODE_OFF:
            result = build_eval_result(records, mode, target_mode_source)
        else:
            result = build_eval_result(
                records,
                mode,
                target_mode_source,
                maneuver_stability_config=maneuver_config,
                maneuver_stability_summary={
                    "config_source": maneuver_config_source,
                    "counters": maneuver_snapshot["counters"],
                },
            )
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(
            json.dumps(result, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        return result
    finally:
        env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 4 OGM SAC checkpoint on fixed cases.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
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
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    maneuver_config = None
    if args.maneuver_stability_mode != MANEUVER_STABILITY_MODE_AUTO:
        maneuver_config = ManeuverStabilityConfig(
            mode=args.maneuver_stability_mode,
            apply_to=args.maneuver_stability_apply_to,
            min_speed=args.maneuver_stability_min_speed,
            hold_steps=args.maneuver_stability_hold_steps,
            max_hold_speed=args.maneuver_stability_max_hold_speed,
        )
    evaluate_checkpoint(
        args.checkpoint,
        args.cases,
        args.output_json,
        target_mode=args.target_mode,
        maneuver_stability_config=maneuver_config,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
