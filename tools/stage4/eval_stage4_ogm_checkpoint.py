from __future__ import annotations

import argparse
import json
import os
import random
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))


def count_gear_shifts(speeds: Iterable[float]) -> int:
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


def summarize_eval_records(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split"])].append(record)

    summary: dict[str, dict[str, float]] = {}
    for split, split_records in grouped.items():
        successes = [record for record in split_records if bool(record["success"])]
        psr = len(successes) / len(split_records) if split_records else 0.0
        if successes:
            angs = sum(float(record["gear_shifts"]) for record in successes) / len(successes)
            pl = sum(float(record["path_length"]) for record in successes) / len(successes)
        else:
            angs = float("inf")
            pl = float("inf")
        summary[split] = {"psr": psr, "angs": angs, "pl": pl}
    return summary


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


def evaluate_checkpoint(checkpoint_path: str | Path, cases_path: str | Path, output_json: str | Path) -> dict[str, object]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")

    import torch
    from configs import VALID_SPEED
    from env.car_parking_base import CarParking
    from env.env_wrapper import CarParkingWrapper
    from env.vehicle import Status
    from model.agent.parking_agent import ParkingAgent, RsPlanner
    from model.agent.sac_agent import SACAgent
    from tools.stage4.stage4_ogm_cases import load_cases

    checkpoint_path = Path(checkpoint_path)
    cases_path = Path(cases_path)
    output_json = Path(output_json)

    raw_env = CarParking(
        render_mode="rgb_array",
        verbose=False,
        use_img_observation=False,
        use_lidar_observation=True,
        use_action_mask=True,
        use_ogm_observation=True,
    )
    env = CarParkingWrapper(raw_env)
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
                done = False
                step_num = 0
                path_length = 0.0
                speeds: list[float] = []
                info: Mapping[str, object] = {"status": Status.CONTINUE, "path_to_dest": None}
                last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)

                while not done:
                    step_num += 1
                    action, _ = parking_agent.get_action(obs)
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

        result = {"records": records, "summary": summarize_eval_records(records)}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return result
    finally:
        raw_env.close()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a Stage 4 OGM SAC checkpoint on fixed cases.")
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    evaluate_checkpoint(args.checkpoint, args.cases, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
