from __future__ import annotations

import argparse
import atexit
import os
import sys
import time
from copy import deepcopy
from pathlib import Path
from shutil import copyfile

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
for import_root in (REPO_ROOT, SRC_ROOT):
    if str(import_root) not in sys.path:
        sys.path.insert(0, str(import_root))

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS  # noqa: E402
from tools.stage4.stage4_maneuver_stability import (  # noqa: E402
    APPLY_TO_MODES,
    MODE_OFF,
    MODES as MANEUVER_STABILITY_MODES,
    ManeuverStabilityConfig,
    ManeuverStabilityTracker,
)
from tools.stage4.stage4_target_transform import (  # noqa: E402
    TARGET_MODE_LEGACY_COSCOS,
    TARGET_MODES,
    build_stage4_observation_func,
    validate_target_mode,
)

FULL_STATE_SCHEMA_VERSION = 1


def _coerce_maneuver_stability_config(
    config: ManeuverStabilityConfig | dict[str, object] | None,
) -> ManeuverStabilityConfig:
    if isinstance(config, ManeuverStabilityConfig):
        return config
    return ManeuverStabilityConfig.from_mapping(config)


def _coerce_maneuver_stability_state(
    config: ManeuverStabilityConfig,
    state: dict[str, object] | None,
) -> dict[str, object]:
    if state is None:
        return ManeuverStabilityTracker(config).snapshot()
    restored = ManeuverStabilityTracker.from_snapshot(state)
    if restored.config.to_dict() != config.to_dict():
        raise ValueError(
            "maneuver_stability_state config %r does not match maneuver_stability_config %r."
            % (restored.config.to_dict(), config.to_dict())
        )
    return restored.snapshot()


def ensure_src_working_directory() -> Path:
    os.chdir(SRC_ROOT)
    return SRC_ROOT


def build_ogm_training_config(env):
    actor_params = deepcopy(ACTOR_CONFIGS)
    critic_params = deepcopy(CRITIC_CONFIGS)
    actor_params.update(
        {
            "n_modal": 3,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
            "action_mask_shape": env.observation_shape["action_mask"][0],
        }
    )
    critic_params.update(
        {
            "n_modal": 4,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
            "action_mask_shape": env.observation_shape["action_mask"][0],
            "input_action_dim": env.action_space.shape[0],
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
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }


def build_stage4_env(
    *,
    visualize: bool,
    verbose: bool,
    target_mode: str = TARGET_MODE_LEGACY_COSCOS,
):
    from env.car_parking_base import CarParking
    from env.env_wrapper import CarParkingWrapper

    mode = validate_target_mode(target_mode)
    raw_env = CarParking(
        fps=100,
        verbose=verbose,
        render_mode=None if visualize else "rgb_array",
        use_img_observation=False,
        use_lidar_observation=True,
        use_action_mask=True,
        use_ogm_observation=True,
    )
    return CarParkingWrapper(raw_env, observation_func=build_stage4_observation_func(raw_env, mode))


def build_ogm_save_path(timestamp: str, run_dir: str | None = None) -> Path:
    if run_dir is not None:
        return Path(run_dir).expanduser()
    return SRC_ROOT / "log" / "exp" / f"sac_ogm_{timestamp}"


def resolve_checkpoint_path(args: argparse.Namespace) -> str | None:
    resume_checkpoint = getattr(args, "resume_checkpoint", None)
    agent_ckpt = getattr(args, "agent_ckpt", None)
    resume_state = getattr(args, "resume_state", None)
    if resume_state and (resume_checkpoint or agent_ckpt):
        raise ValueError("--resume_state conflicts with --resume_checkpoint and legacy --agent_ckpt.")
    if resume_checkpoint and agent_ckpt and resume_checkpoint != agent_ckpt:
        raise ValueError("--resume_checkpoint and legacy --agent_ckpt differ; supply only one checkpoint path.")
    return resume_checkpoint or agent_ckpt


def validate_episode_window(start_episode: int, train_episode: int) -> None:
    if start_episode < 0:
        raise ValueError("start_episode must be greater than or equal to 0.")
    if start_episode >= train_episode:
        raise ValueError("start_episode must be less than train_episode; train_episode is the exclusive target.")


def validate_resume_checkpoint_requirement(start_episode: int, checkpoint_path: str | None) -> None:
    if start_episode > 0 and checkpoint_path is None:
        raise ValueError("start_episode > 0 requires --resume_checkpoint or legacy --agent_ckpt.")


def validate_resume_configuration(
    start_episode: int,
    checkpoint_path: str | None,
    resume_state_path: str | None,
    target_mode: str = TARGET_MODE_LEGACY_COSCOS,
) -> None:
    mode = validate_target_mode(target_mode)
    if mode != TARGET_MODE_LEGACY_COSCOS and checkpoint_path is not None:
        raise ValueError(
            "--target_mode corrected-cossin must start from a fresh run or matching full-state resume; "
            "do not pass --resume_checkpoint or legacy --agent_ckpt."
        )
    if mode != TARGET_MODE_LEGACY_COSCOS and start_episode > 0 and resume_state_path is None:
        raise ValueError(
            "--target_mode corrected-cossin continuation requires --resume_state from a matching corrected run."
        )
    if resume_state_path and checkpoint_path:
        raise ValueError("--resume_state conflicts with --resume_checkpoint and legacy --agent_ckpt.")
    if resume_state_path and start_episode <= 0:
        raise ValueError("--resume_state requires start_episode > 0.")
    if start_episode > 0 and checkpoint_path is None and resume_state_path is None:
        raise ValueError("start_episode > 0 requires --resume_state, --resume_checkpoint, or legacy --agent_ckpt.")


def apply_maneuver_stability_to_training_action(
    *,
    parking_agent,
    tracker: ManeuverStabilityTracker,
    obs,
    action,
    log_prob,
    source: str,
):
    action_result = tracker.apply(action, source=source)
    if not action_result.changed:
        return action, log_prob, action_result
    applied_action = action_result.applied_action
    return applied_action, parking_agent.get_log_prob(obs, applied_action), action_result


def iter_global_episodes(start_episode: int, train_episode: int) -> range:
    validate_episode_window(start_episode, train_episode)
    return range(start_episode, train_episode)


def periodic_checkpoint_name(global_episode: int) -> str:
    return "SAC_%s.pt" % global_episode


def stage4_state_name(global_episode: int) -> str:
    return "stage4_state_%s.pt" % global_episode


def stage4_latest_state_name() -> str:
    return "stage4_state_latest.pt"


def close_training_resources(writer=None, env=None) -> None:
    if writer is not None:
        if hasattr(writer, "flush"):
            writer.flush()
        if hasattr(writer, "close"):
            writer.close()
    if env is not None and hasattr(env, "close"):
        env.close()


def iter_verbose_history_rows(case_id_list, reward_list, reward_info_list, limit: int = 10):
    available = min(limit, len(case_id_list), len(reward_list), len(reward_info_list))
    start = len(reward_list) - available
    for idx in range(start, len(reward_list)):
        yield case_id_list[idx], reward_list[idx], reward_info_list[idx]


def format_best_success_log(global_episode: int, raw_best_success_rate) -> str:
    return "epoch: %s, success rate: %s %s %s %s" % (
        global_episode,
        raw_best_success_rate[0],
        raw_best_success_rate[1],
        raw_best_success_rate[2],
        raw_best_success_rate[3],
    )


class SceneChoose:
    def __init__(self) -> None:
        self.scene_types = {
            0: "Normal",
            1: "Complex",
            2: "Extrem",
            3: "dlp",
        }
        self.target_success_rate = np.array([0.95, 0.95, 0.9, 0.99])
        self.success_record = {}
        for scene_name in self.scene_types:
            self.success_record[scene_name] = []
        self.scene_record = []
        self.history_horizon = 200

    def choose_case(self):
        if len(self.scene_record) < self.history_horizon:
            scene_chosen = self._choose_case_uniform()
        else:
            if np.random.random() > 0.5:
                scene_chosen = self._choose_case_worst_perform()
            else:
                scene_chosen = self._choose_case_uniform()
        self.scene_record.append(scene_chosen)
        return self.scene_types[scene_chosen]

    def update_success_record(self, success: int):
        self.success_record[self.scene_record[-1]].append(success)

    def _choose_case_uniform(self):
        case_count = np.zeros(len(self.scene_types))
        for i in range(min(len(self.scene_record), self.history_horizon)):
            scene_id = self.scene_record[-(i + 1)]
            case_count[scene_id] += 1
        return np.argmin(case_count)

    def _choose_case_worst_perform(self):
        success_rate = []
        for i in self.success_record.keys():
            idx = int(i)
            recent_success_record = self.success_record[idx][-min(250, len(self.success_record[idx])):]
            success_rate.append(np.sum(recent_success_record) / len(recent_success_record))
        fail_rate = self.target_success_rate - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.01, 1)
        fail_rate = fail_rate / np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)


class DlpCaseChoose:
    def __init__(self) -> None:
        self.dlp_case_num = 248
        self.case_record = []
        self.case_success_rate = {}
        for i in range(self.dlp_case_num):
            self.case_success_rate[str(i)] = []
        self.horizon = 500

    def choose_case(self):
        if np.random.random() < 0.2 or len(self.case_record) < self.horizon:
            return np.random.randint(0, self.dlp_case_num)
        success_rate = []
        for i in range(self.dlp_case_num):
            idx = str(i)
            if len(self.case_success_rate[idx]) <= 1:
                success_rate.append(0)
            else:
                recent_success_record = self.case_success_rate[idx][-min(10, len(self.case_success_rate[idx])):]
                success_rate.append(np.sum(recent_success_record) / len(recent_success_record))
        fail_rate = 1 - np.array(success_rate)
        fail_rate = np.clip(fail_rate, 0.005, 1)
        fail_rate = fail_rate / np.sum(fail_rate)
        return np.random.choice(np.arange(len(fail_rate)), p=fail_rate)

    def update_success_record(self, success: int, case_id: int):
        self.case_success_rate[str(case_id)].append(success)
        self.case_record.append(case_id)


def _torch_load(path: Path, map_location=None):
    import torch

    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _unwrap_rl_agent(parking_agent):
    return getattr(parking_agent, "agent", parking_agent)


def _capture_agent_state(rl_agent) -> dict[str, object] | None:
    if not hasattr(rl_agent, "check_list"):
        return None

    checkpoint: dict[str, object] = {}
    for name, item, save_state_dict in rl_agent.check_list:
        checkpoint[name] = item.state_dict() if save_state_dict else deepcopy(item)

    if getattr(getattr(rl_agent, "configs", None), "dist_type", None) == "gaussian" and hasattr(rl_agent, "log_std"):
        checkpoint["log"] = rl_agent.log_std.detach().clone()
    if hasattr(rl_agent, "state_normalize"):
        checkpoint["state_norm"] = deepcopy(rl_agent.state_normalize)

    optimizer_state = {}
    for attr in ("actor_optimizer", "critic_optimizer1", "critic_optimizer2", "log_alpha_optimizer"):
        optimizer = getattr(rl_agent, attr, None)
        if optimizer is not None:
            optimizer_state[attr] = optimizer.state_dict()
    checkpoint["optimizer_state"] = optimizer_state
    return checkpoint


def _restore_agent_state(rl_agent, agent_state: dict[str, object] | None) -> None:
    if not agent_state or not hasattr(rl_agent, "check_list"):
        return

    for name, item, save_state_dict in rl_agent.check_list:
        if name not in agent_state:
            continue
        if save_state_dict:
            item.load_state_dict(agent_state[name])
        elif name in {"log_alpha", "log_std"} and hasattr(rl_agent, name):
            target = getattr(rl_agent, name)
            source = agent_state[name]
            if hasattr(source, "to"):
                source = source.to(target.device)
            target.data.copy_(source)
        else:
            setattr(rl_agent, name, agent_state[name])

    if "log" in agent_state and hasattr(rl_agent, "log_std"):
        log_std = agent_state["log"]
        if hasattr(log_std, "to"):
            log_std = log_std.to(rl_agent.log_std.device)
        rl_agent.log_std.data.copy_(log_std)
    if "state_norm" in agent_state:
        rl_agent.state_normalize = agent_state["state_norm"]

    optimizer_state = agent_state.get("optimizer_state", {})
    for attr, state_dict in optimizer_state.items():
        optimizer = getattr(rl_agent, attr, None)
        if optimizer is not None:
            optimizer.load_state_dict(state_dict)


def _capture_rng_state(torch_module) -> dict[str, object]:
    cuda_rng_state = None
    if torch_module.cuda.is_available():
        cuda_rng_state = torch_module.cuda.get_rng_state_all()
    return {
        "numpy": np.random.get_state(),
        "torch": torch_module.get_rng_state(),
        "cuda": cuda_rng_state,
    }


def _restore_rng_state(rng_state: dict[str, object], torch_module) -> None:
    if "numpy" in rng_state:
        np.random.set_state(rng_state["numpy"])
    if "torch" in rng_state:
        torch_rng_state = rng_state["torch"]
        if hasattr(torch_rng_state, "cpu"):
            torch_rng_state = torch_rng_state.cpu()
        torch_module.set_rng_state(torch_rng_state)
    cuda_rng_state = rng_state.get("cuda")
    if cuda_rng_state is not None and torch_module.cuda.is_available():
        cuda_rng_state = [
            state.cpu() if hasattr(state, "cpu") else state for state in cuda_rng_state
        ]
        torch_module.cuda.set_rng_state_all(cuda_rng_state)


def _atomic_torch_save(payload: dict[str, object], final_path: Path, torch_module) -> None:
    final_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = final_path.with_name(
        ".%s.%s.%s.tmp" % (final_path.name, os.getpid(), time.time_ns())
    )
    try:
        torch_module.save(payload, temp_path)
        temp_path.replace(final_path)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _capture_scene_chooser_state(scene_chooser: SceneChoose) -> dict[str, object]:
    return {
        "scene_record": list(scene_chooser.scene_record),
        "success_record": {int(key): list(value) for key, value in scene_chooser.success_record.items()},
    }


def _restore_scene_chooser_state(scene_chooser: SceneChoose, state: dict[str, object]) -> None:
    scene_chooser.scene_record = list(state["scene_record"])
    scene_chooser.success_record = {int(key): list(value) for key, value in state["success_record"].items()}


def _capture_dlp_case_chooser_state(dlp_case_chooser: DlpCaseChoose) -> dict[str, object]:
    return {
        "case_record": list(dlp_case_chooser.case_record),
        "case_success_rate": {str(key): list(value) for key, value in dlp_case_chooser.case_success_rate.items()},
    }


def _restore_dlp_case_chooser_state(dlp_case_chooser: DlpCaseChoose, state: dict[str, object]) -> None:
    dlp_case_chooser.case_record = list(state["case_record"])
    dlp_case_chooser.case_success_rate = {str(key): list(value) for key, value in state["case_success_rate"].items()}


def build_stage4_state_payload(
    *,
    global_episode: int,
    parking_agent,
    total_step_num: int,
    scene_chooser: SceneChoose,
    dlp_case_chooser: DlpCaseChoose,
    best_success_rate,
    reward_list,
    reward_per_state_list,
    reward_info_list,
    case_id_list,
    succ_record,
    sac_checkpoint_path: Path | None = None,
    target_mode: str = TARGET_MODE_LEGACY_COSCOS,
    maneuver_stability_config: ManeuverStabilityConfig | dict[str, object] | None = None,
    maneuver_stability_state: dict[str, object] | None = None,
    torch_module=None,
) -> dict[str, object]:
    if torch_module is None:
        import torch as torch_module

    rl_agent = _unwrap_rl_agent(parking_agent)
    mode = validate_target_mode(target_mode)
    maneuver_config = _coerce_maneuver_stability_config(maneuver_stability_config)
    maneuver_state = _coerce_maneuver_stability_state(maneuver_config, maneuver_stability_state)
    return {
        "schema_version": FULL_STATE_SCHEMA_VERSION,
        "target_mode": mode,
        "maneuver_stability_config": maneuver_config.to_dict(),
        "maneuver_stability_state": maneuver_state,
        "global_episode": int(global_episode),
        "sac_checkpoint_path": str(sac_checkpoint_path) if sac_checkpoint_path is not None else None,
        "sac_checkpoint_name": sac_checkpoint_path.name if sac_checkpoint_path is not None else None,
        "agent_state": _capture_agent_state(rl_agent),
        "replay_memory": deepcopy(rl_agent.memory),
        "total_step_num": int(total_step_num),
        "rng_state": _capture_rng_state(torch_module),
        "scene_chooser": _capture_scene_chooser_state(scene_chooser),
        "dlp_case_chooser": _capture_dlp_case_chooser_state(dlp_case_chooser),
        "best_success_rate": list(best_success_rate),
        "reward_list": list(reward_list),
        "reward_per_state_list": list(reward_per_state_list),
        "reward_info_list": list(reward_info_list),
        "case_id_list": list(case_id_list),
        "succ_record": list(succ_record),
        "agent_debug_history": {
            "actor_loss_list": list(getattr(rl_agent, "actor_loss_list", [])),
            "critic_loss_list": list(getattr(rl_agent, "critic_loss_list", [])),
        },
    }


def save_stage4_state(
    *,
    save_dir: Path,
    global_episode: int,
    parking_agent,
    total_step_num: int,
    scene_chooser: SceneChoose,
    dlp_case_chooser: DlpCaseChoose,
    best_success_rate,
    reward_list,
    reward_per_state_list,
    reward_info_list,
    case_id_list,
    succ_record,
    sac_checkpoint_path: Path | None = None,
    target_mode: str = TARGET_MODE_LEGACY_COSCOS,
    maneuver_stability_config: ManeuverStabilityConfig | dict[str, object] | None = None,
    maneuver_stability_state: dict[str, object] | None = None,
    torch_module=None,
) -> Path:
    if torch_module is None:
        import torch as torch_module

    save_dir.mkdir(parents=True, exist_ok=True)
    payload = build_stage4_state_payload(
        global_episode=global_episode,
        parking_agent=parking_agent,
        total_step_num=total_step_num,
        scene_chooser=scene_chooser,
        dlp_case_chooser=dlp_case_chooser,
        best_success_rate=best_success_rate,
        reward_list=reward_list,
        reward_per_state_list=reward_per_state_list,
        reward_info_list=reward_info_list,
        case_id_list=case_id_list,
        succ_record=succ_record,
        sac_checkpoint_path=sac_checkpoint_path,
        target_mode=target_mode,
        maneuver_stability_config=maneuver_stability_config,
        maneuver_stability_state=maneuver_stability_state,
        torch_module=torch_module,
    )
    state_path = save_dir / stage4_state_name(global_episode)
    _atomic_torch_save(payload, state_path, torch_module)
    _atomic_torch_save(payload, save_dir / stage4_latest_state_name(), torch_module)
    return state_path


def load_stage4_state(
    *,
    state_path: str | Path,
    parking_agent,
    scene_chooser: SceneChoose,
    dlp_case_chooser: DlpCaseChoose,
    expected_target_mode: str | None = None,
    expected_maneuver_stability_config: ManeuverStabilityConfig | dict[str, object] | None = None,
    torch_module=None,
    map_location=None,
) -> dict[str, object]:
    if torch_module is None:
        import torch as torch_module

    state = _torch_load(Path(state_path), map_location=map_location)
    if state.get("schema_version") != FULL_STATE_SCHEMA_VERSION:
        raise ValueError(
            "Unsupported Stage 4 state schema version: %s" % state.get("schema_version")
        )
    target_mode = validate_target_mode(state.get("target_mode", TARGET_MODE_LEGACY_COSCOS))
    if expected_target_mode is not None:
        expected_mode = validate_target_mode(expected_target_mode)
        if target_mode != expected_mode:
            raise ValueError(
                "Stage 4 state target_mode %r does not match requested target_mode %r."
                % (target_mode, expected_mode)
            )

    maneuver_config = ManeuverStabilityConfig.from_mapping(state.get("maneuver_stability_config"))
    maneuver_state = _coerce_maneuver_stability_state(
        maneuver_config,
        state.get("maneuver_stability_state"),
    )
    if expected_maneuver_stability_config is not None:
        expected_maneuver_config = _coerce_maneuver_stability_config(expected_maneuver_stability_config)
        if maneuver_config.to_dict() != expected_maneuver_config.to_dict():
            if maneuver_config.mode != MODE_OFF or expected_maneuver_config.mode != MODE_OFF:
                raise ValueError(
                    "Stage 4 state maneuver_stability_config %r does not match requested "
                    "maneuver_stability_config %r."
                    % (maneuver_config.to_dict(), expected_maneuver_config.to_dict())
                )

    rl_agent = _unwrap_rl_agent(parking_agent)
    _restore_agent_state(rl_agent, state.get("agent_state"))
    rl_agent.memory = deepcopy(state["replay_memory"])
    debug_history = state.get("agent_debug_history", {})
    rl_agent.actor_loss_list = list(debug_history.get("actor_loss_list", []))
    rl_agent.critic_loss_list = list(debug_history.get("critic_loss_list", []))
    _restore_scene_chooser_state(scene_chooser, state["scene_chooser"])
    _restore_dlp_case_chooser_state(dlp_case_chooser, state["dlp_case_chooser"])
    _restore_rng_state(state["rng_state"], torch_module)

    return {
        "schema_version": state["schema_version"],
        "target_mode": target_mode,
        "maneuver_stability_config": maneuver_config.to_dict(),
        "maneuver_stability_state": maneuver_state,
        "global_episode": state["global_episode"],
        "sac_checkpoint_path": state.get("sac_checkpoint_path"),
        "sac_checkpoint_name": state.get("sac_checkpoint_name"),
        "total_step_num": state["total_step_num"],
        "best_success_rate": list(state["best_success_rate"]),
        "reward_list": list(state["reward_list"]),
        "reward_per_state_list": list(state["reward_per_state_list"]),
        "reward_info_list": list(state["reward_info_list"]),
        "case_id_list": list(state["case_id_list"]),
        "succ_record": list(state["succ_record"]),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Stage 4 HOPE SAC with OGM policy observations.")
    parser.add_argument("--agent_ckpt", type=str, default=None)
    parser.add_argument("--resume_checkpoint", type=str, default=None)
    parser.add_argument("--resume_state", type=str, default=None)
    parser.add_argument("--start_episode", type=int, default=0)
    parser.add_argument("--train_episode", type=int, default=100000)
    parser.add_argument("--eval_episode", type=int, default=2000)
    parser.add_argument("--verbose", type=bool, default=True)
    parser.add_argument("--visualize", type=bool, default=True)
    parser.add_argument("--run_dir", type=str, default=None)
    parser.add_argument("--target_mode", choices=TARGET_MODES, default=TARGET_MODE_LEGACY_COSCOS)
    parser.add_argument("--maneuver_stability_mode", choices=MANEUVER_STABILITY_MODES, default=MODE_OFF)
    parser.add_argument("--maneuver_stability_apply_to", choices=APPLY_TO_MODES, default="rl")
    parser.add_argument("--maneuver_stability_min_speed", type=float, default=1e-6)
    parser.add_argument("--maneuver_stability_hold_steps", type=int, default=1)
    parser.add_argument("--maneuver_stability_max_hold_speed", type=float, default=None)
    return parser.parse_args()


def main() -> int:
    ensure_src_working_directory()

    import matplotlib.pyplot as plt
    import torch
    from torch.utils.tensorboard import SummaryWriter

    from configs import SEED, VALID_SPEED
    from env.vehicle import Status
    from evaluation.eval_utils import eval
    from model.agent.parking_agent import ParkingAgent, RsPlanner
    from model.agent.sac_agent import SACAgent as SAC

    args = _parse_args()
    target_mode = validate_target_mode(args.target_mode)
    try:
        maneuver_stability_config = ManeuverStabilityConfig(
            mode=args.maneuver_stability_mode,
            apply_to=args.maneuver_stability_apply_to,
            min_speed=args.maneuver_stability_min_speed,
            hold_steps=args.maneuver_stability_hold_steps,
            max_hold_speed=args.maneuver_stability_max_hold_speed,
        )
        validate_episode_window(args.start_episode, args.train_episode)
        checkpoint_path = resolve_checkpoint_path(args)
        validate_resume_configuration(args.start_episode, checkpoint_path, args.resume_state, target_mode=target_mode)
    except ValueError as exc:
        print("error: %s" % exc, file=sys.stderr)
        return 2
    verbose = args.verbose

    env = build_stage4_env(
        visualize=args.visualize,
        verbose=verbose,
        target_mode=target_mode,
    )
    writer = None
    resources = {"writer": writer, "env": env}

    def cleanup_resources() -> None:
        close_training_resources(resources["writer"], resources["env"])

    atexit.register(cleanup_resources)
    scene_chooser = SceneChoose()
    dlp_case_chooser = DlpCaseChoose()

    print("Stage 4 OGM runner policy inputs: target + action_mask + ogm")
    print("Stage 4 OGM target mode: %s" % target_mode)
    print("Stage 4 maneuver stability mode: %s" % maneuver_stability_config.mode)

    current_time = time.localtime()
    timestamp = time.strftime("%Y%m%d_%H%M%S", current_time)
    save_path = build_ogm_save_path(timestamp, run_dir=args.run_dir)
    save_path.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(str(save_path))
    resources["writer"] = writer
    copyfile(SRC_ROOT / "configs.py", save_path / "configs.txt")
    print("You can track the training process by command 'tensorboard --log-dir %s'" % save_path)

    seed = SEED
    env.action_space.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    configs = build_ogm_training_config(env)
    print("observation_space:", env.observation_space)

    rl_agent = SAC(configs)
    if checkpoint_path is not None:
        rl_agent.load(checkpoint_path, params_only=True)
        print("load pre-trained model!")

    step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
    rs_planner = RsPlanner(step_ratio)
    parking_agent = ParkingAgent(rl_agent, rs_planner)
    maneuver_tracker = ManeuverStabilityTracker(maneuver_stability_config)

    reward_list = []
    reward_per_state_list = []
    reward_info_list = []
    case_id_list = []
    succ_record = []
    total_step_num = 0
    best_success_rate = [0, 0, 0, 0]
    reward_history_start_episode = args.start_episode

    if args.resume_state is not None:
        restored_state = load_stage4_state(
            state_path=args.resume_state,
            parking_agent=parking_agent,
            scene_chooser=scene_chooser,
            dlp_case_chooser=dlp_case_chooser,
            expected_target_mode=target_mode,
            expected_maneuver_stability_config=maneuver_stability_config,
            torch_module=torch,
            map_location=getattr(rl_agent, "device", None),
        )
        expected_start_episode = int(restored_state["global_episode"]) + 1
        if args.start_episode != expected_start_episode:
            print(
                "error: --start_episode must be %s when resuming from %s."
                % (expected_start_episode, args.resume_state),
                file=sys.stderr,
            )
            return 2
        total_step_num = int(restored_state["total_step_num"])
        best_success_rate = list(restored_state["best_success_rate"])
        reward_list = list(restored_state["reward_list"])
        reward_per_state_list = list(restored_state["reward_per_state_list"])
        reward_info_list = list(restored_state["reward_info_list"])
        case_id_list = list(restored_state["case_id_list"])
        succ_record = list(restored_state["succ_record"])
        maneuver_tracker = ManeuverStabilityTracker.from_snapshot(restored_state["maneuver_stability_state"])
        reward_history_start_episode = max(0, args.start_episode - len(reward_list))
        print("load Stage 4 full-state snapshot!")

    last_completed_episode = None
    for i in iter_global_episodes(args.start_episode, args.train_episode):
        last_completed_episode = i
        scene_chosen = scene_chooser.choose_case()
        if scene_chosen == "dlp":
            case_id = dlp_case_chooser.choose_case()
        else:
            case_id = None
        obs = env.reset(case_id, None, scene_chosen)
        parking_agent.reset()
        maneuver_tracker.reset_episode()
        case_id_list.append(env.map.case_id)
        done = False
        total_reward = 0
        step_num = 0
        reward_info = []
        while not done:
            step_num += 1
            total_step_num += 1
            was_executing_rs = bool(getattr(parking_agent, "executing_rs", False))
            if total_step_num <= parking_agent.configs.memory_size and not was_executing_rs:
                action = env.action_space.sample()
                log_prob = parking_agent.get_log_prob(obs, action)
                action_source = "RL"
            else:
                action, log_prob = parking_agent.get_action(obs)
                action_source = "RS" if was_executing_rs else "RL"

            action, log_prob, _maneuver_result = apply_maneuver_stability_to_training_action(
                parking_agent=parking_agent,
                tracker=maneuver_tracker,
                obs=obs,
                action=action,
                log_prob=log_prob,
                source=action_source,
            )
            next_obs, reward, done, info = env.step(action)
            reward_info.append(list(info["reward_info"].values()))
            total_reward += reward
            reward_per_state_list.append(reward)
            parking_agent.push_memory((obs, action, reward, done, log_prob, next_obs))
            obs = next_obs
            if total_step_num > parking_agent.configs.memory_size and total_step_num % 10 == 0:
                actor_loss, critic_loss = parking_agent.update()
                if total_step_num % 200 == 0:
                    writer.add_scalar("actor_loss", actor_loss, i)
                    writer.add_scalar("critic_loss", critic_loss, i)

            if info["path_to_dest"] is not None:
                parking_agent.set_planner_path(info["path_to_dest"])

            if done:
                if info["status"] == Status.ARRIVED:
                    succ_record.append(1)
                    scene_chooser.update_success_record(1)
                    if scene_chosen == "dlp":
                        dlp_case_chooser.update_success_record(1, case_id)
                else:
                    succ_record.append(0)
                    scene_chooser.update_success_record(0)
                    if scene_chosen == "dlp":
                        dlp_case_chooser.update_success_record(0, case_id)

        writer.add_scalar("total_reward", total_reward, i)
        writer.add_scalar("avg_reward", np.mean(reward_per_state_list[-1000:]), i)
        writer.add_scalar("action_std0", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[0], i)
        writer.add_scalar("action_std1", parking_agent.log_std.detach().cpu().numpy().reshape(-1)[1], i)
        writer.add_scalar("alpha", parking_agent.alpha.detach().cpu().numpy().reshape(-1)[0], i)
        maneuver_counters = maneuver_tracker.snapshot()["counters"]
        writer.add_scalar("maneuver_interventions", maneuver_counters["interventions"], i)
        writer.add_scalar("maneuver_speed_sign_flips", maneuver_counters["speed_sign_flips"], i)
        writer.add_scalar("maneuver_would_change", maneuver_counters["would_change"], i)
        for type_id in scene_chooser.scene_types:
            writer.add_scalar(
                "success_rate_%s" % scene_chooser.scene_types[type_id],
                np.mean(scene_chooser.success_record[type_id][-100:]),
                i,
            )
        writer.add_scalar("step_num", step_num, i)
        reward_list.append(total_reward)
        reward_info = np.sum(np.array(reward_info), axis=0)
        reward_info = np.round(reward_info, 2)
        reward_info_list.append(list(reward_info))

        if verbose and i % 10 == 0 and i > 0:
            print("success rate:", np.sum(succ_record), "/", len(succ_record))
            print(
                parking_agent.log_std.detach().cpu().numpy().reshape(-1),
                parking_agent.alpha.detach().cpu().numpy().reshape(-1),
            )
            print("episode:%s  average reward:%s" % (i, np.mean(reward_list[-50:])))
            print(np.mean(parking_agent.actor_loss_list[-100:]), np.mean(parking_agent.critic_loss_list[-100:]))
            print("time_cost ,rs_dist_reward ,dist_reward ,angle_reward ,box_union_reward")
            for case_id_history, reward_history, reward_info_history in iter_verbose_history_rows(
                case_id_list,
                reward_list,
                reward_info_list,
            ):
                print(case_id_history, reward_history, reward_info_history)
            print("")

        for type_id in scene_chooser.scene_types:
            success_rate_normal = np.mean(scene_chooser.success_record[0][-100:])
            success_rate_complex = np.mean(scene_chooser.success_record[1][-100:])
            success_rate_extreme = np.mean(scene_chooser.success_record[2][-100:])
            success_rate_dlp = np.mean(scene_chooser.success_record[3][-100:])
        if (
            success_rate_normal >= best_success_rate[0]
            and success_rate_complex >= best_success_rate[1]
            and success_rate_extreme >= best_success_rate[2]
            and success_rate_dlp >= best_success_rate[3]
            and i > 100
        ):
            raw_best_success_rate = np.array(
                [success_rate_normal, success_rate_complex, success_rate_extreme, success_rate_dlp]
            )
            best_success_rate = list(np.minimum(raw_best_success_rate, scene_chooser.target_success_rate))
            parking_agent.save(str(save_path / "SAC_best.pt"), params_only=True)
            with (save_path / "best.txt").open("w") as f_best_log:
                f_best_log.write(format_best_success_log(i, raw_best_success_rate))

        if (i + 1) % 2000 == 0:
            sac_checkpoint_path = save_path / periodic_checkpoint_name(i)
            parking_agent.save(str(sac_checkpoint_path), params_only=True)
            save_stage4_state(
                save_dir=save_path,
                global_episode=i,
                parking_agent=parking_agent,
                total_step_num=total_step_num,
                scene_chooser=scene_chooser,
                dlp_case_chooser=dlp_case_chooser,
                best_success_rate=best_success_rate,
                reward_list=reward_list,
                reward_per_state_list=reward_per_state_list,
                reward_info_list=reward_info_list,
                case_id_list=case_id_list,
                succ_record=succ_record,
                sac_checkpoint_path=sac_checkpoint_path,
                target_mode=target_mode,
                maneuver_stability_config=maneuver_stability_config,
                maneuver_stability_state=maneuver_tracker.snapshot(),
                torch_module=torch,
            )

        if verbose and i % 20 == 0:
            episodes = [reward_history_start_episode + j for j in range(len(reward_list))]
            mean_reward = [np.mean(reward_list[max(0, j - 50) : j + 1]) for j in range(len(reward_list))]
            plt.plot(episodes, reward_list)
            plt.plot(episodes, mean_reward)
            plt.xlabel("episodes")
            plt.ylabel("reward")
            f = plt.gcf()
            f.savefig(str(save_path / "reward.png"))
            f.clear()

    if last_completed_episode is not None and (last_completed_episode + 1) % 2000 != 0:
        save_stage4_state(
            save_dir=save_path,
            global_episode=last_completed_episode,
            parking_agent=parking_agent,
            total_step_num=total_step_num,
            scene_chooser=scene_chooser,
            dlp_case_chooser=dlp_case_chooser,
            best_success_rate=best_success_rate,
            reward_list=reward_list,
            reward_per_state_list=reward_per_state_list,
            reward_info_list=reward_info_list,
            case_id_list=case_id_list,
            succ_record=succ_record,
            target_mode=target_mode,
            maneuver_stability_config=maneuver_stability_config,
            maneuver_stability_state=maneuver_tracker.snapshot(),
            torch_module=torch,
        )

    eval_episode = args.eval_episode
    choose_action = False
    with torch.no_grad():
        env.set_level("dlp")
        log_path = save_path / "dlp"
        log_path.mkdir(parents=True, exist_ok=True)
        eval(env, parking_agent, episode=eval_episode, log_path=str(log_path), post_proc_action=choose_action)

        env.set_level("Extrem")
        log_path = save_path / "extreme"
        log_path.mkdir(parents=True, exist_ok=True)
        eval(env, parking_agent, episode=eval_episode, log_path=str(log_path), post_proc_action=choose_action)

        env.set_level("Complex")
        log_path = save_path / "complex"
        log_path.mkdir(parents=True, exist_ok=True)
        eval(env, parking_agent, episode=eval_episode, log_path=str(log_path), post_proc_action=choose_action)

        env.set_level("Normal")
        log_path = save_path / "normalize"
        log_path.mkdir(parents=True, exist_ok=True)
        eval(env, parking_agent, episode=eval_episode, log_path=str(log_path), post_proc_action=choose_action)

    close_training_resources(writer, env)
    resources["writer"] = None
    resources["env"] = None
    atexit.unregister(cleanup_resources)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
