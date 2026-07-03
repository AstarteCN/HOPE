from __future__ import annotations

import argparse
import os
import sys
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest import mock

import numpy as np
import torch
import tools.stage4.train_HOPE_sac_ogm as stage4_runner

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS  # noqa: E402
from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
from model.agent.sac_agent import SACAgent  # noqa: E402
from model.replay_memory import ReplayMemory  # noqa: E402
from tools.stage4.stage4_maneuver_stability import (  # noqa: E402
    MODE_DIAGNOSTIC,
    MODE_DIRECTION_HOLD,
    MODE_OFF,
    ManeuverStabilityConfig,
    ManeuverStabilityTracker,
)
from tools.stage4.stage4_target_transform import (  # noqa: E402
    TARGET_MODE_CORRECTED_COSSIN,
    TARGET_MODE_LEGACY_COSCOS,
)
from tools.stage4.train_HOPE_sac_ogm import (  # noqa: E402
    DlpCaseChoose,
    FULL_STATE_SCHEMA_VERSION,
    SceneChoose,
    build_stage4_env,
    build_ogm_save_path,
    build_ogm_training_config,
    ensure_src_working_directory,
    format_best_success_log,
    iter_global_episodes,
    iter_verbose_history_rows,
    load_stage4_state,
    periodic_checkpoint_name,
    resolve_checkpoint_path,
    save_stage4_state,
    stage4_latest_state_name,
    stage4_state_name,
    validate_episode_window,
    validate_resume_configuration,
    validate_resume_checkpoint_requirement,
)


class _FakeInnerAgent:
    def __init__(self, memory: ReplayMemory) -> None:
        self.memory = memory
        self.actor_loss_list = [1.25]
        self.critic_loss_list = [2.5]


class _FakeParkingAgent:
    def __init__(self, memory: ReplayMemory) -> None:
        self.agent = _FakeInnerAgent(memory)


class _FakeCuda:
    def is_available(self) -> bool:
        return False


class _FakeTorchForSaving:
    def __init__(self) -> None:
        self.cuda = _FakeCuda()
        self.saved_paths = []

    def get_rng_state(self):
        return "fake-torch-rng-state"

    def save(self, payload, path) -> None:
        path = Path(path)
        self.saved_paths.append(path)
        path.write_text("fake stage4 state", encoding="utf-8")


class _RecordingWriter:
    def __init__(self) -> None:
        self.calls = []

    def flush(self) -> None:
        self.calls.append("flush")

    def close(self) -> None:
        self.calls.append("close")


class _RecordingEnv:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _RecordingLogProbParkingAgent:
    def __init__(self) -> None:
        self.log_prob_calls = []

    def get_log_prob(self, obs, action):
        self.log_prob_calls.append((obs, np.asarray(action, dtype=float).copy()))
        return "recomputed-log-prob"


class Stage4RunnerConfigTests(unittest.TestCase):
    def test_build_ogm_training_config_uses_only_target_action_mask_and_ogm(self) -> None:
        actor_configs_before = deepcopy(ACTOR_CONFIGS)
        critic_configs_before = deepcopy(CRITIC_CONFIGS)
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
            config = build_ogm_training_config(env)
            self.assertEqual(set(config["observation_shape"].keys()), {"target", "action_mask", "ogm"})
            self.assertIsNone(config["actor_layers"]["lidar_shape"])
            self.assertIsNone(config["actor_layers"]["img_shape"])
            self.assertIsNone(config["critic_layers"]["lidar_shape"])
            self.assertIsNone(config["critic_layers"]["img_shape"])
            self.assertEqual(config["actor_layers"]["ogm_shape"], (2, 64, 64))
            self.assertEqual(config["critic_layers"]["ogm_shape"], (2, 64, 64))
            self.assertEqual(config["actor_layers"]["action_mask_shape"], env.observation_shape["action_mask"][0])
            self.assertEqual(config["critic_layers"]["action_mask_shape"], env.observation_shape["action_mask"][0])
            self.assertEqual(config["actor_layers"]["n_modal"], 3)
            self.assertEqual(config["critic_layers"]["n_modal"], 4)
            self.assertEqual(ACTOR_CONFIGS, actor_configs_before)
            self.assertEqual(CRITIC_CONFIGS, critic_configs_before)

            agent = SACAgent(config)
            self.assertFalse(hasattr(agent.actor_net, "embed_img"))
            self.assertFalse(hasattr(agent.critic_net1.net, "embed_img"))
        finally:
            raw_env.close()

    def test_build_stage4_env_defaults_to_legacy_target_mode(self) -> None:
        env = build_stage4_env(visualize=False, verbose=False)
        try:
            obs = env.reset()

            self.assertAlmostEqual(obs["target"][3], obs["target"][4])
        finally:
            env.close()

    def test_build_stage4_env_corrected_target_mode_uses_signed_relative_heading(self) -> None:
        env = build_stage4_env(
            visualize=False,
            verbose=False,
            target_mode=TARGET_MODE_CORRECTED_COSSIN,
        )
        try:
            obs = env.reset()
            rel_dest_heading = env.env.map.dest.heading - env.env.vehicle.state.heading

            self.assertAlmostEqual(obs["target"][3], np.cos(rel_dest_heading))
            self.assertAlmostEqual(obs["target"][4], np.sin(rel_dest_heading))
        finally:
            env.close()

    def test_build_ogm_save_path_uses_src_log_exp_for_monitor_compatibility(self) -> None:
        save_path = build_ogm_save_path("20990101_000000")
        self.assertEqual(save_path, SRC_ROOT / "log" / "exp" / "sac_ogm_20990101_000000")
        self.assertEqual(save_path.parent, SRC_ROOT / "log" / "exp")

    def test_build_ogm_save_path_uses_explicit_run_dir_when_supplied(self) -> None:
        explicit_run_dir = REPO_ROOT / "src" / "log" / "exp" / "sac_ogm_explicit"
        save_path = build_ogm_save_path("20990101_000000", run_dir=str(explicit_run_dir))
        self.assertEqual(save_path, explicit_run_dir)

    def test_ensure_src_working_directory_normalizes_from_repo_root(self) -> None:
        old_cwd = Path.cwd()
        try:
            os.chdir(REPO_ROOT)
            self.assertEqual(Path.cwd(), REPO_ROOT)

            normalized_cwd = ensure_src_working_directory()

            self.assertEqual(normalized_cwd, SRC_ROOT)
            self.assertEqual(Path.cwd(), SRC_ROOT)
        finally:
            os.chdir(old_cwd)

    def test_iter_global_episodes_preserves_fresh_run_schedule(self) -> None:
        self.assertEqual(list(iter_global_episodes(start_episode=0, train_episode=5)), [0, 1, 2, 3, 4])

    def test_iter_global_episodes_uses_resume_start_and_exclusive_target(self) -> None:
        episodes = list(iter_global_episodes(start_episode=20000, train_episode=30000))

        self.assertEqual(episodes[0], 20000)
        self.assertEqual(episodes[-1], 29999)
        self.assertEqual(len(episodes), 10000)

    def test_validate_episode_window_rejects_start_at_or_after_target(self) -> None:
        with self.assertRaisesRegex(ValueError, "start_episode"):
            validate_episode_window(start_episode=30000, train_episode=30000)

        with self.assertRaisesRegex(ValueError, "start_episode"):
            validate_episode_window(start_episode=30001, train_episode=30000)

    def test_validate_resume_checkpoint_requirement_rejects_resume_start_without_checkpoint(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume_checkpoint.*agent_ckpt"):
            validate_resume_checkpoint_requirement(start_episode=20000, checkpoint_path=None)

    def test_validate_resume_checkpoint_requirement_allows_fresh_or_checkpointed_runs(self) -> None:
        validate_resume_checkpoint_requirement(start_episode=0, checkpoint_path=None)
        validate_resume_checkpoint_requirement(start_episode=20000, checkpoint_path="SAC_19999.pt")

    def test_validate_resume_configuration_rejects_ambiguous_or_unbacked_continuation(self) -> None:
        with self.assertRaisesRegex(ValueError, "resume_state.*resume_checkpoint"):
            validate_resume_configuration(
                start_episode=20000,
                checkpoint_path="SAC_19999.pt",
                resume_state_path="stage4_state_19999.pt",
            )

        with self.assertRaisesRegex(ValueError, "start_episode > 0"):
            validate_resume_configuration(start_episode=20000, checkpoint_path=None, resume_state_path=None)

        with self.assertRaisesRegex(ValueError, "resume_state.*start_episode"):
            validate_resume_configuration(start_episode=0, checkpoint_path=None, resume_state_path="state.pt")

        validate_resume_configuration(start_episode=0, checkpoint_path=None, resume_state_path=None)
        validate_resume_configuration(start_episode=20000, checkpoint_path="SAC_19999.pt", resume_state_path=None)
        validate_resume_configuration(start_episode=20000, checkpoint_path=None, resume_state_path="state.pt")

    def test_validate_resume_configuration_rejects_corrected_target_mode_checkpoint_continuation(self) -> None:
        with self.assertRaisesRegex(ValueError, "target_mode.*corrected-cossin.*fresh.*full-state"):
            validate_resume_configuration(
                start_episode=20000,
                checkpoint_path="SAC_19999.pt",
                resume_state_path=None,
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
            )

        with self.assertRaisesRegex(ValueError, "target_mode.*corrected-cossin.*resume_state"):
            validate_resume_configuration(
                start_episode=20000,
                checkpoint_path=None,
                resume_state_path=None,
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
            )

        validate_resume_configuration(
            start_episode=0,
            checkpoint_path=None,
            resume_state_path=None,
            target_mode=TARGET_MODE_CORRECTED_COSSIN,
        )
        validate_resume_configuration(
            start_episode=20000,
            checkpoint_path=None,
            resume_state_path="stage4_state_19999.pt",
            target_mode=TARGET_MODE_CORRECTED_COSSIN,
        )

    def test_parse_args_exposes_maneuver_stability_defaults_and_overrides(self) -> None:
        with mock.patch.object(sys, "argv", ["train_HOPE_sac_ogm.py"]):
            default_args = stage4_runner._parse_args()

        self.assertEqual(default_args.maneuver_stability_mode, MODE_OFF)
        self.assertEqual(default_args.maneuver_stability_apply_to, "rl")
        self.assertAlmostEqual(default_args.maneuver_stability_min_speed, 1e-6)
        self.assertEqual(default_args.maneuver_stability_hold_steps, 1)
        self.assertIsNone(default_args.maneuver_stability_max_hold_speed)

        with mock.patch.object(
            sys,
            "argv",
            [
                "train_HOPE_sac_ogm.py",
                "--maneuver_stability_mode",
                MODE_DIAGNOSTIC,
                "--maneuver_stability_apply_to",
                "rl-rs",
                "--maneuver_stability_min_speed",
                "0.05",
                "--maneuver_stability_hold_steps",
                "2",
                "--maneuver_stability_max_hold_speed",
                "0.5",
            ],
        ):
            explicit_args = stage4_runner._parse_args()

        self.assertEqual(explicit_args.maneuver_stability_mode, MODE_DIAGNOSTIC)
        self.assertEqual(explicit_args.maneuver_stability_apply_to, "rl-rs")
        self.assertAlmostEqual(explicit_args.maneuver_stability_min_speed, 0.05)
        self.assertEqual(explicit_args.maneuver_stability_hold_steps, 2)
        self.assertAlmostEqual(explicit_args.maneuver_stability_max_hold_speed, 0.5)

    def test_resolve_checkpoint_path_prefers_resume_checkpoint_and_keeps_agent_ckpt_compatibility(self) -> None:
        resume_args = argparse.Namespace(resume_checkpoint="resume.pt", agent_ckpt=None)
        legacy_args = argparse.Namespace(resume_checkpoint=None, agent_ckpt="legacy.pt")
        matching_args = argparse.Namespace(resume_checkpoint="same.pt", agent_ckpt="same.pt")
        fresh_args = argparse.Namespace(resume_checkpoint=None, agent_ckpt=None)

        self.assertEqual(resolve_checkpoint_path(resume_args), "resume.pt")
        self.assertEqual(resolve_checkpoint_path(legacy_args), "legacy.pt")
        self.assertEqual(resolve_checkpoint_path(matching_args), "same.pt")
        self.assertIsNone(resolve_checkpoint_path(fresh_args))

    def test_resolve_checkpoint_path_rejects_conflicting_resume_names(self) -> None:
        args = argparse.Namespace(resume_checkpoint="resume.pt", agent_ckpt="legacy.pt")

        with self.assertRaisesRegex(ValueError, "resume_checkpoint.*agent_ckpt"):
            resolve_checkpoint_path(args)

    def test_periodic_checkpoint_name_uses_global_episode_number(self) -> None:
        self.assertEqual(periodic_checkpoint_name(29999), "SAC_29999.pt")

    def test_stage4_state_names_are_explicit_and_do_not_overlap_sac_checkpoints(self) -> None:
        self.assertEqual(stage4_state_name(29999), "stage4_state_29999.pt")
        self.assertEqual(stage4_latest_state_name(), "stage4_state_latest.pt")
        self.assertNotEqual(stage4_state_name(29999), periodic_checkpoint_name(29999))

    def test_stage4_full_state_round_trip_restores_replay_rng_choosers_and_histories(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        memory.push(("obs-2", "action-2", 2.0, True, "log-2", "next-2"))
        parking_agent = _FakeParkingAgent(memory)

        scene_chooser = SceneChoose()
        scene_chooser.scene_record = [0, 3, 1]
        scene_chooser.success_record[0].extend([1, 0])
        scene_chooser.success_record[3].append(1)
        dlp_case_chooser = DlpCaseChoose()
        dlp_case_chooser.case_record = [7, 8]
        dlp_case_chooser.case_success_rate["7"] = [0, 1]

        np.random.seed(123)
        torch.manual_seed(456)
        expected_np_state = np.random.get_state()
        expected_torch_state = torch.get_rng_state().clone()

        with tempfile.TemporaryDirectory(prefix="stage4 state ") as temp_dir:
            state_path = save_stage4_state(
                save_dir=Path(temp_dir),
                global_episode=29999,
                parking_agent=parking_agent,
                total_step_num=4567,
                scene_chooser=scene_chooser,
                dlp_case_chooser=dlp_case_chooser,
                best_success_rate=[0.1, 0.2, 0.3, 0.4],
                reward_list=[10.0, 11.0],
                reward_per_state_list=[1.0, 2.0, 3.0],
                reward_info_list=[[1, 2], [3, 4]],
                case_id_list=["case-a", "case-b"],
                succ_record=[1, 0],
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
            )

            self.assertEqual(state_path.name, stage4_state_name(29999))
            self.assertTrue((Path(temp_dir) / stage4_latest_state_name()).exists())

            np.random.random(10)
            torch.rand(10)
            parking_agent.agent.memory.clear()
            parking_agent.agent.actor_loss_list = []
            parking_agent.agent.critic_loss_list = []
            scene_chooser.scene_record = []
            scene_chooser.success_record[0] = []
            dlp_case_chooser.case_record = []
            dlp_case_chooser.case_success_rate["7"] = []

            restored = load_stage4_state(
                state_path=state_path,
                parking_agent=parking_agent,
                scene_chooser=scene_chooser,
                dlp_case_chooser=dlp_case_chooser,
                expected_target_mode=TARGET_MODE_CORRECTED_COSSIN,
            )

        self.assertEqual(restored["schema_version"], FULL_STATE_SCHEMA_VERSION)
        self.assertEqual(restored["target_mode"], TARGET_MODE_CORRECTED_COSSIN)
        self.assertEqual(restored["global_episode"], 29999)
        self.assertEqual(restored["total_step_num"], 4567)
        self.assertEqual(restored["best_success_rate"], [0.1, 0.2, 0.3, 0.4])
        self.assertEqual(restored["reward_list"], [10.0, 11.0])
        self.assertEqual(restored["reward_per_state_list"], [1.0, 2.0, 3.0])
        self.assertEqual(restored["reward_info_list"], [[1, 2], [3, 4]])
        self.assertEqual(restored["case_id_list"], ["case-a", "case-b"])
        self.assertEqual(restored["succ_record"], [1, 0])
        self.assertEqual(len(parking_agent.agent.memory), 2)
        self.assertEqual(list(parking_agent.agent.memory.memory["state"]), ["obs-1", "obs-2"])
        self.assertEqual(parking_agent.agent.actor_loss_list, [1.25])
        self.assertEqual(parking_agent.agent.critic_loss_list, [2.5])
        self.assertEqual(scene_chooser.scene_record, [0, 3, 1])
        self.assertEqual(scene_chooser.success_record[0], [1, 0])
        self.assertEqual(scene_chooser.success_record[3], [1])
        self.assertEqual(dlp_case_chooser.case_record, [7, 8])
        self.assertEqual(dlp_case_chooser.case_success_rate["7"], [0, 1])
        np.testing.assert_array_equal(np.random.get_state()[1], expected_np_state[1])
        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))

    def test_load_stage4_state_rejects_target_mode_mismatch(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)

        with tempfile.TemporaryDirectory(prefix="stage4 target mode mismatch ") as temp_dir:
            state_path = save_stage4_state(
                save_dir=Path(temp_dir),
                global_episode=19999,
                parking_agent=parking_agent,
                total_step_num=1234,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                best_success_rate=[0.0, 0.0, 0.0, 0.0],
                reward_list=[1.0],
                reward_per_state_list=[0.5],
                reward_info_list=[[0, 0, 0, 0, 0]],
                case_id_list=["case-a"],
                succ_record=[1],
                target_mode=TARGET_MODE_LEGACY_COSCOS,
            )

            with self.assertRaisesRegex(ValueError, "target_mode"):
                load_stage4_state(
                    state_path=state_path,
                    parking_agent=parking_agent,
                    scene_chooser=SceneChoose(),
                    dlp_case_chooser=DlpCaseChoose(),
                    expected_target_mode=TARGET_MODE_CORRECTED_COSSIN,
                )

    def test_stage4_state_round_trip_records_maneuver_stability_config(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)
        config = ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC, min_speed=0.05, max_hold_speed=0.5)
        tracker = ManeuverStabilityTracker(config)
        tracker.apply(np.array([0.0, 0.6], dtype=float), source="RL")
        tracker.apply(np.array([0.0, -0.4], dtype=float), source="RL")

        with tempfile.TemporaryDirectory(prefix="stage4 maneuver state ") as temp_dir:
            state_path = save_stage4_state(
                save_dir=Path(temp_dir),
                global_episode=19999,
                parking_agent=parking_agent,
                total_step_num=1234,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                best_success_rate=[0.0, 0.0, 0.0, 0.0],
                reward_list=[1.0],
                reward_per_state_list=[0.5],
                reward_info_list=[[0, 0, 0, 0, 0]],
                case_id_list=["case-a"],
                succ_record=[1],
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
                maneuver_stability_config=config,
                maneuver_stability_state=tracker.snapshot(),
            )

            restored = load_stage4_state(
                state_path=state_path,
                parking_agent=parking_agent,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                expected_target_mode=TARGET_MODE_CORRECTED_COSSIN,
                expected_maneuver_stability_config=config,
            )

        self.assertEqual(restored["maneuver_stability_config"]["mode"], MODE_DIAGNOSTIC)
        self.assertAlmostEqual(restored["maneuver_stability_config"]["min_speed"], 0.05)
        self.assertAlmostEqual(restored["maneuver_stability_config"]["max_hold_speed"], 0.5)
        self.assertEqual(restored["maneuver_stability_state"]["counters"]["would_change"], 1)
        self.assertEqual(restored["maneuver_stability_state"]["counters"]["speed_sign_flips"], 1)

    def test_load_stage4_state_defaults_legacy_maneuver_config_to_off(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)

        with tempfile.TemporaryDirectory(prefix="stage4 legacy maneuver state ") as temp_dir:
            payload = stage4_runner.build_stage4_state_payload(
                global_episode=19999,
                parking_agent=parking_agent,
                total_step_num=1234,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                best_success_rate=[0.0, 0.0, 0.0, 0.0],
                reward_list=[1.0],
                reward_per_state_list=[0.5],
                reward_info_list=[[0, 0, 0, 0, 0]],
                case_id_list=["case-a"],
                succ_record=[1],
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
            )
            payload.pop("maneuver_stability_config", None)
            state_path = Path(temp_dir) / "legacy_state.pt"
            torch.save(payload, state_path)

            restored = load_stage4_state(
                state_path=state_path,
                parking_agent=parking_agent,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                expected_target_mode=TARGET_MODE_CORRECTED_COSSIN,
                expected_maneuver_stability_config=ManeuverStabilityConfig(mode=MODE_OFF),
            )

        self.assertEqual(restored["maneuver_stability_config"]["mode"], MODE_OFF)

    def test_load_stage4_state_rejects_mismatched_non_off_maneuver_config(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)

        with tempfile.TemporaryDirectory(prefix="stage4 maneuver mismatch ") as temp_dir:
            state_path = save_stage4_state(
                save_dir=Path(temp_dir),
                global_episode=19999,
                parking_agent=parking_agent,
                total_step_num=1234,
                scene_chooser=SceneChoose(),
                dlp_case_chooser=DlpCaseChoose(),
                best_success_rate=[0.0, 0.0, 0.0, 0.0],
                reward_list=[1.0],
                reward_per_state_list=[0.5],
                reward_info_list=[[0, 0, 0, 0, 0]],
                case_id_list=["case-a"],
                succ_record=[1],
                target_mode=TARGET_MODE_CORRECTED_COSSIN,
                maneuver_stability_config=ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC),
            )

            with self.assertRaisesRegex(ValueError, "maneuver_stability_config"):
                load_stage4_state(
                    state_path=state_path,
                    parking_agent=parking_agent,
                    scene_chooser=SceneChoose(),
                    dlp_case_chooser=DlpCaseChoose(),
                    expected_target_mode=TARGET_MODE_CORRECTED_COSSIN,
                    expected_maneuver_stability_config=ManeuverStabilityConfig(
                        mode=MODE_DIRECTION_HOLD,
                        hold_steps=2,
                    ),
                )

    def test_maneuver_stability_training_helper_keeps_off_action_and_log_prob_objects(self) -> None:
        parking_agent = _RecordingLogProbParkingAgent()
        tracker = ManeuverStabilityTracker(ManeuverStabilityConfig(mode=MODE_OFF))
        action = np.array([0.25, -0.75], dtype=float)
        log_prob = object()

        applied_action, applied_log_prob, result = stage4_runner.apply_maneuver_stability_to_training_action(
            parking_agent=parking_agent,
            tracker=tracker,
            obs="obs",
            action=action,
            log_prob=log_prob,
            source="RL",
        )

        self.assertIs(applied_action, action)
        self.assertIs(applied_log_prob, log_prob)
        self.assertFalse(result.changed)
        self.assertEqual(result.reason, "off")
        self.assertEqual(parking_agent.log_prob_calls, [])

    def test_maneuver_stability_training_helper_recomputes_log_prob_for_changed_action(self) -> None:
        parking_agent = _RecordingLogProbParkingAgent()
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=1)
        )
        first_action = np.array([0.0, 0.6], dtype=float)
        flip_action = np.array([0.0, -0.4], dtype=float)

        stage4_runner.apply_maneuver_stability_to_training_action(
            parking_agent=parking_agent,
            tracker=tracker,
            obs="obs-1",
            action=first_action,
            log_prob="initial-log-prob",
            source="RL",
        )
        applied_action, applied_log_prob, result = stage4_runner.apply_maneuver_stability_to_training_action(
            parking_agent=parking_agent,
            tracker=tracker,
            obs="obs-2",
            action=flip_action,
            log_prob="old-log-prob",
            source="RL",
        )

        self.assertTrue(result.changed)
        self.assertEqual(result.reason, "direction_hold")
        self.assertEqual(applied_log_prob, "recomputed-log-prob")
        np.testing.assert_array_equal(applied_action, np.array([0.0, 0.4], dtype=float))
        np.testing.assert_array_equal(flip_action, np.array([0.0, -0.4], dtype=float))
        self.assertEqual(len(parking_agent.log_prob_calls), 1)
        self.assertEqual(parking_agent.log_prob_calls[0][0], "obs-2")
        np.testing.assert_array_equal(parking_agent.log_prob_calls[0][1], applied_action)

    def test_save_stage4_state_replaces_named_and_latest_snapshots_atomically(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)
        fake_torch = _FakeTorchForSaving()

        with tempfile.TemporaryDirectory(prefix="stage4 atomic state ") as temp_dir:
            temp_root = Path(temp_dir)
            replacements = []

            def record_replace(source, target):
                source = Path(source)
                target = Path(target)
                replacements.append((source, target))
                self.assertEqual(source.parent, target.parent)
                self.assertNotEqual(source.name, target.name)
                return target

            with mock.patch("pathlib.Path.replace", autospec=True, side_effect=record_replace):
                state_path = save_stage4_state(
                    save_dir=temp_root,
                    global_episode=29999,
                    parking_agent=parking_agent,
                    total_step_num=4567,
                    scene_chooser=SceneChoose(),
                    dlp_case_chooser=DlpCaseChoose(),
                    best_success_rate=[0.1, 0.2, 0.3, 0.4],
                    reward_list=[10.0],
                    reward_per_state_list=[1.0],
                    reward_info_list=[[1, 2]],
                    case_id_list=["case-a"],
                    succ_record=[1],
                    torch_module=fake_torch,
                )

        final_names = {stage4_state_name(29999), stage4_latest_state_name()}
        self.assertEqual(state_path.name, stage4_state_name(29999))
        self.assertEqual({target.name for _, target in replacements}, final_names)
        self.assertEqual(len(fake_torch.saved_paths), 2)
        self.assertTrue(all(path.parent == temp_root for path in fake_torch.saved_paths))
        self.assertFalse(any(path.name in final_names for path in fake_torch.saved_paths))

    def test_close_training_resources_flushes_and_closes_writer_then_env_when_present(self) -> None:
        self.assertTrue(hasattr(stage4_runner, "close_training_resources"))
        writer = _RecordingWriter()
        env = _RecordingEnv()

        stage4_runner.close_training_resources(writer, env)
        stage4_runner.close_training_resources(writer=None, env=object())

        self.assertEqual(writer.calls, ["flush", "close"])
        self.assertTrue(env.closed)

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this environment")
    def test_stage4_full_state_round_trip_restores_cuda_rng_state_when_available(self) -> None:
        memory = ReplayMemory(5, ["log_prob", "next_obs"])
        memory.push(("obs-1", "action-1", 1.0, False, "log-1", "next-1"))
        parking_agent = _FakeParkingAgent(memory)
        scene_chooser = SceneChoose()
        dlp_case_chooser = DlpCaseChoose()

        torch.cuda.manual_seed_all(789)
        expected_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]

        with tempfile.TemporaryDirectory(prefix="stage4 cuda state ") as temp_dir:
            state_path = save_stage4_state(
                save_dir=Path(temp_dir),
                global_episode=19999,
                parking_agent=parking_agent,
                total_step_num=1234,
                scene_chooser=scene_chooser,
                dlp_case_chooser=dlp_case_chooser,
                best_success_rate=[0.0, 0.0, 0.0, 0.0],
                reward_list=[1.0],
                reward_per_state_list=[0.5],
                reward_info_list=[[0, 0, 0, 0, 0]],
                case_id_list=["case-a"],
                succ_record=[1],
            )

            for device_index in range(torch.cuda.device_count()):
                torch.rand(4, device=f"cuda:{device_index}")

            load_stage4_state(
                state_path=state_path,
                parking_agent=parking_agent,
                scene_chooser=scene_chooser,
                dlp_case_chooser=dlp_case_chooser,
            )

        actual_cuda_states = torch.cuda.get_rng_state_all()
        self.assertEqual(len(actual_cuda_states), len(expected_cuda_states))
        for actual, expected in zip(actual_cuda_states, expected_cuda_states):
            self.assertTrue(torch.equal(actual, expected))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this environment")
    def test_restore_rng_state_accepts_cuda_mapped_cpu_rng_tensor(self) -> None:
        expected_torch_state = torch.get_rng_state().clone()
        mapped_rng_state = {"torch": expected_torch_state.to("cuda:0")}

        torch.rand(10)

        stage4_runner._restore_rng_state(mapped_rng_state, torch)

        self.assertTrue(torch.equal(torch.get_rng_state(), expected_torch_state))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA is not available in this environment")
    def test_restore_rng_state_accepts_cuda_mapped_cuda_rng_tensors(self) -> None:
        expected_cuda_states = [state.clone() for state in torch.cuda.get_rng_state_all()]
        mapped_rng_state = {
            "cuda": [state.to("cuda:%d" % index) for index, state in enumerate(expected_cuda_states)]
        }

        for device_index in range(torch.cuda.device_count()):
            torch.rand(4, device=f"cuda:{device_index}")

        stage4_runner._restore_rng_state(mapped_rng_state, torch)

        actual_cuda_states = torch.cuda.get_rng_state_all()
        self.assertEqual(len(actual_cuda_states), len(expected_cuda_states))
        for actual, expected in zip(actual_cuda_states, expected_cuda_states):
            self.assertTrue(torch.equal(actual, expected))

    def test_iter_verbose_history_rows_does_not_overread_local_resume_history(self) -> None:
        rows = list(iter_verbose_history_rows(["case-a"], [12.5], [[1, 2, 3]], limit=10))

        self.assertEqual(rows, [("case-a", 12.5, [1, 2, 3])])

    def test_iter_verbose_history_rows_uses_most_recent_ten_local_items(self) -> None:
        case_ids = [f"case-{i}" for i in range(12)]
        rewards = list(range(12))
        reward_infos = [[i] for i in range(12)]

        rows = list(iter_verbose_history_rows(case_ids, rewards, reward_infos, limit=10))

        self.assertEqual(rows[0], ("case-2", 2, [2]))
        self.assertEqual(rows[-1], ("case-11", 11, [11]))
        self.assertEqual(len(rows), 10)

    def test_format_best_success_log_uses_global_episode_identifier(self) -> None:
        log_line = format_best_success_log(20000, [0.1, 0.2, 0.3, 0.4])

        self.assertEqual(log_line, "epoch: 20000, success rate: 0.1 0.2 0.3 0.4")


if __name__ == "__main__":
    unittest.main()
