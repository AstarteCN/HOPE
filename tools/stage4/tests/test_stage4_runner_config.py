from __future__ import annotations

import argparse
import os
import sys
import unittest
from copy import deepcopy
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS  # noqa: E402
from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
from model.agent.sac_agent import SACAgent  # noqa: E402
from tools.stage4.train_HOPE_sac_ogm import (  # noqa: E402
    build_ogm_save_path,
    build_ogm_training_config,
    ensure_src_working_directory,
    format_best_success_log,
    iter_global_episodes,
    iter_verbose_history_rows,
    periodic_checkpoint_name,
    resolve_checkpoint_path,
    validate_episode_window,
    validate_resume_checkpoint_requirement,
)


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
