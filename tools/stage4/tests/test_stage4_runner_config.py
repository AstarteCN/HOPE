from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
