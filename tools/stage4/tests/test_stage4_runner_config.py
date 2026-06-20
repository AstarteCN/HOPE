from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
from tools.stage4.train_HOPE_sac_ogm import build_ogm_training_config  # noqa: E402


class Stage4RunnerConfigTests(unittest.TestCase):
    def test_build_ogm_training_config_uses_only_target_action_mask_and_ogm(self) -> None:
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
            self.assertEqual(config["actor_layers"]["ogm_shape"], (2, 64, 64))
            self.assertEqual(config["actor_layers"]["n_modal"], 3)
            self.assertEqual(config["critic_layers"]["n_modal"], 4)
        finally:
            raw_env.close()


if __name__ == "__main__":
    unittest.main()
