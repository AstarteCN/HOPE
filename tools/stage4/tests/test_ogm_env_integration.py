from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402


class OGMEnvironmentIntegrationTests(unittest.TestCase):
    def test_default_env_does_not_advertise_ogm_observation_space(self) -> None:
        env = CarParking(render_mode="rgb_array", verbose=False)
        try:
            self.assertNotIn("ogm", env.observation_space)
        finally:
            env.close()

    def test_ogm_enabled_env_returns_hwc_ogm(self) -> None:
        env = CarParking(
            render_mode="rgb_array",
            verbose=False,
            use_img_observation=False,
            use_lidar_observation=True,
            use_action_mask=True,
            use_ogm_observation=True,
        )
        try:
            obs = env.reset(0, None, "Normal")
            self.assertIn("ogm", obs)
            self.assertEqual(obs["ogm"].shape, (64, 64, 2))
            self.assertEqual(obs["ogm"].dtype, np.float32)
            self.assertGreaterEqual(float(obs["ogm"].min()), 0.0)
            self.assertLessEqual(float(obs["ogm"].max()), 1.0)
            self.assertIsNotNone(obs["action_mask"])
            self.assertIsNotNone(obs["target"])
        finally:
            env.close()

    def test_wrapper_transposes_ogm_to_chw(self) -> None:
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
            obs = env.reset(0, None, "Normal")
            self.assertEqual(env.observation_shape["ogm"], (2, 64, 64))
            self.assertEqual(obs["ogm"].shape, (2, 64, 64))
        finally:
            raw_env.close()


if __name__ == "__main__":
    unittest.main()
