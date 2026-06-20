from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LinearRing

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.ogm import OGMConfig, build_ego_ogm, world_to_grid  # noqa: E402
from env.vehicle import State  # noqa: E402


class ProxyOGMRasterizerTests(unittest.TestCase):
    def test_world_to_grid_places_forward_points_right_of_center(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        row, col = world_to_grid(ego_state=State([0.0, 0.0, 0.0]), x=2.0, y=0.0, config=config)
        self.assertEqual(row, 32)
        self.assertGreater(col, 32)

    def test_world_to_grid_places_left_points_above_center(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        row, col = world_to_grid(ego_state=State([0.0, 0.0, 0.0]), x=0.0, y=2.0, config=config)
        self.assertLess(row, 32)
        self.assertEqual(col, 32)

    def test_world_to_grid_respects_ego_heading(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        ego = State([0.0, 0.0, math.pi / 2.0])
        row, col = world_to_grid(ego_state=ego, x=0.0, y=2.0, config=config)
        self.assertEqual(row, 32)
        self.assertGreater(col, 32)

    def test_build_ego_ogm_marks_obstacle_and_target_channels(self) -> None:
        config = OGMConfig(size=64, resolution=0.25, channels=2)
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        ogm = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        self.assertEqual(ogm.shape, (64, 64, 2))
        self.assertEqual(ogm.dtype, np.float32)
        self.assertGreater(float(ogm[:, :, 0].sum()), 0.0)
        self.assertGreater(float(ogm[:, :, 1].sum()), 0.0)

    def test_build_ego_ogm_is_deterministic(self) -> None:
        config = OGMConfig(size=32, resolution=0.5, channels=2)
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        first = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)
        second = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
