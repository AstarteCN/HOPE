from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LinearRing, Point

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.ogm import (  # noqa: E402
    OGMConfig,
    _cached_cell_footprints,
    _cell_center_world,
    build_ego_ogm,
    world_to_grid,
)
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

    def test_cell_center_world_maps_grid_center_to_world_coordinates(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        world_x, world_y = _cell_center_world(
            ego_state=State([0.0, 0.0, 0.0]),
            row=32,
            col=32,
            config=config,
        )
        self.assertAlmostEqual(world_x, 0.25)
        self.assertAlmostEqual(world_y, -0.25)

    def test_cached_cell_footprints_are_local_and_reused(self) -> None:
        first = _cached_cell_footprints(size=64, resolution=0.5)
        second = _cached_cell_footprints(size=64, resolution=0.5)

        self.assertIs(first, second)
        self.assertEqual(len(first), 64)
        self.assertEqual(len(first[0]), 64)
        self.assertTrue(first[32][32].covers(Point(0.25, -0.25)))

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

    def test_build_ego_ogm_marks_thin_wall_at_default_resolution(self) -> None:
        config = OGMConfig()
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(-0.05, 1.0), (0.05, 1.0), (0.05, 2.0), (-0.05, 2.0)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        ogm = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        self.assertGreater(float(ogm[:, :, config.obstacle_channel].sum()), 0.0)

    def test_build_ego_ogm_is_deterministic(self) -> None:
        config = OGMConfig(size=32, resolution=0.5, channels=2)
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        first = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)
        second = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        np.testing.assert_array_equal(first, second)

    def test_ogm_config_rejects_bad_size_or_resolution(self) -> None:
        invalid_configs = [
            {"size": 0},
            {"size": 32.0},
            {"resolution": 0.0},
            {"resolution": math.inf},
        ]
        for kwargs in invalid_configs:
            with self.subTest(kwargs=kwargs):
                with self.assertRaises(ValueError):
                    OGMConfig(**kwargs)

    def test_ogm_config_rejects_negative_channel(self) -> None:
        with self.assertRaises(ValueError):
            OGMConfig(obstacle_channel=-1)

    def test_ogm_config_rejects_duplicate_channels(self) -> None:
        with self.assertRaises(ValueError):
            OGMConfig(obstacle_channel=0, target_channel=0)

    def test_ogm_config_rejects_non_floating_dtype(self) -> None:
        with self.assertRaises(ValueError):
            OGMConfig(dtype=np.int32)


if __name__ == "__main__":
    unittest.main()
