import math
import unittest

import numpy as np

from tools.stage4.debug_trace_schema import (
    densify_polyline,
    default_layer_groups,
    ogm_default_grid,
    project_lidar_hits,
    sparsify_ogm,
    vehicle_body_corners_local,
    vehicle_params_from_configs,
)


class DebugTraceSchemaTest(unittest.TestCase):
    def test_vehicle_params_and_body_geometry_use_config_values(self):
        params = vehicle_params_from_configs()

        self.assertAlmostEqual(params["length_m"], 4.69)
        self.assertAlmostEqual(params["width_m"], 1.94)
        self.assertAlmostEqual(params["wheelbase_m"], 2.8)
        corners = vehicle_body_corners_local(params)

        self.assertEqual(len(corners), 4)
        self.assertAlmostEqual(corners[0][0], -0.93)
        self.assertAlmostEqual(corners[1][0], 3.76)
        self.assertAlmostEqual(corners[0][1], -0.97)
        self.assertAlmostEqual(corners[2][1], 0.97)

    def test_densify_polyline_preserves_endpoints_and_spacing(self):
        dense = densify_polyline([[0.0, 0.0, 0.0], [1.0, 0.0, 0.5]], max_spacing_m=0.4)

        self.assertEqual(dense[0], [0.0, 0.0, 0.0])
        self.assertEqual(dense[-1], [1.0, 0.0, 0.5])
        for first, second in zip(dense, dense[1:]):
            self.assertLessEqual(math.dist(first[:2], second[:2]), 0.400001)

    def test_project_lidar_hits_uses_120_rays_and_positive_x_first_ray(self):
        distances = np.ones(120, dtype=float) * 2.0

        points = project_lidar_hits(distances)

        self.assertEqual(len(points), 120)
        self.assertAlmostEqual(points[0][0], 2.0)
        self.assertAlmostEqual(points[0][1], 0.0)

    def test_ogm_grid_and_sparse_cells_are_channel_first_compatible(self):
        grid = ogm_default_grid()
        ogm = np.zeros((2, 64, 64), dtype=np.float32)
        ogm[0, 10, 20] = 1.0
        ogm[1, 30, 40] = 0.5

        cells = sparsify_ogm(ogm)

        self.assertEqual(grid["channels"], ["obstacle", "target"])
        self.assertEqual(
            cells,
            [
                {"channel": "obstacle", "row": 10, "col": 20, "value": 1.0},
                {"channel": "target", "row": 30, "col": 40, "value": 0.5},
            ],
        )

    def test_layer_groups_separate_rl_and_rs_trajectory_toggles(self):
        groups = default_layer_groups()

        self.assertIn("trajectory_control", groups)
        self.assertIn("rl_trajectory", groups["trajectory_control"])
        self.assertIn("rs_trajectory", groups["trajectory_control"])


if __name__ == "__main__":
    unittest.main()
