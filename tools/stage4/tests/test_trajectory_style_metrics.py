import unittest

from tools.stage4.trajectory_style_metrics import (
    count_cusps,
    count_gear_shifts,
    count_low_speed_chatter,
    curvature_stats,
    fourier_descriptor_distance,
    hausdorff_distance,
    mean_l2_distance,
    min_obstacle_clearance,
    resample_polyline,
    steering_sign_changes,
)


class TrajectoryStyleMetricsTests(unittest.TestCase):
    def test_resample_preserves_endpoints_and_identical_l2_is_zero(self):
        path = [[0.0, 0.0], [2.0, 0.0]]

        sampled = resample_polyline(path, sample_count=5)

        self.assertEqual(sampled[0], [0.0, 0.0])
        self.assertEqual(sampled[-1], [2.0, 0.0])
        self.assertAlmostEqual(mean_l2_distance(sampled, sampled), 0.0)

    def test_shifted_path_increases_l2_and_hausdorff(self):
        base = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]
        shifted = [[0.0, 1.0], [1.0, 1.0], [2.0, 1.0]]

        self.assertGreater(mean_l2_distance(base, shifted), 0.9)
        self.assertGreater(hausdorff_distance(base, shifted), 0.9)

    def test_hausdorff_matches_identical_geometry_with_different_waypoint_density(self):
        sparse = [[0.0, 0.0], [2.0, 0.0]]
        dense = [[0.0, 0.0], [1.0, 0.0], [2.0, 0.0]]

        self.assertAlmostEqual(hausdorff_distance(sparse, dense), 0.0)

    def test_fourier_distance_is_small_for_identical_path(self):
        path = [[0.0, 0.0], [1.0, 0.5], [2.0, 0.0], [3.0, -0.5]]

        self.assertAlmostEqual(fourier_descriptor_distance(path, path), 0.0, places=7)

    def test_curvature_cusps_and_chatter_are_detected(self):
        path = [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [2.0, 1.0]]

        stats = curvature_stats(path)

        self.assertGreater(stats["max_abs_curvature"], 0.0)
        self.assertEqual(count_gear_shifts([-1.0, -0.5, 0.3, 0.4, -0.2]), 2)
        self.assertEqual(count_cusps([-1.0, -0.5, 0.3, 0.4, -0.2]), 2)
        self.assertEqual(count_low_speed_chatter([-0.1, 0.1, -0.1, 0.1], max_abs_speed=0.2), 3)
        self.assertEqual(steering_sign_changes([0.1, 0.2, -0.1, -0.2, 0.3]), 2)

    def test_min_obstacle_clearance_uses_polygon_edges(self):
        path = [[0.0, 0.0], [1.0, 0.0]]
        obstacle = [[[2.0, -1.0], [3.0, -1.0], [3.0, 1.0], [2.0, 1.0]]]

        self.assertAlmostEqual(min_obstacle_clearance(path, obstacle), 1.0)


if __name__ == "__main__":
    unittest.main()
