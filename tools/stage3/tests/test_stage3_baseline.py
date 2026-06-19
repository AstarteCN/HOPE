import unittest

from tools.stage3.stage3_baseline import (
    BASELINE_20K,
    QUALITY_THRESHOLDS,
    SPEED_MIN_IMPROVEMENT_RATIO,
    quality_gate_status,
    speed_gate_passed,
)


class Stage3BaselineTests(unittest.TestCase):
    def test_baseline_identity_and_thresholds(self):
        self.assertEqual(BASELINE_20K["run_dir"], r"D:\Github\HOPE\src\log\exp\sac_20260619_004316")
        self.assertEqual(BASELINE_20K["checkpoint"], r"D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt")
        self.assertEqual(BASELINE_20K["wall_time_hours"], 12.964)
        self.assertEqual(BASELINE_20K["episodes_per_hour"], 1545.71)
        self.assertEqual(BASELINE_20K["env_steps_per_second"], 39.10)
        self.assertEqual(BASELINE_20K["stop_episode_snapshot"], 20038)
        self.assertEqual(BASELINE_20K["env_steps"], 1824904)
        self.assertAlmostEqual(BASELINE_20K["external_eval"]["mean"], 0.88625)
        self.assertAlmostEqual(QUALITY_THRESHOLDS["mean"], 0.85625)
        self.assertAlmostEqual(SPEED_MIN_IMPROVEMENT_RATIO, 0.10)

    def test_quality_gate_accepts_baseline_level_eval(self):
        result = quality_gate_status(
            eval_success={"Normal": 0.985, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960},
            tensorboard_has_nonfinite=False,
            multi_scene_collapse=False,
            checkpoint_missing=False,
            parity_failed=False,
        )
        self.assertTrue(result["quality_pass"])
        self.assertEqual(result["hard_reject_reasons"], [])

    def test_quality_gate_rejects_missing_checkpoint(self):
        result = quality_gate_status(
            eval_success={"Normal": 1.0, "Complex": 1.0, "Extrem": 1.0, "DLP": 1.0},
            tensorboard_has_nonfinite=False,
            multi_scene_collapse=False,
            checkpoint_missing=True,
            parity_failed=False,
        )
        self.assertFalse(result["quality_pass"])
        self.assertIn("missing_20k_checkpoint", result["hard_reject_reasons"])

    def test_speed_gate_uses_any_primary_metric(self):
        self.assertTrue(speed_gate_passed({"wall_time_hours": 11.0, "env_steps_per_second": 39.10, "episodes_per_hour": 1545.71})["speed_pass"])
        self.assertTrue(speed_gate_passed({"wall_time_hours": 12.964, "env_steps_per_second": 44.0, "episodes_per_hour": 1545.71})["speed_pass"])
        self.assertTrue(speed_gate_passed({"wall_time_hours": 12.964, "env_steps_per_second": 39.10, "episodes_per_hour": 1701.0})["speed_pass"])
        self.assertFalse(speed_gate_passed({"wall_time_hours": 12.5, "env_steps_per_second": 40.0, "episodes_per_hour": 1600.0})["speed_pass"])


if __name__ == "__main__":
    unittest.main()
