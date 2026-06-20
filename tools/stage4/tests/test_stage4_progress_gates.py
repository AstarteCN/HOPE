from __future__ import annotations

import unittest

from tools.stage4.stage4_ogm_targets import (
    BASELINE_FAST_ACTION_MASK_20K,
    OGM_SIM_TARGETS,
    relative_band,
)


class Stage4TargetTests(unittest.TestCase):
    def test_ogm_sim_targets_match_prd_values(self) -> None:
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["psr"], 0.9933)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["angs"], 1.5)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["pl"], 20.3)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["psr"], 0.977)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["angs"], 1.9)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["pl"], 23.6)

    def test_relative_band_uses_three_percent_window(self) -> None:
        low, high = relative_band(100.0)
        self.assertAlmostEqual(low, 97.0)
        self.assertAlmostEqual(high, 103.0)

    def test_fast_action_mask_20k_baseline_is_recorded(self) -> None:
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["id"], "hope-fast-action-mask-20k")
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["run_dir"], "src/log/exp/sac_20260620_085208")
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["checkpoint"], "SAC_19999.pt")
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["wall_time_hours"], 10.947863)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["episodes_per_hour"], 1826.840564)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["env_steps_per_second"], 46.223481)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Normal"], 0.985)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Complex"], 0.945)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Extrem"], 0.655)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["DLP"], 0.960)


if __name__ == "__main__":
    unittest.main()
