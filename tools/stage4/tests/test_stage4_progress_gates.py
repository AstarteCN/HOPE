from __future__ import annotations

import unittest

from tools.stage4.stage4_ogm_targets import (
    BASELINE_FAST_ACTION_MASK_20K,
    OGM_SIM_TARGETS,
    relative_band,
)
from tools.stage4.stage4_ogm_progress import decide_progress_gate, recommend_recovery_action


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


class Stage4ProgressGateTests(unittest.TestCase):
    def test_20k_gate_allows_weaker_than_hope_baseline_when_trend_improves(self) -> None:
        current = {
            "episode": 20000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.08, "mean_psr_delta": 0.05, "step_num_improvement": 0.08},
            "summary": {"Sim-Normal": {"psr": 0.70, "angs": 3.0, "pl": 35.0}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")
        self.assertIn("early_ogm_can_lag_hope_20k_baseline", decision["notes"])

    def test_first_stagnant_gate_gets_one_10k_grace_window(self) -> None:
        current = {
            "episode": 30000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.01, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}},
        }
        previous = {"episode": 20000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "grace-10k")

    def test_second_consecutive_stagnant_gate_stops_before_100k(self) -> None:
        current = {
            "episode": 40000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.02, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.2, "pl": 46.0}},
        }
        previous = {"episode": 30000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_recovery_restarts_when_observation_or_rasterizer_changes(self) -> None:
        self.assertEqual(recommend_recovery_action(["ogm_rasterizer_changed"]), "restart-from-scratch")
        self.assertEqual(recommend_recovery_action(["reward_logging_only"]), "resume-from-checkpoint")


if __name__ == "__main__":
    unittest.main()
