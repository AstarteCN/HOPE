from __future__ import annotations

import json
import inspect
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from tools.stage4.eval_stage4_ogm_checkpoint import (
    apply_maneuver_stability_to_eval_action,
    build_eval_result,
    count_gear_shifts,
    evaluate_checkpoint,
    get_eval_action_with_maneuver_stability,
    resolve_checkpoint_maneuver_stability_config,
    summarize_eval_records,
)
from tools.stage4.stage4_maneuver_stability import (
    MODE_DIAGNOSTIC,
    MODE_DIRECTION_HOLD,
    MODE_OFF,
    ManeuverStabilityConfig,
    ManeuverStabilityTracker,
)
from tools.stage4.stage4_target_transform import (
    TARGET_MODE_AUTO,
    TARGET_MODE_CORRECTED_COSSIN,
    TARGET_MODE_LEGACY_COSCOS,
    resolve_checkpoint_target_mode,
)


class _FinalRsActionAgent:
    def __init__(self, action: np.ndarray) -> None:
        self.action = action
        self.executing_rs = True

    def get_action(self, _obs):
        self.executing_rs = False
        return self.action, None


class Stage4EvalMetricTests(unittest.TestCase):
    def test_count_gear_shifts_counts_speed_sign_changes_ignoring_zero(self) -> None:
        self.assertEqual(count_gear_shifts([1.0, 1.0, -1.0, -1.0, 0.0, 1.0]), 2)
        self.assertEqual(count_gear_shifts([0.0, 0.0, 1.0, 1.0]), 0)
        self.assertEqual(count_gear_shifts([-1.0, -1.0, -1.0]), 0)

    def test_summarize_eval_records_reports_psr_angs_and_pl_by_split(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
            {"split": "Sim-Normal", "success": False, "gear_shifts": 3, "path_length": 30.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]
        summary = summarize_eval_records(records)
        self.assertAlmostEqual(summary["Sim-Normal"]["psr"], 0.5)
        self.assertAlmostEqual(summary["Sim-Normal"]["angs"], 1.0)
        self.assertAlmostEqual(summary["Sim-Normal"]["pl"], 10.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["psr"], 1.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["angs"], 2.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["pl"], 20.0)

    def test_summarize_eval_records_reports_no_success_metrics_as_none(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": False, "gear_shifts": 3, "path_length": 30.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]
        summary = summarize_eval_records(records)
        self.assertAlmostEqual(summary["Sim-Normal"]["psr"], 0.0)
        self.assertIsNone(summary["Sim-Normal"]["angs"])
        self.assertIsNone(summary["Sim-Normal"]["pl"])

    def test_summarize_eval_records_raises_when_expected_split_is_missing(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
        ]
        with self.assertRaisesRegex(ValueError, "Missing eval records for expected split: Sim-Complex"):
            summarize_eval_records(records)

    def test_no_success_summary_serializes_as_strict_json(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": False, "gear_shifts": 3, "path_length": 30.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]
        summary = summarize_eval_records(records)
        serialized = json.dumps(summary, allow_nan=False)
        self.assertIn('"angs": null', serialized)

    def test_evaluate_checkpoint_uses_auto_target_mode_by_default(self) -> None:
        signature = inspect.signature(evaluate_checkpoint)

        self.assertEqual(signature.parameters["target_mode"].default, TARGET_MODE_AUTO)

    def test_build_eval_result_records_target_mode_metadata(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]

        result = build_eval_result(records, TARGET_MODE_CORRECTED_COSSIN)

        self.assertEqual(result["target_mode"], TARGET_MODE_CORRECTED_COSSIN)
        self.assertEqual(result["summary"], summarize_eval_records(records))

    def test_build_eval_result_records_maneuver_metadata(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]

        result = build_eval_result(
            records,
            TARGET_MODE_CORRECTED_COSSIN,
            maneuver_stability_config=ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC),
            maneuver_stability_summary={"counters": {"interventions": 0}},
        )

        self.assertEqual(result["maneuver_stability_config"]["mode"], MODE_DIAGNOSTIC)
        self.assertEqual(result["maneuver_stability_summary"]["counters"]["interventions"], 0)

    def test_build_eval_result_omits_off_maneuver_metadata(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]

        result = build_eval_result(
            records,
            TARGET_MODE_CORRECTED_COSSIN,
            maneuver_stability_config=ManeuverStabilityConfig(mode=MODE_OFF),
            maneuver_stability_summary={"counters": {"steps": 0}},
        )

        self.assertNotIn("maneuver_stability_config", result)
        self.assertNotIn("maneuver_stability_summary", result)

    def test_eval_action_helper_preserves_off_action(self) -> None:
        tracker = ManeuverStabilityTracker()
        action = np.array([0.1, -1.0], dtype=np.float64)

        result = apply_maneuver_stability_to_eval_action(
            tracker=tracker,
            action=action,
            source="RL",
        )

        self.assertFalse(result.changed)
        np.testing.assert_array_equal(result.applied_action, action)
        self.assertEqual(tracker.snapshot()["counters"]["steps"], 0)

    def test_eval_action_helper_applies_direction_hold(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=1)
        )
        apply_maneuver_stability_to_eval_action(
            tracker=tracker,
            action=np.array([0.1, 1.0], dtype=np.float64),
            source="RL",
        )

        result = apply_maneuver_stability_to_eval_action(
            tracker=tracker,
            action=np.array([0.1, -1.0], dtype=np.float64),
            source="RL",
        )

        self.assertTrue(result.changed)
        self.assertGreater(result.applied_action[1], 0.0)
        self.assertEqual(tracker.snapshot()["counters"]["interventions"], 1)

    def test_eval_action_helper_uses_pre_get_action_rs_source_for_final_rs_action(self) -> None:
        tracker = ManeuverStabilityTracker(
            ManeuverStabilityConfig(mode=MODE_DIRECTION_HOLD, hold_steps=1)
        )
        tracker.apply(np.array([0.1, 1.0], dtype=np.float64), source="RL")
        parking_agent = _FinalRsActionAgent(np.array([0.1, -1.0], dtype=np.float64))

        action, result = get_eval_action_with_maneuver_stability(
            parking_agent=parking_agent,
            tracker=tracker,
            obs={"target": np.zeros(5)},
        )

        self.assertEqual(result.source, "RS")
        self.assertFalse(result.changed)
        self.assertLess(action[1], 0.0)
        self.assertEqual(result.reason, "source_not_enabled")

    def test_auto_target_mode_preserves_legacy_when_no_state_metadata_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 eval target mode ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_best.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")

            mode, source = resolve_checkpoint_target_mode(checkpoint_path, TARGET_MODE_AUTO)

        self.assertEqual(mode, TARGET_MODE_LEGACY_COSCOS)
        self.assertEqual(source["type"], "legacy_default")

    def test_auto_target_mode_uses_matching_state_metadata_when_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 eval target mode ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_19999.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")
            torch.save({"target_mode": TARGET_MODE_CORRECTED_COSSIN}, Path(temp_dir) / "stage4_state_19999.pt")

            mode, source = resolve_checkpoint_target_mode(checkpoint_path, TARGET_MODE_AUTO)

        self.assertEqual(mode, TARGET_MODE_CORRECTED_COSSIN)
        self.assertEqual(source["type"], "stage4_state")

    def test_auto_maneuver_config_preserves_off_when_no_state_metadata_exists(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 eval maneuver ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_best.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")

            config, source = resolve_checkpoint_maneuver_stability_config(checkpoint_path)

        self.assertEqual(config.mode, MODE_OFF)
        self.assertEqual(source["type"], "legacy_default")

    def test_auto_maneuver_config_uses_matching_state_metadata_when_available(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 eval maneuver ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_19999.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")
            torch.save(
                {
                    "maneuver_stability_config": ManeuverStabilityConfig(
                        mode=MODE_DIAGNOSTIC,
                        min_speed=0.05,
                        max_hold_speed=0.5,
                    ).to_dict()
                },
                Path(temp_dir) / "stage4_state_19999.pt",
            )

            config, source = resolve_checkpoint_maneuver_stability_config(checkpoint_path)

        self.assertEqual(config.mode, MODE_DIAGNOSTIC)
        self.assertAlmostEqual(config.min_speed, 0.05)
        self.assertAlmostEqual(config.max_hold_speed, 0.5)
        self.assertEqual(source["type"], "stage4_state")

    def test_explicit_off_maneuver_config_overrides_state_metadata(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 eval maneuver ") as temp_dir:
            checkpoint_path = Path(temp_dir) / "SAC_19999.pt"
            checkpoint_path.write_text("not loaded", encoding="utf-8")
            torch.save(
                {"maneuver_stability_config": ManeuverStabilityConfig(mode=MODE_DIAGNOSTIC).to_dict()},
                Path(temp_dir) / "stage4_state_19999.pt",
            )

            config, source = resolve_checkpoint_maneuver_stability_config(
                checkpoint_path,
                requested_config=ManeuverStabilityConfig(mode=MODE_OFF),
            )

        self.assertEqual(config.mode, MODE_OFF)
        self.assertEqual(source["type"], "explicit")


if __name__ == "__main__":
    unittest.main()
