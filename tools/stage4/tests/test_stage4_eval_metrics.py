from __future__ import annotations

import json
import unittest

from tools.stage4.eval_stage4_ogm_checkpoint import (
    count_gear_shifts,
    summarize_eval_records,
)


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


if __name__ == "__main__":
    unittest.main()
