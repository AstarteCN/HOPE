import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tools.stage3.compare_stage3_20k import classify_decision, build_comparison_report, main


BASELINE_LEVEL_EVAL = {
    "Normal": 0.985,
    "Complex": 0.945,
    "Extrem": 0.655,
    "DLP": 0.960,
}


def write_resource_csv(path: Path) -> None:
    path.write_text(
        "timestamp,pid,process_name,cpu_percent,working_set_mb,private_memory_mb,gpu_util_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_power_w\n"
        "2026-06-19T00:00:00,1,python,10,100,80,20,1000,12000,50\n"
        "2026-06-19T00:00:05,1,python,30,120,90,40,1500,12000,60\n",
        encoding="utf-8",
    )


def write_eval_root(path: Path, eval_success: dict[str, float]) -> None:
    scene_dirs = {
        "Normal": "normalize",
        "Complex": "complex",
        "Extrem": "extreme",
        "DLP": "dlp",
    }
    for scene, directory in scene_dirs.items():
        result_dir = path / directory
        result_dir.mkdir(parents=True)
        (result_dir / "result.txt").write_text(
            f"success rate: {eval_success[scene]}\nstep num: 90.0 +-(1.0)\n",
            encoding="utf-8",
        )


def manifest() -> dict:
    return {
        "baseline_id": "stage3_command_only_20k_20260619",
        "candidate_type": "20k_validation",
        "run_dir": r"D:\Github\HOPE\src\log\exp\sac_candidate",
        "changed_knobs": {"resource_monitor": "enabled"},
        "command": ["python", "train_HOPE_sac.py", "--train_episode", "20000"],
    }


def tensorboard_summary(
    *,
    has_nonfinite: bool = False,
    hard_reject_has_nonfinite: bool = False,
    scene_scalars: dict[str, dict[str, float]] | None = None,
) -> dict:
    return {
        "has_nonfinite": has_nonfinite,
        "hard_reject_has_nonfinite": hard_reject_has_nonfinite,
        "nonfinite_scalar_tags": ["success_rate_Extrem"] if has_nonfinite else [],
        "hard_reject_nonfinite_scalar_tags": ["actor_loss"] if hard_reject_has_nonfinite else [],
        "scalars": scene_scalars or {
            "success_rate_Normal": {"mean_last100": 1.0, "recent_mean": 1.0, "latest": 1.0},
            "success_rate_Complex": {"mean_last100": 0.94, "recent_mean": 0.94, "latest": 0.94},
            "success_rate_Extrem": {"mean_last100": 0.71, "recent_mean": 0.70, "latest": 0.69},
            "success_rate_dlp": {"mean_last100": 0.79, "recent_mean": 0.78, "latest": 0.77},
        },
    }


class CompareStage320kTests(unittest.TestCase):
    def test_classify_decision_passes_when_quality_and_speed_pass(self):
        decision = classify_decision({"quality_pass": True, "hard_reject_reasons": []}, {"speed_pass": True})
        self.assertEqual(decision, "pass")

    def test_classify_decision_labels_speed_neutral_when_quality_passes_and_speed_fails(self):
        decision = classify_decision({"quality_pass": True, "hard_reject_reasons": []}, {"speed_pass": False})
        self.assertEqual(decision, "quality-pass-speed-neutral")

    def test_classify_decision_rejects_clear_quality_threshold_failure(self):
        decision = classify_decision({"quality_pass": False, "hard_reject_reasons": []}, {"speed_pass": True})
        self.assertEqual(decision, "reject")

    def test_classify_decision_investigates_when_quality_passes_with_investigate_reasons(self):
        decision = classify_decision(
            {
                "quality_pass": True,
                "hard_reject_reasons": [],
                "investigate_reasons": ["review_tensorboard_trend"],
            },
            {"speed_pass": True},
        )
        self.assertEqual(decision, "investigate")

    def test_build_comparison_report_passes_for_baseline_level_eval_speed_pass_and_no_hard_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="pass",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "pass")
        self.assertTrue(report["quality"]["quality_pass"])
        self.assertTrue(report["speed"]["speed_pass"])
        self.assertEqual(report["baseline_id"], "stage3_command_only_20k_20260619")
        self.assertEqual(report["candidate"]["candidate_type"], "20k_validation")
        self.assertEqual(report["resource_profile"]["sample_count"], 2)
        self.assertAlmostEqual(report["scene_eval"]["mean_success"], 0.88625)
        self.assertEqual(report["parity_status"], "pass")
        self.assertEqual(report["command"], ["python", "train_HOPE_sac.py", "--train_episode", "20000"])

    def test_top_level_tensorboard_nonfinite_without_hard_reject_does_not_reject(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(has_nonfinite=True, hard_reject_has_nonfinite=False),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "pass")
        self.assertTrue(report["tensorboard"]["has_nonfinite"])
        self.assertFalse(report["tensorboard"]["hard_reject_has_nonfinite"])
        self.assertNotIn("nonfinite_tensorboard_metric", report["quality"]["hard_reject_reasons"])

    def test_hard_reject_tensorboard_nonfinite_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(has_nonfinite=True, hard_reject_has_nonfinite=True),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertFalse(report["quality"]["quality_pass"])
        self.assertIn("nonfinite_tensorboard_metric", report["quality"]["hard_reject_reasons"])

    def test_clear_eval_threshold_quality_failure_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(),
                resource_csv_path=resource_csv_path,
                eval_success={"Normal": 0.40, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960},
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertFalse(report["quality"]["quality_pass"])
        self.assertEqual(report["quality"]["hard_reject_reasons"], [])

    def test_multi_scene_collapse_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(
                    scene_scalars={
                        "success_rate_Normal": {"mean_last100": 0.40, "recent_mean": 1.0, "latest": 1.0},
                        "success_rate_Complex": {"recent_mean": 0.45, "latest": 1.0},
                        "success_rate_Extrem": {"mean_last100": 0.71, "recent_mean": 0.70, "latest": 0.69},
                        "success_rate_dlp": {"latest": 0.77},
                    }
                ),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertTrue(report["tensorboard"]["multi_scene_collapse"])
        self.assertIn("multi_scene_collapse", report["quality"]["hard_reject_reasons"])

    def test_severe_normal_tensorboard_collapse_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(
                    scene_scalars={
                        "success_rate_Normal": {"mean_last100": 0.40, "recent_mean": 1.0, "latest": 1.0},
                        "success_rate_Complex": {"mean_last100": 0.94, "recent_mean": 0.94, "latest": 0.94},
                        "success_rate_Extrem": {"mean_last100": 0.71, "recent_mean": 0.70, "latest": 0.69},
                        "success_rate_dlp": {"mean_last100": 0.79, "recent_mean": 0.78, "latest": 0.77},
                    }
                ),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertIn("severe_core_scene_collapse_Normal", report["quality"]["hard_reject_reasons"])

    def test_severe_complex_tensorboard_collapse_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(
                    scene_scalars={
                        "success_rate_Normal": {"mean_last100": 1.0, "recent_mean": 1.0, "latest": 1.0},
                        "success_rate_Complex": {"mean_last100": 0.45, "recent_mean": 0.94, "latest": 0.94},
                        "success_rate_Extrem": {"mean_last100": 0.71, "recent_mean": 0.70, "latest": 0.69},
                        "success_rate_dlp": {"mean_last100": 0.79, "recent_mean": 0.78, "latest": 0.77},
                    }
                ),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertIn("severe_core_scene_collapse_Complex", report["quality"]["hard_reject_reasons"])

    def test_checkpoint_missing_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="not-required",
                checkpoint_missing=True,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertIn("missing_20k_checkpoint", report["quality"]["hard_reject_reasons"])

    def test_parity_failure_rejects_report(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv_path = Path(tmp) / "resources.csv"
            write_resource_csv(resource_csv_path)

            report = build_comparison_report(
                manifest=manifest(),
                tensorboard_summary=tensorboard_summary(),
                resource_csv_path=resource_csv_path,
                eval_success=BASELINE_LEVEL_EVAL,
                candidate_speed={
                    "wall_time_hours": 12.964,
                    "episodes_per_hour": 1800.0,
                    "env_steps_per_second": 39.10,
                },
                parity_status="fail",
                checkpoint_missing=False,
            )

        self.assertEqual(report["decision"], "reject")
        self.assertIn("parity_failed", report["quality"]["hard_reject_reasons"])

    def test_cli_writes_reports_and_updates_manifest(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            manifest_path = root / "manifest.json"
            tensorboard_path = root / "tensorboard.json"
            resource_csv_path = root / "resources.csv"
            eval_root = root / "eval"
            json_path = root / "comparison.json"
            markdown_path = root / "comparison.md"

            manifest_path.write_text(json.dumps(manifest()), encoding="utf-8")
            tensorboard_path.write_text(
                json.dumps(tensorboard_summary(has_nonfinite=True, hard_reject_has_nonfinite=False)),
                encoding="utf-8",
            )
            write_resource_csv(resource_csv_path)
            write_eval_root(eval_root, BASELINE_LEVEL_EVAL)

            argv = [
                "compare_stage3_20k.py",
                "--manifest",
                str(manifest_path),
                "--tensorboard-json",
                str(tensorboard_path),
                "--resource-csv",
                str(resource_csv_path),
                "--eval-root",
                str(eval_root),
                "--wall-time-hours",
                "12.964",
                "--episodes-per-hour",
                "1800.0",
                "--env-steps-per-second",
                "39.10",
                "--parity-status",
                "pass",
                "--json",
                str(json_path),
                "--markdown",
                str(markdown_path),
            ]
            with patch.object(sys, "argv", argv):
                result = main()

            updated_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            report = json.loads(json_path.read_text(encoding="utf-8"))
            markdown = markdown_path.read_text(encoding="utf-8")

            self.assertEqual(result, 0)
            self.assertTrue(json_path.exists())
            self.assertTrue(markdown_path.exists())
            self.assertEqual(report["decision"], "pass")
            self.assertEqual(updated_manifest["gate_status"], "pass")
            self.assertEqual(updated_manifest["comparison_report_json"], str(json_path))
            self.assertEqual(updated_manifest["comparison_report_markdown"], str(markdown_path))
            self.assertIn("## Resource Summary", markdown)
            self.assertIn("Sample count", markdown)
            self.assertIn("Avg process CPU percent", markdown)
            self.assertIn("Avg whole GPU util percent", markdown)
            self.assertIn("## TensorBoard", markdown)
            self.assertIn("Has nonfinite", markdown)
            self.assertIn("Hard-reject has nonfinite", markdown)
            self.assertIn("Nonfinite scalar tags", markdown)
            self.assertIn("Hard-reject nonfinite scalar tags", markdown)
            self.assertIn("Multi-scene collapse", markdown)


if __name__ == "__main__":
    unittest.main()
