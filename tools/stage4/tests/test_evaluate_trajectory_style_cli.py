import hashlib
import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.stage4.debug_trace_export import build_fixture_trace
from tools.stage4.trajectory_style_report import write_json_reports


class EvaluateTrajectoryStyleCliTests(unittest.TestCase):
    def _write_trace(
        self,
        root: Path,
        *,
        case_uid: str = "parallel_001",
        parking_type: str = "parallel",
        name: str = "trace.json",
    ) -> Path:
        trace_path = root / name
        trace = build_fixture_trace(frame_count=4)
        trace["case"]["case_uid"] = case_uid
        trace["case"]["parking_type"] = parking_type
        trace["frames"][-1]["ego_state"]["x"] = 8.0
        trace_path.write_text(json.dumps(trace), encoding="utf-8")
        return trace_path

    def _run_cli(self, args: list[str]) -> subprocess.CompletedProcess[str]:
        return subprocess.run(
            [sys.executable, "tools/stage4/evaluate_trajectory_style.py", *args],
            cwd=Path(__file__).resolve().parents[3],
            text=True,
            capture_output=True,
            check=False,
        )

    def test_cli_writes_json_markdown_and_case_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            trace_path = self._write_trace(root)

            completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(output_dir),
                    "--slot-types",
                    "parallel",
                    "--write-markdown",
                    "--write-json",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report_json = output_dir / "style_eval_report.json"
            report_md = output_dir / "style_eval_report.md"
            unsupported_json = output_dir / "unsupported_cases.json"
            case_summary_json = output_dir / "case_summaries" / "parallel_001.json"
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            self.assertTrue(unsupported_json.exists())
            self.assertTrue(case_summary_json.exists())
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(payload["supported_case_count"], 1)
            self.assertEqual(payload["cases"][0]["case_uid"], "parallel_001")
            case_payload = json.loads(case_summary_json.read_text(encoding="utf-8"))
            self.assertEqual(case_payload["case_uid"], "parallel_001")
            self.assertIn("parallel", report_md.read_text(encoding="utf-8"))

    def test_case_summary_names_cannot_escape_output_dir(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            trace_path = self._write_trace(root, case_uid=r"..\escape")

            completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(output_dir),
                    "--slot-types",
                    "parallel",
                    "--write-json",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            self.assertFalse((output_dir / "escape.json").exists())
            case_files = list((output_dir / "case_summaries").glob("*.json"))
            self.assertEqual(len(case_files), 1)
            self.assertNotIn("..", case_files[0].name)
            self.assertEqual(
                json.loads(case_files[0].read_text(encoding="utf-8"))["case_uid"],
                r"..\escape",
            )

    def test_unsupported_allowed_remains_in_cases_and_case_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            output_dir = root / "out"
            trace_path = self._write_trace(root, case_uid="perp_001", parking_type="perpendicular")

            completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(output_dir),
                    "--slot-types",
                    "parallel",
                    "--allow-unsupported",
                    "--write-json",
                ]
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            payload = json.loads((output_dir / "style_eval_report.json").read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(payload["supported_case_count"], 0)
            self.assertEqual(payload["unsupported_case_count"], 1)
            self.assertEqual(len(payload["cases"]), 1)
            self.assertEqual(payload["cases"][0]["unsupported_reason"], "slot_type filtered out: perpendicular")
            self.assertEqual(payload["top_style_mismatches"], [])
            self.assertTrue((output_dir / "case_summaries" / "perp_001.json").exists())

    def test_duplicate_safe_case_summary_names_raise(self):
        unsafe_uid = r"..\escape"
        colliding_uid = "escape_%s" % hashlib.sha1(unsafe_uid.encode("utf-8")).hexdigest()[:10]
        report = {
            "cases": [
                {"case_uid": unsafe_uid},
                {"case_uid": colliding_uid},
            ],
            "unsupported_cases": [],
        }
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaises(ValueError):
                write_json_reports(report, Path(temp_dir) / "out")

    def test_strict_unsupported_exits_two_without_allow_unsupported(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = self._write_trace(root, case_uid="perp_001", parking_type="perpendicular")

            completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(root / "out"),
                    "--slot-types",
                    "parallel",
                    "--strict",
                    "--write-json",
                ]
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("unsupported", completed.stderr)

    def test_case_filter_zero_matches_exits_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = self._write_trace(root)

            completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(root / "out"),
                    "--slot-types",
                    "parallel",
                    "--case-filter",
                    "does-not-exist",
                ]
            )

            self.assertEqual(completed.returncode, 2)
            self.assertIn("matched zero traces", completed.stderr)

    def test_unimplemented_optional_flags_exit_two(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = self._write_trace(root)
            annotations_path = root / "annotations.json"
            annotations_path.write_text("{}", encoding="utf-8")

            reference_completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(root / "out-reference"),
                    "--slot-types",
                    "parallel",
                    "--reference-annotations",
                    str(annotations_path),
                ]
            )
            visualization_completed = self._run_cli(
                [
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(root / "out-visualization"),
                    "--slot-types",
                    "parallel",
                    "--visualization-dir",
                    str(root / "viz"),
                ]
            )

            self.assertEqual(reference_completed.returncode, 2)
            self.assertIn("not implemented", reference_completed.stderr)
            self.assertEqual(visualization_completed.returncode, 2)
            self.assertIn("not implemented", visualization_completed.stderr)


if __name__ == "__main__":
    unittest.main()
