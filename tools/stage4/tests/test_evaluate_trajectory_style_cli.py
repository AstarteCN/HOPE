import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from tools.stage4.debug_trace_export import build_fixture_trace


class EvaluateTrajectoryStyleCliTests(unittest.TestCase):
    def test_cli_writes_json_markdown_and_case_summaries(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            trace_path = root / "trace.json"
            output_dir = root / "out"
            trace = build_fixture_trace(frame_count=4)
            trace["case"]["case_uid"] = "parallel_001"
            trace["case"]["parking_type"] = "parallel"
            trace["frames"][-1]["ego_state"]["x"] = 8.0
            trace_path.write_text(json.dumps(trace), encoding="utf-8")

            completed = subprocess.run(
                [
                    sys.executable,
                    "tools/stage4/evaluate_trajectory_style.py",
                    "--trace-json",
                    str(trace_path),
                    "--output-dir",
                    str(output_dir),
                    "--slot-types",
                    "parallel",
                    "--write-markdown",
                    "--write-json",
                ],
                cwd=Path(__file__).resolve().parents[3],
                text=True,
                capture_output=True,
                check=False,
            )

            self.assertEqual(completed.returncode, 0, completed.stderr)
            report_json = output_dir / "style_eval_report.json"
            report_md = output_dir / "style_eval_report.md"
            unsupported_json = output_dir / "unsupported_cases.json"
            self.assertTrue(report_json.exists())
            self.assertTrue(report_md.exists())
            self.assertTrue(unsupported_json.exists())
            payload = json.loads(report_json.read_text(encoding="utf-8"))
            self.assertEqual(payload["case_count"], 1)
            self.assertEqual(payload["supported_case_count"], 1)
            self.assertEqual(payload["cases"][0]["case_uid"], "parallel_001")
            self.assertIn("parallel", report_md.read_text(encoding="utf-8"))


if __name__ == "__main__":
    unittest.main()
