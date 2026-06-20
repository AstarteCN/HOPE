from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tools" / "stage4" / "launch_stage4_ogm.ps1"


class Stage4LaunchResumeTests(unittest.TestCase):
    def _launch_dry_run(self, temp_dir: str, resume_checkpoint: Path | None = None) -> dict:
        command = [
            "powershell",
            "-NoProfile",
            "-ExecutionPolicy",
            "Bypass",
            "-File",
            str(LAUNCHER),
            "-RunName",
            "resume test",
            "-TrainEpisode",
            "30000",
            "-StartEpisode",
            "20000" if resume_checkpoint is not None else "0",
            "-EvalEpisode",
            "25",
            "-ExpDirOverride",
            temp_dir,
            "-DryRun",
        ]
        if resume_checkpoint is not None:
            command.extend(["-ResumeCheckpoint", str(resume_checkpoint)])

        result = subprocess.run(command, capture_output=True, text=True, timeout=30)

        self.assertEqual(result.returncode, 0, result.stderr)
        manifest_path = Path(result.stdout.strip().splitlines()[-1])
        self.assertEqual(manifest_path.parent, Path(temp_dir))
        return json.loads(manifest_path.read_text(encoding="utf-8-sig"))

    def test_launcher_dry_run_records_resume_manifest_and_quoted_runner_args_with_spaces(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            resume_checkpoint = Path(temp_dir) / "resume checkpoints" / "SAC 19999.pt"
            manifest = self._launch_dry_run(temp_dir, resume_checkpoint=resume_checkpoint)
            runner_args = manifest["runner_args"]
            monitor_args = manifest["monitor_args"]

            self.assertTrue(manifest["dry_run"])
            self.assertIsNone(manifest["workload_pid"])
            self.assertEqual(manifest["train_episode"], 30000)
            self.assertEqual(manifest["start_episode"], 20000)
            self.assertEqual(manifest["eval_episode"], 25)
            self.assertEqual(manifest["resume_checkpoint"], str(resume_checkpoint))
            self.assertIn("--start_episode", runner_args)
            self.assertEqual(runner_args[runner_args.index("--start_episode") + 1], "20000")
            self.assertIn("--resume_checkpoint", runner_args)
            self.assertEqual(runner_args[runner_args.index("--resume_checkpoint") + 1], str(resume_checkpoint))
            self.assertIn(str(resume_checkpoint), manifest["runner_argument_string"])
            self.assertIn(f'"{resume_checkpoint}"', manifest["runner_argument_string"])
            self.assertIn("--run_dir", runner_args)
            self.assertIn(f'"{runner_args[runner_args.index("--run_dir") + 1]}"', manifest["runner_argument_string"])
            self.assertIn("-OutputPath", monitor_args)
            self.assertIn(f'"{monitor_args[monitor_args.index("-OutputPath") + 1]}"', manifest["monitor_argument_string"])

    def test_launcher_dry_run_omits_resume_checkpoint_arg_for_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            manifest = self._launch_dry_run(temp_dir)
            runner_args = manifest["runner_args"]

            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["start_episode"], 0)
            self.assertIsNone(manifest["resume_checkpoint"])
            self.assertNotIn("--resume_checkpoint", runner_args)


if __name__ == "__main__":
    unittest.main()
