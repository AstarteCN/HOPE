from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tools" / "stage4" / "launch_stage4_ogm.ps1"


class Stage4LaunchResumeTests(unittest.TestCase):
    def test_launcher_exposes_resume_parameters_and_forwards_them_to_runner(self) -> None:
        launcher_text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("[string]$ResumeCheckpoint", launcher_text)
        self.assertIn("[int]$StartEpisode = 0", launcher_text)
        self.assertIn("'--start_episode'", launcher_text)
        self.assertIn("'--resume_checkpoint'", launcher_text)

    def test_launcher_manifest_records_resume_parameters(self) -> None:
        launcher_text = LAUNCHER.read_text(encoding="utf-8")

        self.assertIn("resume_checkpoint = $ResumeCheckpoint", launcher_text)
        self.assertIn("start_episode = $StartEpisode", launcher_text)

    def test_launcher_dry_run_records_resume_manifest_and_runner_args_without_starting_training(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            resume_checkpoint = Path(temp_dir) / "SAC_19999.pt"
            result = subprocess.run(
                [
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
                    "20000",
                    "-EvalEpisode",
                    "25",
                    "-ResumeCheckpoint",
                    str(resume_checkpoint),
                    "-ExpDirOverride",
                    temp_dir,
                    "-DryRun",
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            self.assertEqual(result.returncode, 0, result.stderr)
            manifest_path = Path(result.stdout.strip().splitlines()[-1])
            self.assertEqual(manifest_path.parent, Path(temp_dir))

            manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
            runner_args = manifest["runner_args"]

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


if __name__ == "__main__":
    unittest.main()
