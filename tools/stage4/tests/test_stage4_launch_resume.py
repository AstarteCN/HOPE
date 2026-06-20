from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
