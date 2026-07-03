from __future__ import annotations

import json
import subprocess
import tempfile
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
LAUNCHER = REPO_ROOT / "tools" / "stage4" / "launch_stage4_ogm.ps1"


class Stage4LaunchResumeTests(unittest.TestCase):
    def _launch_dry_run(
        self,
        temp_dir: str,
        resume_checkpoint: Path | None = None,
        resume_state: Path | None = None,
        target_mode: str | None = None,
        extra_args: list[str] | None = None,
    ) -> dict:
        start_episode = "20000" if resume_checkpoint is not None or resume_state is not None else "0"
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
            start_episode,
            "-EvalEpisode",
            "25",
            "-ExpDirOverride",
            temp_dir,
            "-DryRun",
        ]
        if target_mode is not None:
            command.extend(["-TargetMode", target_mode])
        if resume_checkpoint is not None:
            command.extend(["-ResumeCheckpoint", str(resume_checkpoint)])
        if resume_state is not None:
            command.extend(["-ResumeState", str(resume_state)])
        if extra_args:
            command.extend(extra_args)

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
            self.assertIsNone(manifest["resume_state"])
            self.assertEqual(manifest["continuation_type"], "checkpoint_based")
            self.assertIn("--start_episode", runner_args)
            self.assertEqual(runner_args[runner_args.index("--start_episode") + 1], "20000")
            self.assertIn("--resume_checkpoint", runner_args)
            self.assertEqual(runner_args[runner_args.index("--resume_checkpoint") + 1], str(resume_checkpoint))
            self.assertNotIn("--resume_state", runner_args)
            self.assertIn(str(resume_checkpoint), manifest["runner_argument_string"])
            self.assertIn(f'"{resume_checkpoint}"', manifest["runner_argument_string"])
            self.assertIn("--run_dir", runner_args)
            self.assertIn(f'"{runner_args[runner_args.index("--run_dir") + 1]}"', manifest["runner_argument_string"])
            self.assertIn("-OutputPath", monitor_args)
            self.assertIn(f'"{monitor_args[monitor_args.index("-OutputPath") + 1]}"', manifest["monitor_argument_string"])

    def test_launcher_dry_run_records_full_state_resume_manifest_and_runner_args(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            resume_state = Path(temp_dir) / "resume states" / "stage4 state 19999.pt"
            manifest = self._launch_dry_run(temp_dir, resume_state=resume_state)
            runner_args = manifest["runner_args"]

            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["start_episode"], 20000)
            self.assertIsNone(manifest["resume_checkpoint"])
            self.assertEqual(manifest["resume_state"], str(resume_state))
            self.assertEqual(manifest["continuation_type"], "full_state")
            self.assertIn("--resume_state", runner_args)
            self.assertEqual(runner_args[runner_args.index("--resume_state") + 1], str(resume_state))
            self.assertNotIn("--resume_checkpoint", runner_args)
            self.assertIn(f'"{resume_state}"', manifest["runner_argument_string"])

    def test_launcher_dry_run_omits_resume_checkpoint_arg_for_fresh_run(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            manifest = self._launch_dry_run(temp_dir)
            runner_args = manifest["runner_args"]

            self.assertTrue(manifest["dry_run"])
            self.assertEqual(manifest["start_episode"], 0)
            self.assertIsNone(manifest["resume_checkpoint"])
            self.assertIsNone(manifest["resume_state"])
            self.assertEqual(manifest["continuation_type"], "fresh")
            self.assertNotIn("--resume_checkpoint", runner_args)
            self.assertNotIn("--resume_state", runner_args)

    def test_launcher_dry_run_records_target_mode_manifest_and_runner_arg(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            manifest = self._launch_dry_run(temp_dir, target_mode="corrected-cossin")
            runner_args = manifest["runner_args"]

            self.assertEqual(manifest["target_mode"], "corrected-cossin")
            self.assertIn("--target_mode", runner_args)
            self.assertEqual(runner_args[runner_args.index("--target_mode") + 1], "corrected-cossin")
            self.assertEqual(manifest["changed_knobs"]["target_mode"], "corrected-cossin")

    def test_launcher_dry_run_records_maneuver_stability_args_and_changed_knobs(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            manifest = self._launch_dry_run(
                temp_dir,
                extra_args=[
                    "-ManeuverStabilityMode",
                    "diagnostic",
                    "-ManeuverStabilityApplyTo",
                    "all",
                    "-ManeuverStabilityMinSpeed",
                    "0.002",
                    "-ManeuverStabilityHoldSteps",
                    "2",
                    "-ManeuverStabilityMaxHoldSpeed",
                    "0.5",
                ],
            )
            runner_args = manifest["runner_args"]
            changed_knobs = manifest["changed_knobs"]

            self.assertIn("--maneuver_stability_mode", runner_args)
            self.assertEqual(runner_args[runner_args.index("--maneuver_stability_mode") + 1], "diagnostic")
            self.assertIn("--maneuver_stability_apply_to", runner_args)
            self.assertEqual(runner_args[runner_args.index("--maneuver_stability_apply_to") + 1], "all")
            self.assertIn("--maneuver_stability_min_speed", runner_args)
            self.assertEqual(runner_args[runner_args.index("--maneuver_stability_min_speed") + 1], "0.002")
            self.assertIn("--maneuver_stability_hold_steps", runner_args)
            self.assertEqual(runner_args[runner_args.index("--maneuver_stability_hold_steps") + 1], "2")
            self.assertIn("--maneuver_stability_max_hold_speed", runner_args)
            self.assertEqual(runner_args[runner_args.index("--maneuver_stability_max_hold_speed") + 1], "0.5")
            self.assertEqual(changed_knobs["maneuver_stability_mode"], "diagnostic")
            self.assertEqual(changed_knobs["maneuver_stability_apply_to"], "all")
            self.assertEqual(float(changed_knobs["maneuver_stability_min_speed"]), 0.002)
            self.assertEqual(changed_knobs["maneuver_stability_hold_steps"], 2)
            self.assertEqual(float(changed_knobs["maneuver_stability_max_hold_speed"]), 0.5)

    def test_launcher_allows_corrected_target_mode_with_full_state_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            resume_state = Path(temp_dir) / "resume states" / "stage4 state 19999.pt"
            manifest = self._launch_dry_run(
                temp_dir,
                resume_state=resume_state,
                target_mode="corrected-cossin",
            )
            runner_args = manifest["runner_args"]

            self.assertEqual(manifest["target_mode"], "corrected-cossin")
            self.assertEqual(manifest["continuation_type"], "full_state")
            self.assertEqual(manifest["resume_state"], str(resume_state))
            self.assertIn("--resume_state", runner_args)
            self.assertEqual(runner_args[runner_args.index("--resume_state") + 1], str(resume_state))

    def test_launcher_rejects_corrected_target_mode_with_checkpoint_resume(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
            result = subprocess.run(
                [
                    "powershell",
                    "-NoProfile",
                    "-ExecutionPolicy",
                    "Bypass",
                    "-File",
                    str(LAUNCHER),
                    "-RunName",
                    "target mode test",
                    "-TrainEpisode",
                    "30000",
                    "-StartEpisode",
                    "20000",
                    "-EvalEpisode",
                    "25",
                    "-ExpDirOverride",
                    temp_dir,
                    "-DryRun",
                    "-TargetMode",
                    "corrected-cossin",
                    "-ResumeCheckpoint",
                    str(Path(temp_dir) / "SAC_19999.pt"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            manifests = list(Path(temp_dir).glob("*.meta.json"))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("TargetMode", result.stderr)
        self.assertIn("corrected-cossin", result.stderr)
        self.assertEqual(manifests, [])

    def test_launcher_dry_run_rejects_resume_state_and_resume_checkpoint_together(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
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
                    "-ExpDirOverride",
                    temp_dir,
                    "-DryRun",
                    "-ResumeCheckpoint",
                    str(Path(temp_dir) / "SAC_19999.pt"),
                    "-ResumeState",
                    str(Path(temp_dir) / "stage4_state_19999.pt"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ResumeState", result.stderr)
        self.assertIn("ResumeCheckpoint", result.stderr)

    def test_launcher_dry_run_rejects_resume_state_with_zero_start_episode_before_manifest(self) -> None:
        with tempfile.TemporaryDirectory(prefix="stage4 launch ") as temp_dir:
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
                    "0",
                    "-EvalEpisode",
                    "25",
                    "-ExpDirOverride",
                    temp_dir,
                    "-DryRun",
                    "-ResumeState",
                    str(Path(temp_dir) / "stage4_state_19999.pt"),
                ],
                capture_output=True,
                text=True,
                timeout=30,
            )

            manifests = list(Path(temp_dir).glob("*.meta.json"))

        self.assertNotEqual(result.returncode, 0)
        self.assertIn("ResumeState", result.stderr)
        self.assertIn("StartEpisode", result.stderr)
        self.assertEqual(manifests, [])


if __name__ == "__main__":
    unittest.main()
