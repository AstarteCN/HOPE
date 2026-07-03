from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import time
import unittest
from pathlib import Path

from tensorboard.compat.proto.event_pb2 import Event
from tensorboard.compat.proto.summary_pb2 import Summary
from tensorboard.summary.writer.event_file_writer import EventFileWriter


REPO_ROOT = Path(__file__).resolve().parents[3]
MONITOR = REPO_ROOT / "tools" / "stage4" / "monitor_stage4_ogm_progress.ps1"


def _powershell() -> str:
    powershell = shutil.which("powershell") or shutil.which("pwsh")
    if powershell is None:
        raise unittest.SkipTest("PowerShell executable is not available")
    return powershell


def _write_step_events(log_dir: Path, values: list[float], start_step: int = 0) -> None:
    writer = EventFileWriter(str(log_dir))
    try:
        for offset, value in enumerate(values):
            writer.add_event(
                Event(
                    wall_time=time.time(),
                    step=start_step + offset,
                    summary=Summary(
                        value=[Summary.Value(tag="step_num", simple_value=float(value))]
                    ),
                )
            )
        writer.flush()
    finally:
        writer.close()


def _gate_summary_path(output_dir: Path, gate_episode: int, suffix: str) -> Path:
    return output_dir / f"stage4_ogm_gate_{gate_episode}.tensorboard.{suffix}"


def _run_monitor(
    manifest_path: Path,
    run_dir: Path,
    gate_episode: int,
    output_dir: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    command = [
        _powershell(),
        "-NoProfile",
        "-ExecutionPolicy",
        "Bypass",
        "-File",
        str(MONITOR),
        "-ManifestPath",
        str(manifest_path),
        "-RunDir",
        str(run_dir),
        "-GateEpisode",
        str(gate_episode),
    ]
    if output_dir is not None:
        command.extend(["-OutputDir", str(output_dir)])
    return subprocess.run(
        command,
        capture_output=True,
        text=True,
        timeout=30,
    )


class Stage4MonitorProgressTests(unittest.TestCase):
    def test_output_dir_keeps_gate_summary_out_of_docs_research(self) -> None:
        gate_episode = 11
        docs_before = sorted(
            path.name
            for path in (REPO_ROOT / "docs" / "research").glob("stage4_ogm_gate_*.tensorboard.*")
        )

        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0])
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertTrue(_gate_summary_path(output_dir, gate_episode, "json").is_file())
            self.assertTrue(_gate_summary_path(output_dir, gate_episode, "md").is_file())

        docs_after = sorted(
            path.name
            for path in (REPO_ROOT / "docs" / "research").glob("stage4_ogm_gate_*.tensorboard.*")
        )
        self.assertEqual(docs_after, docs_before)

    def test_pending_gate_exits_zero_without_checkpoint(self) -> None:
        gate_episode = 5
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0])
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("episode_count: 3", result.stdout)
            self.assertIn("training_budget_met: false", result.stdout)
            self.assertIn("gate_ready: false", result.stdout)

    def test_manifest_relative_run_dir_resolves_from_manifest_directory(self) -> None:
        gate_episode = 5
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0])
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": "run"}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("gate_ready: false", result.stdout)

    def test_summary_failure_does_not_use_stale_pending_json(self) -> None:
        gate_episode = 7
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            stale_json = _gate_summary_path(output_dir, gate_episode, "json")
            stale_json.write_text(
                json.dumps(
                    {
                        "episode_count": 3,
                        "training_budget_met": False,
                        "env_step_count": 300,
                    }
                ),
                encoding="utf-8",
            )
            _gate_summary_path(output_dir, gate_episode, "md").write_text("stale\n", encoding="utf-8")
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertNotIn("gate_ready: false", result.stdout)

    def test_ready_gate_fails_when_checkpoint_is_missing(self) -> None:
        gate_episode = 3
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0])
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Expected checkpoint missing at gate 3", result.stderr)

    def test_ready_gate_fails_when_state_snapshot_is_missing(self) -> None:
        gate_episode = 3
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0])
            (run_dir / "SAC_2.pt").write_bytes(b"checkpoint")
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertNotEqual(result.returncode, 0, result.stdout)
            self.assertIn("Expected state snapshot missing at gate 3", result.stderr)

    def test_resumed_run_uses_global_step_to_detect_ready_gate(self) -> None:
        gate_episode = 20000
        with tempfile.TemporaryDirectory(prefix="stage4 monitor ") as temp_dir:
            temp_path = Path(temp_dir)
            run_dir = temp_path / "run"
            output_dir = temp_path / "summary"
            run_dir.mkdir()
            output_dir.mkdir()
            _write_step_events(run_dir, [100.0, 120.0, 122.0], start_step=19997)
            (run_dir / "SAC_19999.pt").write_bytes(b"checkpoint")
            (run_dir / "stage4_state_19999.pt").write_bytes(b"state")
            manifest_path = temp_path / "manifest.json"
            manifest_path.write_text(
                json.dumps({"run_dir": str(run_dir)}),
                encoding="utf-8",
            )

            result = _run_monitor(manifest_path, run_dir, gate_episode, output_dir=output_dir)

            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertIn("episode_count: 3", result.stdout)
            self.assertIn("gate_ready: true", result.stdout)
            self.assertIn("Checkpoint:", result.stdout)
            self.assertIn("StateSnapshot:", result.stdout)


if __name__ == "__main__":
    unittest.main()
