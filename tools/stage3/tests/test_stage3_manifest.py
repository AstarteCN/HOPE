import json
import subprocess
import tempfile
import unittest
from pathlib import Path

from tools.stage3.stage3_manifest import (
    build_manifest,
    protected_source_diff,
    read_manifest,
    update_manifest,
    write_manifest,
)


REPO_ROOT = Path(__file__).resolve().parents[3]


class Stage3ManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            run_dir = REPO_ROOT / "src" / "log" / "exp" / "sac_example"
            manifest = build_manifest(
                repo_root=REPO_ROOT,
                run_name="candidate_render_headless",
                candidate_type="20k_validation",
                changed_knobs={"render_mode": "rgb_array"},
                command=["python", "train.py"],
                python_executable=str(REPO_ROOT / ".venv" / "Scripts" / "python.exe"),
                env={"SDL_VIDEODRIVER": "dummy"},
                run_dir=str(run_dir),
                stdout_path=str(run_dir.with_name(f"{run_dir.name}.stdout.log")),
                stderr_path=str(run_dir.with_name(f"{run_dir.name}.stderr.log")),
                resource_csv_path=str(run_dir.with_name(f"{run_dir.name}.resources.csv")),
            )
            write_manifest(manifest, path)
            update_manifest(path, {"gate_status": "running"})
            loaded = read_manifest(path)
            required_keys = {
                "baseline_id",
                "git_branch",
                "git_status_sb",
                "protected_source_diff",
                "expected_checkpoint_20k",
                "launcher_pid",
                "workload_pid",
                "resource_monitor_pid",
            }
            self.assertLessEqual(required_keys, set(loaded))
            self.assertEqual(loaded["run_name"], "candidate_render_headless")
            self.assertEqual(loaded["candidate_type"], "20k_validation")
            self.assertEqual(loaded["changed_knobs"], {"render_mode": "rgb_array"})
            self.assertEqual(loaded["gate_status"], "running")
            json.dumps(loaded)

    def test_protected_source_diff_returns_text(self):
        diff_text = protected_source_diff(REPO_ROOT)
        self.assertIsInstance(diff_text, str)

    def test_protected_source_diff_includes_staged_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            repo_root = Path(tmp)
            self._git(repo_root, "init")
            self._git(repo_root, "config", "user.email", "stage3-test@example.invalid")
            self._git(repo_root, "config", "user.name", "Stage3 Test")

            example_path = repo_root / "src" / "train" / "example.py"
            example_path.parent.mkdir(parents=True, exist_ok=True)
            example_path.write_text("value = 1\n", encoding="utf-8")
            self._git(repo_root, "add", "src/train/example.py")
            self._git(repo_root, "commit", "-m", "initial")

            example_path.write_text("value = 2\n", encoding="utf-8")
            self._git(repo_root, "add", "src/train/example.py")

            diff_text = protected_source_diff(repo_root)
            self.assertIn("value = 2", diff_text)

    def _git(self, repo_root: Path, *args: str) -> None:
        subprocess.run(
            ["git", *args],
            cwd=repo_root,
            check=True,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )


if __name__ == "__main__":
    unittest.main()
