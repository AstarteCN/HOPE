# HOPE Stage 3 Safe-Speed 20K Framework Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build opt-in Stage 3 tooling that profiles, launches, stops, evaluates, and gates 20K HOPE speed experiments against the locked command-only baseline without changing the original HOPE training framework.

**Architecture:** Keep all new behavior outside `src/train`, `src/env`, and `src/model` by adding focused tools under `tools/stage3/`. The framework records manifests, profiles hot paths, runs semantic parity checks, launches original training commands, stops at the 20K checkpoint, parses TensorBoard/resource/eval artifacts, and writes comparison reports under `docs/research/`.

**Tech Stack:** Windows PowerShell, project `.venv` Python, Python stdlib `argparse`/`json`/`csv`/`unittest`, existing TensorBoard event reader, existing HOPE SAC training/evaluation scripts, existing `monitor_stage3_resources.ps1`.

---

## Source Inputs

- PRD: `D:\Github\HOPE\docs\superpowers\specs\2026-06-19-hope-stage3-safe-speed-20k-prd.md`
- Research basis: `D:\Github\HOPE\docs\research\2026-06-19-stage3-safe-speed-deep-research.md`
- Locked 20K baseline run: `D:\Github\HOPE\src\log\exp\sac_20260619_004316`
- Locked 20K baseline checkpoint: `D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt`
- Original training entry point: `D:\Github\HOPE\src\train\train_HOPE_sac.py`
- Original evaluation entry point: `D:\Github\HOPE\src\evaluation\eval_mix_scene.py`

## Protected Boundaries

This plan creates tools and documentation only. It must leave the original HOPE source paths unchanged:

- `D:\Github\HOPE\src\train`
- `D:\Github\HOPE\src\env`
- `D:\Github\HOPE\src\model`

Every task that runs verification includes:

```powershell
git diff -- src/train src/env src/model
```

Expected: no output.

## File Structure

- Create `D:\Github\HOPE\tools\__init__.py`
  - Marks `tools` as importable for test execution from the repository root.
- Create `D:\Github\HOPE\tools\stage3\__init__.py`
  - Marks `tools.stage3` as importable.
- Create `D:\Github\HOPE\tools\stage3\stage3_baseline.py`
  - Stores the locked 20K baseline constants, pass thresholds, scene labels, protected source paths, and gate helper functions.
- Create `D:\Github\HOPE\tools\stage3\stage3_manifest.py`
  - Creates, reads, writes, and updates run manifests; captures Git status and protected-source diffs.
- Modify `D:\Github\HOPE\tools\stage3\tensorboard_stage3_summary.py`
  - Preserves current CLI behavior and adds first/recent window means plus non-finite detection for gate reports.
- Create `D:\Github\HOPE\tools\stage3\stage3_result_parsers.py`
  - Parses evaluation `result.txt` files and resource monitor CSV files.
- Create `D:\Github\HOPE\tools\stage3\compare_stage3_20k.py`
  - Produces JSON/Markdown candidate comparison reports and assigns `pass`, `quality-pass-speed-neutral`, `investigate`, or `reject`.
- Create `D:\Github\HOPE\tools\stage3\parity_stage3_env.py`
  - Runs deterministic original-vs-candidate semantic parity checks across Normal, Complex, Extrem, and DLP.
- Create `D:\Github\HOPE\tools\stage3\profile_stage3_components.py`
  - Measures Stage 3 training components without treating the diagnostic as quality evidence.
- Create `D:\Github\HOPE\tools\stage3\launch_stage3_20k.ps1`
  - Starts a 20K validation run through the original training script, locks the run directory, starts resource monitoring, and writes a manifest.
- Create `D:\Github\HOPE\tools\stage3\stop_stage3_at_20k.ps1`
  - Stops launcher/workload/monitor processes once TensorBoard reaches around 20K and `SAC_19999.pt` exists.
- Create `D:\Github\HOPE\tools\stage3\eval_stage3_checkpoint.ps1`
  - Runs the 200-episode external eval for a candidate checkpoint.
- Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_baseline.py`
- Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_manifest.py`
- Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_result_parsers.py`
- Create `D:\Github\HOPE\tools\stage3\tests\test_compare_stage3_20k.py`
- Create `D:\Github\HOPE\docs\research\stage3_20k_candidate_report_template.md`
  - Human-readable report template that matches the generated Markdown report.
- Modify `D:\Github\HOPE\task_plan.md`
- Modify `D:\Github\HOPE\findings.md`
- Modify `D:\Github\HOPE\progress.md`

## Task 1: Baseline Constants And Gate Helpers

**Files:**
- Create: `D:\Github\HOPE\tools\__init__.py`
- Create: `D:\Github\HOPE\tools\stage3\__init__.py`
- Create: `D:\Github\HOPE\tools\stage3\stage3_baseline.py`
- Test: `D:\Github\HOPE\tools\stage3\tests\test_stage3_baseline.py`

- [ ] **Step 1: Write the failing baseline tests**

Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_baseline.py` with:

```python
import unittest

from tools.stage3.stage3_baseline import (
    BASELINE_20K,
    QUALITY_THRESHOLDS,
    SPEED_MIN_IMPROVEMENT_RATIO,
    quality_gate_status,
    speed_gate_passed,
)


class Stage3BaselineTests(unittest.TestCase):
    def test_baseline_identity_and_thresholds(self):
        self.assertEqual(BASELINE_20K["run_dir"], r"D:\Github\HOPE\src\log\exp\sac_20260619_004316")
        self.assertEqual(BASELINE_20K["checkpoint"], r"D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt")
        self.assertAlmostEqual(BASELINE_20K["external_eval"]["mean"], 0.88625)
        self.assertAlmostEqual(QUALITY_THRESHOLDS["mean"], 0.85625)
        self.assertAlmostEqual(SPEED_MIN_IMPROVEMENT_RATIO, 0.10)

    def test_quality_gate_accepts_baseline_level_eval(self):
        result = quality_gate_status(
            eval_success={"Normal": 0.985, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960},
            tensorboard_has_nonfinite=False,
            multi_scene_collapse=False,
            checkpoint_missing=False,
            parity_failed=False,
        )
        self.assertTrue(result["quality_pass"])
        self.assertEqual(result["hard_reject_reasons"], [])

    def test_quality_gate_rejects_missing_checkpoint(self):
        result = quality_gate_status(
            eval_success={"Normal": 1.0, "Complex": 1.0, "Extrem": 1.0, "DLP": 1.0},
            tensorboard_has_nonfinite=False,
            multi_scene_collapse=False,
            checkpoint_missing=True,
            parity_failed=False,
        )
        self.assertFalse(result["quality_pass"])
        self.assertIn("missing_20k_checkpoint", result["hard_reject_reasons"])

    def test_speed_gate_uses_any_primary_metric(self):
        self.assertTrue(speed_gate_passed({"wall_time_hours": 11.0, "env_steps_per_second": 39.10, "episodes_per_hour": 1545.71})["speed_pass"])
        self.assertTrue(speed_gate_passed({"wall_time_hours": 12.964, "env_steps_per_second": 44.0, "episodes_per_hour": 1545.71})["speed_pass"])
        self.assertFalse(speed_gate_passed({"wall_time_hours": 12.5, "env_steps_per_second": 40.0, "episodes_per_hour": 1600.0})["speed_pass"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing test**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_baseline -v
```

Expected: fails with `ModuleNotFoundError` for `tools.stage3.stage3_baseline`.

- [ ] **Step 3: Add package markers and baseline implementation**

Create `D:\Github\HOPE\tools\__init__.py` with:

```python
"""Local HOPE research tooling package."""
```

Create `D:\Github\HOPE\tools\stage3\__init__.py` with:

```python
"""Stage 3 HOPE experiment tooling."""
```

Create `D:\Github\HOPE\tools\stage3\stage3_baseline.py` with:

```python
from __future__ import annotations

from statistics import mean
from typing import Mapping


SCENES = ("Normal", "Complex", "Extrem", "DLP")

PROTECTED_SOURCE_PATHS = (
    "src/train",
    "src/env",
    "src/model",
)

BASELINE_ID = "stage3_command_only_20k_20260619"

BASELINE_20K = {
    "id": BASELINE_ID,
    "run_dir": r"D:\Github\HOPE\src\log\exp\sac_20260619_004316",
    "checkpoint": r"D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt",
    "stop_episode_snapshot": 20038,
    "env_steps": 1824904,
    "wall_time_hours": 12.964,
    "episodes_per_hour": 1545.71,
    "env_steps_per_second": 39.10,
    "resource_profile": {
        "avg_process_cpu_percent": 37.76,
        "avg_whole_gpu_util_percent": 29.42,
    },
    "tensorboard": {
        "avg_reward": {"latest": 0.076828, "recent_mean": 0.061052},
        "actor_loss": {"latest": -0.471469, "recent_mean": -0.543129},
        "critic_loss": {"latest": 0.080817, "recent_mean": 0.103880},
        "success_rate_Normal": {"latest": 1.0, "recent_mean": 1.0},
        "success_rate_Complex": {"latest": 0.94, "recent_mean": 0.94},
        "success_rate_Extrem": {"latest": 0.69, "recent_mean": 0.7108},
        "success_rate_dlp": {"latest": 0.82, "recent_mean": 0.7924},
        "step_num": {"latest": 52.0, "recent_mean": 77.22},
    },
    "external_eval": {
        "Normal": 0.985,
        "Complex": 0.945,
        "Extrem": 0.655,
        "DLP": 0.960,
        "mean": 0.88625,
    },
}

QUALITY_THRESHOLDS = {
    "Normal": 0.95,
    "Complex": 0.90,
    "Extrem": 0.58,
    "DLP": 0.91,
    "mean": 0.85625,
}

SPEED_MIN_IMPROVEMENT_RATIO = 0.10


def scene_mean(eval_success: Mapping[str, float]) -> float:
    missing = [scene for scene in SCENES if scene not in eval_success]
    if missing:
        raise ValueError(f"missing scene results: {', '.join(missing)}")
    return mean(float(eval_success[scene]) for scene in SCENES)


def quality_gate_status(
    *,
    eval_success: Mapping[str, float],
    tensorboard_has_nonfinite: bool,
    multi_scene_collapse: bool,
    checkpoint_missing: bool,
    parity_failed: bool,
) -> dict:
    hard_reject_reasons: list[str] = []
    if checkpoint_missing:
        hard_reject_reasons.append("missing_20k_checkpoint")
    if tensorboard_has_nonfinite:
        hard_reject_reasons.append("nonfinite_tensorboard_metric")
    if multi_scene_collapse:
        hard_reject_reasons.append("multi_scene_collapse")
    if parity_failed:
        hard_reject_reasons.append("parity_failed")

    per_scene = {
        scene: float(eval_success.get(scene, -1.0)) >= QUALITY_THRESHOLDS[scene]
        for scene in SCENES
    }
    mean_success = scene_mean(eval_success) if all(scene in eval_success for scene in SCENES) else -1.0
    mean_pass = mean_success >= QUALITY_THRESHOLDS["mean"]
    quality_pass = not hard_reject_reasons and all(per_scene.values()) and mean_pass

    return {
        "quality_pass": quality_pass,
        "mean_success": mean_success,
        "mean_pass": mean_pass,
        "per_scene_pass": per_scene,
        "hard_reject_reasons": hard_reject_reasons,
    }


def speed_gate_passed(candidate_metrics: Mapping[str, float]) -> dict:
    baseline_hours = float(BASELINE_20K["wall_time_hours"])
    baseline_eps_hour = float(BASELINE_20K["episodes_per_hour"])
    baseline_env_steps_sec = float(BASELINE_20K["env_steps_per_second"])

    wall_time_hours = float(candidate_metrics.get("wall_time_hours", baseline_hours))
    episodes_per_hour = float(candidate_metrics.get("episodes_per_hour", baseline_eps_hour))
    env_steps_per_second = float(candidate_metrics.get("env_steps_per_second", baseline_env_steps_sec))

    wall_time_improvement = (baseline_hours - wall_time_hours) / baseline_hours
    eps_hour_improvement = (episodes_per_hour - baseline_eps_hour) / baseline_eps_hour
    env_steps_improvement = (env_steps_per_second - baseline_env_steps_sec) / baseline_env_steps_sec

    metrics = {
        "wall_time_improvement_ratio": wall_time_improvement,
        "episodes_per_hour_improvement_ratio": eps_hour_improvement,
        "env_steps_per_second_improvement_ratio": env_steps_improvement,
    }
    speed_pass = any(value >= SPEED_MIN_IMPROVEMENT_RATIO for value in metrics.values())
    return {"speed_pass": speed_pass, **metrics}
```

- [ ] **Step 4: Run baseline tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_baseline -v
git diff -- src/train src/env src/model
```

Expected: all tests pass; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 1**

Run:

```powershell
cd D:\Github\HOPE
git add tools\__init__.py tools\stage3\__init__.py tools\stage3\stage3_baseline.py tools\stage3\tests\test_stage3_baseline.py
git commit -m "feat(stage3): add 20k baseline gates"
```

Expected: commit succeeds when Git author identity is configured. If Git reports missing author identity, stop execution and ask the user to configure repository-local Git identity.

## Task 2: Manifest Utility

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\stage3_manifest.py`
- Test: `D:\Github\HOPE\tools\stage3\tests\test_stage3_manifest.py`

- [ ] **Step 1: Write manifest tests**

Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_manifest.py` with:

```python
import json
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


class Stage3ManifestTests(unittest.TestCase):
    def test_manifest_round_trip_and_update(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "manifest.json"
            manifest = build_manifest(
                repo_root=Path(r"D:\Github\HOPE"),
                run_name="candidate_render_headless",
                candidate_type="20k_validation",
                changed_knobs={"render_mode": "rgb_array"},
                command=["python", "train.py"],
                python_executable=r"D:\Github\HOPE\.venv\Scripts\python.exe",
                env={"SDL_VIDEODRIVER": "dummy"},
                run_dir=r"D:\Github\HOPE\src\log\exp\sac_example",
                stdout_path=r"D:\Github\HOPE\src\log\exp\sac_example.stdout.log",
                stderr_path=r"D:\Github\HOPE\src\log\exp\sac_example.stderr.log",
                resource_csv_path=r"D:\Github\HOPE\src\log\exp\sac_example.resources.csv",
            )
            write_manifest(manifest, path)
            update_manifest(path, {"gate_status": "running"})
            loaded = read_manifest(path)
            self.assertEqual(loaded["run_name"], "candidate_render_headless")
            self.assertEqual(loaded["candidate_type"], "20k_validation")
            self.assertEqual(loaded["changed_knobs"], {"render_mode": "rgb_array"})
            self.assertEqual(loaded["gate_status"], "running")
            json.dumps(loaded)

    def test_protected_source_diff_returns_text(self):
        diff_text = protected_source_diff(Path(r"D:\Github\HOPE"))
        self.assertIsInstance(diff_text, str)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing manifest tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_manifest -v
```

Expected: fails with `ModuleNotFoundError` for `tools.stage3.stage3_manifest`.

- [ ] **Step 3: Implement manifest utilities**

Create `D:\Github\HOPE\tools\stage3\stage3_manifest.py` with:

```python
from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence

from tools.stage3.stage3_baseline import BASELINE_ID, PROTECTED_SOURCE_PATHS


def _run_git(repo_root: Path, args: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *args],
        cwd=repo_root,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    return (completed.stdout or completed.stderr).strip()


def utc_timestamp() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def git_branch(repo_root: Path) -> str:
    return _run_git(repo_root, ["branch", "--show-current"])


def git_status_short(repo_root: Path) -> str:
    return _run_git(repo_root, ["status", "-sb"])


def protected_source_diff(repo_root: Path) -> str:
    return _run_git(repo_root, ["diff", "--", *PROTECTED_SOURCE_PATHS])


def build_manifest(
    *,
    repo_root: Path,
    run_name: str,
    candidate_type: str,
    changed_knobs: Mapping[str, Any],
    command: Sequence[str],
    python_executable: str,
    env: Mapping[str, str],
    run_dir: str,
    stdout_path: str,
    stderr_path: str,
    resource_csv_path: str,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "created_at_utc": utc_timestamp(),
        "updated_at_utc": utc_timestamp(),
        "baseline_id": BASELINE_ID,
        "run_name": run_name,
        "candidate_type": candidate_type,
        "changed_knobs": dict(changed_knobs),
        "git_branch": git_branch(repo_root),
        "git_status_sb": git_status_short(repo_root),
        "protected_source_diff": protected_source_diff(repo_root),
        "command": list(command),
        "python_executable": python_executable,
        "environment": dict(env),
        "run_dir": run_dir,
        "stdout_path": stdout_path,
        "stderr_path": stderr_path,
        "resource_csv_path": resource_csv_path,
        "tensorboard_summary_json": "",
        "tensorboard_summary_markdown": "",
        "expected_checkpoint_20k": str(Path(run_dir) / "SAC_19999.pt"),
        "eval_result_dirs": {},
        "launcher_pid": None,
        "workload_pid": None,
        "resource_monitor_pid": None,
        "gate_status": "created",
        "notes": [],
    }


def write_manifest(manifest: Mapping[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def read_manifest(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def update_manifest(path: Path, updates: Mapping[str, Any]) -> dict[str, Any]:
    manifest = read_manifest(path)
    manifest.update(updates)
    manifest["updated_at_utc"] = utc_timestamp()
    write_manifest(manifest, path)
    return manifest
```

- [ ] **Step 4: Run manifest tests and protected diff check**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_manifest -v
git diff -- src/train src/env src/model
```

Expected: all tests pass; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 2**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\stage3_manifest.py tools\stage3\tests\test_stage3_manifest.py
git commit -m "feat(stage3): add experiment manifest utility"
```

Expected: commit succeeds when Git author identity is configured.

## Task 3: TensorBoard Summary Gate Fields

**Files:**
- Modify: `D:\Github\HOPE\tools\stage3\tensorboard_stage3_summary.py`
- Test: `D:\Github\HOPE\tools\stage3\tests\test_tensorboard_stage3_summary_windows.py`

- [ ] **Step 1: Write TensorBoard window-summary test**

Create `D:\Github\HOPE\tools\stage3\tests\test_tensorboard_stage3_summary_windows.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from torch.utils.tensorboard import SummaryWriter

from tools.stage3.tensorboard_stage3_summary import summarize_event_dir


class TensorboardStage3SummaryWindowTests(unittest.TestCase):
    def test_summary_includes_window_means_and_nonfinite_flag(self):
        with tempfile.TemporaryDirectory() as tmp:
            run_dir = Path(tmp)
            writer = SummaryWriter(str(run_dir))
            for step in range(600):
                writer.add_scalar("avg_reward", float(step), step)
                writer.add_scalar("critic_loss", float(step) / 10.0, step)
            writer.flush()
            writer.close()

            report = summarize_event_dir(run_dir, min_episodes=500)
            reward = report["scalars"]["avg_reward"]
            self.assertEqual(reward["count"], 600)
            self.assertAlmostEqual(reward["mean_first500"], 249.5)
            self.assertAlmostEqual(reward["mean_last100"], 549.5)
            self.assertAlmostEqual(reward["mean_last500"], 349.5)
            self.assertFalse(reward["has_nonfinite"])
            self.assertFalse(report["has_nonfinite"])


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing TensorBoard test**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_tensorboard_stage3_summary_windows -v
```

Expected: fails because `mean_first500`, `mean_last100`, `mean_last500`, or report-level `has_nonfinite` is absent.

- [ ] **Step 3: Extend scalar summaries without breaking existing keys**

In `D:\Github\HOPE\tools\stage3\tensorboard_stage3_summary.py`, ensure the imports include:

```python
import math
import statistics
```

Then update the scalar aggregation function so every scalar dictionary includes these keys while preserving all current fields:

```python
finite_values = [value for value in values if math.isfinite(value)]
has_nonfinite = len(finite_values) != len(values)
summary.update(
    {
        "has_nonfinite": has_nonfinite,
        "mean_first500": statistics.fmean(finite_values[:500]) if finite_values[:500] else None,
        "mean_last100": statistics.fmean(finite_values[-100:]) if finite_values[-100:] else None,
        "mean_last500": statistics.fmean(finite_values[-500:]) if finite_values[-500:] else None,
    }
)
```

In the report construction, add:

```python
report["has_nonfinite"] = any(
    scalar.get("has_nonfinite", False)
    for scalar in report["scalars"].values()
)
```

Keep the existing `WATCHED_SCALARS`, CLI arguments, JSON output fields, Markdown output, and `training_budget_met` behavior intact.

- [ ] **Step 4: Run TensorBoard summary tests and a live baseline summary**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_tensorboard_stage3_summary_windows -v
.\.venv\Scripts\python.exe tools\stage3\tensorboard_stage3_summary.py src\log\exp\sac_20260619_004316 --min-episodes 20000 --json docs\research\stage3_command_only_20k_summary_check.json --markdown docs\research\stage3_command_only_20k_summary_check.md
git diff -- src/train src/env src/model
```

Expected: unittest passes; baseline summary writes JSON and Markdown; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 3**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\tensorboard_stage3_summary.py tools\stage3\tests\test_tensorboard_stage3_summary_windows.py docs\research\stage3_command_only_20k_summary_check.json docs\research\stage3_command_only_20k_summary_check.md
git commit -m "feat(stage3): add tensorboard gate windows"
```

Expected: commit succeeds when Git author identity is configured.

## Task 4: Evaluation And Resource Parsers

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\stage3_result_parsers.py`
- Test: `D:\Github\HOPE\tools\stage3\tests\test_stage3_result_parsers.py`

- [ ] **Step 1: Write parser tests**

Create `D:\Github\HOPE\tools\stage3\tests\test_stage3_result_parsers.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from tools.stage3.stage3_result_parsers import (
    collect_eval_success,
    parse_eval_result_file,
    summarize_resource_csv,
)


class Stage3ResultParserTests(unittest.TestCase):
    def test_parse_eval_result_file(self):
        with tempfile.TemporaryDirectory() as tmp:
            result_path = Path(tmp) / "result.txt"
            result_path.write_text("success rate: 0.945\nstep num: 88.1 +-(12.3)\n", encoding="utf-8")
            parsed = parse_eval_result_file(result_path)
            self.assertEqual(parsed["success_rate"], 0.945)
            self.assertEqual(parsed["step_num_mean"], 88.1)

    def test_collect_eval_success_from_scene_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for scene_dir, value in {"normalize": 0.985, "complex": 0.945, "extreme": 0.655, "dlp": 0.960}.items():
                path = root / scene_dir
                path.mkdir()
                (path / "result.txt").write_text(f"success rate: {value}\nstep num: 90.0 +-(1.0)\n", encoding="utf-8")
            success = collect_eval_success(root)
            self.assertEqual(success, {"Normal": 0.985, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960})

    def test_summarize_resource_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "resources.csv"
            path.write_text(
                "timestamp,pid,process_name,cpu_percent,working_set_mb,private_memory_mb,gpu_util_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_power_w\n"
                "2026-06-19T00:00:00,1,python,10,100,80,20,1000,12000,50\n"
                "2026-06-19T00:00:05,1,python,30,120,90,40,1500,12000,60\n",
                encoding="utf-8",
            )
            summary = summarize_resource_csv(path)
            self.assertEqual(summary["sample_count"], 2)
            self.assertEqual(summary["avg_process_cpu_percent"], 20.0)
            self.assertEqual(summary["avg_whole_gpu_util_percent"], 30.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing parser tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_result_parsers -v
```

Expected: fails with `ModuleNotFoundError` for `tools.stage3.stage3_result_parsers`.

- [ ] **Step 3: Implement parsers**

Create `D:\Github\HOPE\tools\stage3\stage3_result_parsers.py` with:

```python
from __future__ import annotations

import csv
import re
from pathlib import Path


SCENE_DIRS = {
    "Normal": "normalize",
    "Complex": "complex",
    "Extrem": "extreme",
    "DLP": "dlp",
}


def _float_or_none(value: str) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def parse_eval_result_file(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8", errors="replace")
    success_match = re.search(r"success rate:\s*([0-9.]+)", text)
    step_match = re.search(r"step num:\s*([0-9.]+)", text)
    if not success_match:
        raise ValueError(f"missing success rate in {path}")
    return {
        "success_rate": float(success_match.group(1)),
        "step_num_mean": float(step_match.group(1)) if step_match else float("nan"),
    }


def collect_eval_success(eval_root: Path) -> dict[str, float]:
    success: dict[str, float] = {}
    for scene, directory in SCENE_DIRS.items():
        result_path = eval_root / directory / "result.txt"
        success[scene] = parse_eval_result_file(result_path)["success_rate"]
    return success


def summarize_resource_csv(path: Path) -> dict[str, float | int | str]:
    rows = list(csv.DictReader(path.read_text(encoding="utf-8", errors="replace").splitlines()))
    cpu_values = [_float_or_none(row.get("cpu_percent", "")) for row in rows]
    gpu_values = [_float_or_none(row.get("gpu_util_percent", "")) for row in rows]
    power_values = [_float_or_none(row.get("gpu_power_w", "")) for row in rows]
    memory_values = [_float_or_none(row.get("gpu_memory_used_mb", "")) for row in rows]

    def average(values: list[float | None]) -> float:
        finite = [value for value in values if value is not None]
        return sum(finite) / len(finite) if finite else 0.0

    return {
        "path": str(path),
        "sample_count": len(rows),
        "avg_process_cpu_percent": average(cpu_values),
        "avg_whole_gpu_util_percent": average(gpu_values),
        "avg_gpu_power_w": average(power_values),
        "avg_gpu_memory_used_mb": average(memory_values),
    }
```

- [ ] **Step 4: Run parser tests and protected diff check**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_stage3_result_parsers -v
git diff -- src/train src/env src/model
```

Expected: all tests pass; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 4**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\stage3_result_parsers.py tools\stage3\tests\test_stage3_result_parsers.py
git commit -m "feat(stage3): parse eval and resource outputs"
```

Expected: commit succeeds when Git author identity is configured.

## Task 5: 20K Comparison And Gate Report

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\compare_stage3_20k.py`
- Test: `D:\Github\HOPE\tools\stage3\tests\test_compare_stage3_20k.py`

- [ ] **Step 1: Write comparison tests**

Create `D:\Github\HOPE\tools\stage3\tests\test_compare_stage3_20k.py` with:

```python
import tempfile
import unittest
from pathlib import Path

from tools.stage3.compare_stage3_20k import build_comparison_report, classify_decision


class Stage3Compare20KTests(unittest.TestCase):
    def test_classify_pass(self):
        decision = classify_decision(
            quality={"quality_pass": True, "hard_reject_reasons": []},
            speed={"speed_pass": True},
        )
        self.assertEqual(decision, "pass")

    def test_classify_quality_pass_speed_neutral(self):
        decision = classify_decision(
            quality={"quality_pass": True, "hard_reject_reasons": []},
            speed={"speed_pass": False},
        )
        self.assertEqual(decision, "quality-pass-speed-neutral")

    def test_build_report_uses_baseline_and_candidate_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            resource_csv = Path(tmp) / "resource.csv"
            resource_csv.write_text(
                "timestamp,pid,process_name,cpu_percent,working_set_mb,private_memory_mb,gpu_util_percent,gpu_memory_used_mb,gpu_memory_total_mb,gpu_power_w\n"
                "2026-06-19T00:00:00,1,python,20,100,80,30,1000,12000,50\n",
                encoding="utf-8",
            )
            report = build_comparison_report(
                manifest={
                    "run_name": "candidate_test",
                    "candidate_type": "20k_validation",
                    "changed_knobs": {"batch_size": 512},
                    "run_dir": r"D:\Github\HOPE\src\log\exp\sac_candidate",
                    "command": ["python", "train_HOPE_sac.py"],
                    "protected_source_diff": "",
                },
                tensorboard_summary={
                    "episode_count": 20000,
                    "training_budget_met": True,
                    "has_nonfinite": False,
                    "scalars": {
                        "success_rate_Normal": {"latest": 1.0, "mean_last100": 1.0},
                        "success_rate_Complex": {"latest": 0.94, "mean_last100": 0.94},
                        "success_rate_Extrem": {"latest": 0.70, "mean_last100": 0.70},
                        "success_rate_dlp": {"latest": 0.82, "mean_last100": 0.82},
                    },
                },
                resource_csv_path=resource_csv,
                eval_success={"Normal": 0.985, "Complex": 0.945, "Extrem": 0.655, "DLP": 0.960},
                candidate_speed={"wall_time_hours": 11.0, "env_steps_per_second": 44.0, "episodes_per_hour": 1750.0},
                parity_status="pass",
                checkpoint_missing=False,
            )
            self.assertEqual(report["decision"], "pass")
            self.assertAlmostEqual(report["quality"]["mean_success"], 0.88625)
            self.assertEqual(report["resource_summary"]["sample_count"], 1)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the failing comparison tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_compare_stage3_20k -v
```

Expected: fails with `ModuleNotFoundError` for `tools.stage3.compare_stage3_20k`.

- [ ] **Step 3: Implement comparison report generator**

Create `D:\Github\HOPE\tools\stage3\compare_stage3_20k.py` with these public functions and CLI behavior:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping

from tools.stage3.stage3_baseline import BASELINE_20K, quality_gate_status, scene_mean, speed_gate_passed
from tools.stage3.stage3_manifest import read_manifest, update_manifest
from tools.stage3.stage3_result_parsers import collect_eval_success, summarize_resource_csv


def classify_decision(*, quality: Mapping[str, Any], speed: Mapping[str, Any]) -> str:
    if quality.get("hard_reject_reasons"):
        return "reject"
    if not quality.get("quality_pass", False):
        return "investigate"
    if speed.get("speed_pass", False):
        return "pass"
    return "quality-pass-speed-neutral"


def _multi_scene_collapse(tensorboard_summary: Mapping[str, Any]) -> bool:
    scalars = tensorboard_summary.get("scalars", {})
    scene_tags = ("success_rate_Normal", "success_rate_Complex", "success_rate_Extrem", "success_rate_dlp")
    low_count = 0
    for tag in scene_tags:
        scalar = scalars.get(tag, {})
        value = scalar.get("mean_last100", scalar.get("recent_mean", scalar.get("latest")))
        if value is not None and float(value) < 0.5:
            low_count += 1
    return low_count >= 2


def build_comparison_report(
    *,
    manifest: Mapping[str, Any],
    tensorboard_summary: Mapping[str, Any],
    resource_csv_path: Path,
    eval_success: Mapping[str, float],
    candidate_speed: Mapping[str, float],
    parity_status: str,
    checkpoint_missing: bool,
) -> dict[str, Any]:
    quality = quality_gate_status(
        eval_success=eval_success,
        tensorboard_has_nonfinite=bool(tensorboard_summary.get("has_nonfinite", False)),
        multi_scene_collapse=_multi_scene_collapse(tensorboard_summary),
        checkpoint_missing=checkpoint_missing,
        parity_failed=parity_status == "fail",
    )
    speed = speed_gate_passed(candidate_speed)
    decision = classify_decision(quality=quality, speed=speed)
    return {
        "baseline": BASELINE_20K,
        "candidate": {
            "run_name": manifest.get("run_name", ""),
            "candidate_type": manifest.get("candidate_type", ""),
            "changed_knobs": manifest.get("changed_knobs", {}),
            "run_dir": manifest.get("run_dir", ""),
            "command": manifest.get("command", []),
            "protected_source_diff_present": bool(str(manifest.get("protected_source_diff", "")).strip()),
            "parity_status": parity_status,
            "eval_success": dict(eval_success),
            "eval_mean": scene_mean(eval_success),
            "speed": dict(candidate_speed),
        },
        "tensorboard_summary": tensorboard_summary,
        "resource_summary": summarize_resource_csv(resource_csv_path),
        "quality": quality,
        "speed": speed,
        "decision": decision,
    }


def write_markdown(report: Mapping[str, Any], path: Path) -> None:
    candidate = report["candidate"]
    speed = candidate["speed"]
    quality = report["quality"]
    lines = [
        f"# Stage 3 20K Candidate Report: {candidate['run_name']}",
        "",
        f"- Decision: `{report['decision']}`",
        f"- Baseline: `{report['baseline']['id']}`",
        f"- Candidate type: `{candidate['candidate_type']}`",
        f"- Run dir: `{candidate['run_dir']}`",
        f"- Changed knobs: `{json.dumps(candidate['changed_knobs'], ensure_ascii=False)}`",
        f"- Parity status: `{candidate['parity_status']}`",
        "",
        "## Speed",
        "",
        f"- Wall time hours: `{speed.get('wall_time_hours')}`",
        f"- Episodes/hour: `{speed.get('episodes_per_hour')}`",
        f"- Env steps/sec: `{speed.get('env_steps_per_second')}`",
        f"- Speed gate pass: `{report['speed']['speed_pass']}`",
        "",
        "## Quality",
        "",
        f"- Mean eval success: `{quality['mean_success']}`",
        f"- Quality pass: `{quality['quality_pass']}`",
        f"- Hard reject reasons: `{quality['hard_reject_reasons']}`",
        "",
        "## Scene Eval",
        "",
    ]
    for scene, value in candidate["eval_success"].items():
        lines.append(f"- {scene}: `{value}`")
    lines.extend(["", "## Command", "", "```powershell", " ".join(candidate["command"]), "```", ""])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare a Stage 3 20K candidate against the locked baseline.")
    parser.add_argument("--manifest", required=True, type=Path)
    parser.add_argument("--tensorboard-json", required=True, type=Path)
    parser.add_argument("--resource-csv", required=True, type=Path)
    parser.add_argument("--eval-root", required=True, type=Path)
    parser.add_argument("--wall-time-hours", required=True, type=float)
    parser.add_argument("--episodes-per-hour", required=True, type=float)
    parser.add_argument("--env-steps-per-second", required=True, type=float)
    parser.add_argument("--parity-status", choices=("not-required", "pass", "fail"), default="not-required")
    parser.add_argument("--checkpoint-missing", action="store_true")
    parser.add_argument("--json", required=True, type=Path)
    parser.add_argument("--markdown", required=True, type=Path)
    args = parser.parse_args()

    manifest = read_manifest(args.manifest)
    tensorboard_summary = json.loads(args.tensorboard_json.read_text(encoding="utf-8"))
    eval_success = collect_eval_success(args.eval_root)
    report = build_comparison_report(
        manifest=manifest,
        tensorboard_summary=tensorboard_summary,
        resource_csv_path=args.resource_csv,
        eval_success=eval_success,
        candidate_speed={
            "wall_time_hours": args.wall_time_hours,
            "episodes_per_hour": args.episodes_per_hour,
            "env_steps_per_second": args.env_steps_per_second,
        },
        parity_status=args.parity_status,
        checkpoint_missing=args.checkpoint_missing,
    )
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    update_manifest(args.manifest, {"gate_status": report["decision"], "comparison_report_json": str(args.json), "comparison_report_markdown": str(args.markdown)})
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run comparison tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_compare_stage3_20k -v
git diff -- src/train src/env src/model
```

Expected: all tests pass; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 5**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\compare_stage3_20k.py tools\stage3\tests\test_compare_stage3_20k.py
git commit -m "feat(stage3): compare 20k candidates"
```

Expected: commit succeeds when Git author identity is configured.

## Task 6: Semantic Parity Harness

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\parity_stage3_env.py`
- Test through command smoke: no training artifacts committed.

- [ ] **Step 1: Create the parity harness**

Create `D:\Github\HOPE\tools\stage3\parity_stage3_env.py` with a CLI that accepts:

```powershell
.\.venv\Scripts\python.exe tools\stage3\parity_stage3_env.py --episodes-per-scene 2 --max-steps 20 --json docs\research\stage3_parity_check.json --markdown docs\research\stage3_parity_check.md
```

The script must:

```python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
sys.path.insert(0, str(SRC_ROOT))
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper


SCENES = ("Normal", "Complex", "Extrem", "dlp")
SCRIPTED_ACTIONS = (
    np.array([0.0, 0.4], dtype=np.float32),
    np.array([0.25, 0.3], dtype=np.float32),
    np.array([-0.25, 0.3], dtype=np.float32),
    np.array([0.5, -0.2], dtype=np.float32),
    np.array([-0.5, -0.2], dtype=np.float32),
)


def _make_env(render_mode: str | None) -> CarParkingWrapper:
    kwargs = {"fps": 100, "verbose": False}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    return CarParkingWrapper(CarParking(**kwargs))


def _reset(env: CarParkingWrapper, scene: str, case_index: int):
    if scene == "dlp":
        return env.reset(case_index, False, scene)
    return env.reset(None, None, scene)


def _as_array(value):
    return np.asarray(value)


def _compare_obs(left, right) -> list[str]:
    failures: list[str] = []
    for key in ("img", "lidar", "target", "action_mask"):
        if key not in left or key not in right:
            failures.append(f"missing_obs_{key}")
            continue
        l_value = _as_array(left[key])
        r_value = _as_array(right[key])
        if key == "action_mask":
            if not np.array_equal(l_value, r_value):
                failures.append("action_mask_mismatch")
        else:
            if not np.allclose(l_value, r_value, rtol=1e-6, atol=1e-6):
                failures.append(f"{key}_mismatch")
    return failures


def run_parity(episodes_per_scene: int, max_steps: int) -> dict:
    left_env = _make_env(None)
    right_env = _make_env("rgb_array")
    results = []
    try:
        for scene in SCENES:
            for case_index in range(episodes_per_scene):
                left_obs = _reset(left_env, scene, case_index)
                right_obs = _reset(right_env, scene, case_index)
                failures = _compare_obs(left_obs, right_obs)
                terminal_step = None
                for step in range(max_steps):
                    action = SCRIPTED_ACTIONS[step % len(SCRIPTED_ACTIONS)]
                    left_obs, left_reward, left_done, left_info = left_env.step(action)
                    right_obs, right_reward, right_done, right_info = right_env.step(action)
                    failures.extend(_compare_obs(left_obs, right_obs))
                    if abs(float(left_reward) - float(right_reward)) > 1e-9:
                        failures.append("reward_mismatch")
                    if bool(left_done) != bool(right_done):
                        failures.append("done_mismatch")
                    if left_info.get("status") != right_info.get("status"):
                        failures.append("status_mismatch")
                    if bool(left_info.get("path_to_dest")) != bool(right_info.get("path_to_dest")):
                        failures.append("path_to_dest_presence_mismatch")
                    if left_done or right_done:
                        terminal_step = step + 1
                        break
                results.append(
                    {
                        "scene": scene,
                        "case_index": case_index,
                        "terminal_step": terminal_step,
                        "failures": sorted(set(failures)),
                    }
                )
    finally:
        left_env.close()
        right_env.close()
    failed = [item for item in results if item["failures"]]
    return {"status": "fail" if failed else "pass", "results": results}


def write_markdown(report: dict, path: Path) -> None:
    lines = ["# Stage 3 Parity Check", "", f"- Status: `{report['status']}`", ""]
    for item in report["results"]:
        lines.append(f"- {item['scene']} case {item['case_index']}: `{item['failures']}`")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Check HOPE Stage 3 environment semantic parity.")
    parser.add_argument("--episodes-per-scene", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=20)
    parser.add_argument("--json", type=Path, required=True)
    parser.add_argument("--markdown", type=Path, required=True)
    args = parser.parse_args()
    report = run_parity(args.episodes_per_scene, args.max_steps)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_markdown(report, args.markdown)
    return 1 if report["status"] == "fail" else 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run a quick parity smoke**

Run:

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe tools\stage3\parity_stage3_env.py --episodes-per-scene 1 --max-steps 5 --json docs\research\stage3_parity_smoke.json --markdown docs\research\stage3_parity_smoke.md
git diff -- src/train src/env src/model
```

Expected: script exits `0` and writes JSON/Markdown if render-mode parity holds; if it exits `1`, inspect `docs\research\stage3_parity_smoke.md` before approving environment micro-optimization candidates. Protected-source diff command prints no output.

- [ ] **Step 3: Commit Task 6**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\parity_stage3_env.py docs\research\stage3_parity_smoke.json docs\research\stage3_parity_smoke.md
git commit -m "feat(stage3): add environment parity harness"
```

Expected: commit succeeds when Git author identity is configured.

## Task 7: Component Profiling Diagnostic

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\profile_stage3_components.py`

- [ ] **Step 1: Create profiling diagnostic**

Create `D:\Github\HOPE\tools\stage3\profile_stage3_components.py`. The script must:

- set `SDL_VIDEODRIVER=dummy`;
- import original HOPE classes from `src`;
- run a bounded diagnostic with `--episodes`, `--max-steps`, `--warmup-steps`, and `--updates`;
- time `env.reset`, `env.step`, `render`, image observation, lidar observation, action mask, RS path probing, `ParkingAgent.get_action`, replay push, replay sample, SAC update, TensorBoard logging, and checkpoint save where the corresponding callable is visible from the original objects;
- write JSON and Markdown reports;
- include a top-level note: `measurement_only_not_quality_evidence`.

Use this implementation pattern:

```python
class TimerTable:
    def __init__(self):
        self.rows = {}

    def record(self, name, seconds):
        row = self.rows.setdefault(name, {"seconds": 0.0, "calls": 0})
        row["seconds"] += seconds
        row["calls"] += 1

    def summary(self):
        total = sum(row["seconds"] for row in self.rows.values())
        return {
            name: {
                "seconds": row["seconds"],
                "calls": row["calls"],
                "avg_ms": (row["seconds"] / row["calls"]) * 1000.0 if row["calls"] else 0.0,
                "percent_measured_wall": (row["seconds"] / total) * 100.0 if total else 0.0,
            }
            for name, row in sorted(self.rows.items())
        }
```

For monkeypatch timing, wrap only methods on local objects created by the diagnostic process:

```python
def timed_method(table, obj, attr_name, label):
    original = getattr(obj, attr_name)

    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        try:
            return original(*args, **kwargs)
        finally:
            table.record(label, time.perf_counter() - start)

    setattr(obj, attr_name, wrapper)
```

- [ ] **Step 2: Run profiling smoke**

Run:

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
.\.venv\Scripts\python.exe tools\stage3\profile_stage3_components.py --episodes 3 --max-steps 50 --updates 5 --json docs\research\stage3_profile_smoke.json --markdown docs\research\stage3_profile_smoke.md
git diff -- src/train src/env src/model
```

Expected: JSON and Markdown reports are written; report marks the run as measurement-only; protected-source diff command prints no output.

- [ ] **Step 3: Commit Task 7**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\profile_stage3_components.py docs\research\stage3_profile_smoke.json docs\research\stage3_profile_smoke.md
git commit -m "feat(stage3): add component profiling diagnostic"
```

Expected: commit succeeds when Git author identity is configured.

## Task 8: 20K Launch, Stop, And Eval Wrappers

**Files:**
- Create: `D:\Github\HOPE\tools\stage3\launch_stage3_20k.ps1`
- Create: `D:\Github\HOPE\tools\stage3\stop_stage3_at_20k.ps1`
- Create: `D:\Github\HOPE\tools\stage3\eval_stage3_checkpoint.ps1`

- [ ] **Step 1: Create launch wrapper**

Create `D:\Github\HOPE\tools\stage3\launch_stage3_20k.ps1` with parameters:

```powershell
param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$CandidateType,
    [Parameter(Mandatory=$true)][string]$ChangedKnobsJson,
    [int]$TrainEpisode = 100000,
    [int]$EvalEpisode = 200
)
```

The script must:

- resolve `$RepoRoot = Resolve-Path "$PSScriptRoot\..\.."`;
- use `$Python = Join-Path $RepoRoot ".venv\Scripts\python.exe"`;
- set `$env:SDL_VIDEODRIVER = "dummy"` and `$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"`;
- start the original command from `D:\Github\HOPE\src`:

```powershell
& $Python .\train\train_HOPE_sac.py --train_episode $TrainEpisode --eval_episode $EvalEpisode --visualize= --verbose=
```

- redirect stdout/stderr to files under `D:\Github\HOPE\src\log\exp`;
- detect the newest `sac_*` directory created after launcher start time and lock it as `run_dir`;
- start `D:\Github\HOPE\tools\stage3\monitor_stage3_resources.ps1` against the Python workload PID;
- write a manifest using `tools.stage3.stage3_manifest.build_manifest`;
- record launcher PID, workload PID, monitor PID, stdout path, stderr path, resource CSV path, and expected `SAC_19999.pt`.

- [ ] **Step 2: Create stop wrapper**

Create `D:\Github\HOPE\tools\stage3\stop_stage3_at_20k.ps1` with parameters:

```powershell
param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [int]$MinEpisodes = 20000
)
```

The script must:

- read the manifest JSON;
- run `tools\stage3\tensorboard_stage3_summary.py` against the locked `run_dir` with `--min-episodes $MinEpisodes`;
- require the expected `SAC_19999.pt` file before stopping;
- stop workload, launcher, and monitor PIDs if they are still active;
- preserve logs/checkpoints;
- update manifest fields `stopped_at_utc`, `tensorboard_summary_json`, `tensorboard_summary_markdown`, and `gate_status = "stopped_at_20k"`.

- [ ] **Step 3: Create eval wrapper**

Create `D:\Github\HOPE\tools\stage3\eval_stage3_checkpoint.ps1` with parameters:

```powershell
param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$CheckpointPath,
    [int]$EvalEpisode = 200
)
```

The script must:

- run from `D:\Github\HOPE\src`;
- use `.venv\Scripts\python.exe`;
- call:

```powershell
& $Python .\evaluation\eval_mix_scene.py $CheckpointPath --eval_episode $EvalEpisode --visualize= --verbose=
```

- discover the evaluation output directory created by the command;
- update manifest `eval_result_dirs.candidate_20k` with that directory path.

- [ ] **Step 4: Run syntax checks without launching training**

Run:

```powershell
cd D:\Github\HOPE
$null = [System.Management.Automation.Language.Parser]::ParseFile('D:\Github\HOPE\tools\stage3\launch_stage3_20k.ps1', [ref]$null, [ref]$null)
$null = [System.Management.Automation.Language.Parser]::ParseFile('D:\Github\HOPE\tools\stage3\stop_stage3_at_20k.ps1', [ref]$null, [ref]$null)
$null = [System.Management.Automation.Language.Parser]::ParseFile('D:\Github\HOPE\tools\stage3\eval_stage3_checkpoint.ps1', [ref]$null, [ref]$null)
git diff -- src/train src/env src/model
```

Expected: parser calls return without errors; protected-source diff command prints no output.

- [ ] **Step 5: Commit Task 8**

Run:

```powershell
cd D:\Github\HOPE
git add tools\stage3\launch_stage3_20k.ps1 tools\stage3\stop_stage3_at_20k.ps1 tools\stage3\eval_stage3_checkpoint.ps1
git commit -m "feat(stage3): add 20k run wrappers"
```

Expected: commit succeeds when Git author identity is configured.

## Task 9: Report Template And Planning Records

**Files:**
- Create: `D:\Github\HOPE\docs\research\stage3_20k_candidate_report_template.md`
- Modify: `D:\Github\HOPE\task_plan.md`
- Modify: `D:\Github\HOPE\findings.md`
- Modify: `D:\Github\HOPE\progress.md`

- [ ] **Step 1: Create candidate report template**

Create `D:\Github\HOPE\docs\research\stage3_20k_candidate_report_template.md` with:

```markdown
# Stage 3 20K Candidate Report

Date:
Run name:

## Decision

- Decision label:
- Baseline id: `stage3_command_only_20k_20260619`
- Candidate type:
- Changed knobs:
- Eligible for 36.5K or 100K follow-up:

## Protected Boundaries

- `git diff -- src/train src/env src/model` output:
- Action mask semantics preserved:
- Curriculum and DLP scheduling preserved:
- Reward/terminal/observation semantics preserved:

## Speed

| Metric | Baseline 20K | Candidate 20K | Delta |
|---|---:|---:|---:|
| Wall time hours | `12.964` |  |  |
| Episodes/hour | `1545.71` |  |  |
| Env steps/sec | `39.10` |  |  |

## Quality

| Scene | Baseline Eval | Candidate Eval | Gate |
|---|---:|---:|---|
| Normal | `0.985` |  | `>=0.95` |
| Complex | `0.945` |  | `>=0.90` |
| Extrem | `0.655` |  | `>=0.58` |
| DLP | `0.960` |  | `>=0.91` |
| Mean | `0.88625` |  | `>=0.85625` |

## TensorBoard Signals

- Non-finite losses:
- Multi-scene collapse:
- Timeout/reward/success interaction:

## Resource Profile

- Process CPU:
- Whole-GPU utilization:
- GPU memory:
- Notes:

## Conclusion

State whether the candidate is accepted, neutral, requires investigation, or rejected.
```

- [ ] **Step 2: Update planning files**

Append concise entries:

`D:\Github\HOPE\task_plan.md`

```markdown
## Stage 3 Safe-Speed 20K Framework Plan

- [ ] Implement opt-in Stage 3 20K experiment framework from `docs/superpowers/plans/2026-06-19-hope-stage3-safe-speed-20k-framework.md`.
- [ ] Preserve `src/train`, `src/env`, and `src/model` while building measurement, parity, launch, stop, eval, and comparison tooling.
- [ ] Use the command-only 20K run `src/log/exp/sac_20260619_004316` as the comparison baseline for future 20K candidates.
```

`D:\Github\HOPE\findings.md`

```markdown
## 2026-06-19 Stage 3 Safe-Speed Framework Planning

- Locked future speed experiments to the command-only 20K baseline: 12.964 h, 1545.71 episodes/hour, 39.10 env steps/sec, 20K external eval mean 0.88625.
- Confirmed hard red lines for implementation: action mask, curriculum and DLP scheduling, RS switching, observation/reward/terminal semantics, and original HOPE training path must be preserved.
- The implementation plan keeps new framework logic under `tools/stage3/` and reports under `docs/research/`.
```

`D:\Github\HOPE\progress.md`

```markdown
## 2026-06-19

- Created executable implementation plan for the Stage 3 safe-speed 20K experiment framework: `docs/superpowers/plans/2026-06-19-hope-stage3-safe-speed-20k-framework.md`.
- Next step is execution via `superpowers:subagent-driven-development` or `superpowers:executing-plans`.
```

- [ ] **Step 3: Commit Task 9**

Run:

```powershell
cd D:\Github\HOPE
git add docs\research\stage3_20k_candidate_report_template.md task_plan.md findings.md progress.md
git commit -m "docs(stage3): document 20k safe-speed workflow"
```

Expected: commit succeeds when Git author identity is configured.

## Task 10: Full Verification

**Files:**
- No new files.

- [ ] **Step 1: Run all Stage 3 tool tests**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -p "test_*.py" -v
```

Expected: all tests pass.

- [ ] **Step 2: Run baseline TensorBoard summary check**

Run:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe tools\stage3\tensorboard_stage3_summary.py src\log\exp\sac_20260619_004316 --min-episodes 20000 --json docs\research\stage3_command_only_20k_final_summary.json --markdown docs\research\stage3_command_only_20k_final_summary.md
```

Expected: command exits `0`, `training_budget_met` is `true`, and the output includes finite scalar summaries.

- [ ] **Step 3: Run parser and comparison dry check using known baseline artifacts**

Use existing baseline artifacts if available. If no external eval directory is present, skip this command and record `baseline_eval_directory_missing` in `progress.md`.

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe tools\stage3\compare_stage3_20k.py --manifest src\log\exp\stage3_command_only_100k_20260619_004315.meta.json --tensorboard-json docs\research\stage3_command_only_20k_final_summary.json --resource-csv src\log\exp\stage3_command_only_100k_20260619_004315.resources.csv --eval-root docs\research\stage3_eval_command_only_20k --wall-time-hours 12.964 --episodes-per-hour 1545.71 --env-steps-per-second 39.10 --parity-status not-required --json docs\research\stage3_command_only_20k_compare_check.json --markdown docs\research\stage3_command_only_20k_compare_check.md
```

Expected: command exits `0` when the eval root exists and produces a `quality-pass-speed-neutral` or `pass` report for the baseline-equivalent data.

- [ ] **Step 4: Confirm protected source remains untouched**

Run:

```powershell
cd D:\Github\HOPE
git diff -- src/train src/env src/model
git status -sb
```

Expected: protected-source diff command prints no output. `git status -sb` shows only intentional tool/docs changes from this plan.

- [ ] **Step 5: Commit final verification artifacts**

Run:

```powershell
cd D:\Github\HOPE
git add docs\research\stage3_command_only_20k_final_summary.json docs\research\stage3_command_only_20k_final_summary.md docs\research\stage3_command_only_20k_compare_check.json docs\research\stage3_command_only_20k_compare_check.md progress.md
git commit -m "test(stage3): verify 20k safe-speed framework"
```

Expected: commit succeeds when Git author identity is configured. If comparison dry check was skipped due a missing eval root, stage only the final TensorBoard summary and the progress note explaining the skipped comparison.

## Completion Criteria

- `tools/stage3/` contains importable, tested tools for baseline constants, manifests, TensorBoard summaries, eval/resource parsing, 20K comparison, parity checking, profiling, and PowerShell launch/stop/eval wrappers.
- A future candidate can be measured, parity-checked, run to 20K, stopped, evaluated, and classified against the locked baseline.
- Generated reports clearly state when a run is measurement-only and must not be used as training-quality evidence.
- `git diff -- src/train src/env src/model` remains empty.
- Root planning files point future agents to this implementation plan.
