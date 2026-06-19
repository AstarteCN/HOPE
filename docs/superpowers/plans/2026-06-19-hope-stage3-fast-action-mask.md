# HOPE Stage 3 Fast Action Mask Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a default-off, opt-in fast path for `ActionMask.get_steps` that preserves exact action-mask outputs while reducing environment-step cost.

**Architecture:** Keep the original HOPE behavior as the default path. Add a `fast_get_steps` constructor flag to `src/model/action_mask.py`, implement the optimized calculation as a separate private method, and expose the mode only through explicit test/profiling hooks until a later 20K candidate run is approved.

**Tech Stack:** Python, NumPy, unittest, existing HOPE `.venv`, existing Stage 3 profiling tooling under `tools/stage3/`.

---

## File Structure

- Modify: `src/model/action_mask.py`
  - Add `fast_get_steps: bool = False`.
  - Keep `get_steps()` defaulting to the original implementation.
  - Add `_get_steps_original()` and `_get_steps_fast()` so the optimized branch is opt-in and reviewable.
- Create: `tools/stage3/tests/test_fast_action_mask.py`
  - Unit tests proving default behavior is original/off and fast path matches original on deterministic edge and seeded random lidar samples.
- Modify: `tools/stage3/profile_stage3_components.py`
  - Add `--action-mask-mode {original,fast}` for bounded diagnostics only.
  - Apply fast mode only to diagnostic-created env/agent objects.
- Modify: `tools/stage3/tests/test_profile_stage3_components.py`
  - Add coverage for the new profile argument and invalid mode validation.
- Create: `docs/research/2026-06-19-stage3-fast-action-mask-opt-in.md`
  - Record why this is the first safe optimization target, parity evidence, profiler evidence, and boundaries.
- Modify: `task_plan.md`, `findings.md`, `progress.md`
  - Record implementation status and verification results.

---

### Task 1: Add Failing Action-Mask Parity Tests

**Files:**
- Create: `tools/stage3/tests/test_fast_action_mask.py`

- [x] **Step 1: Write the failing test**

Create `tools/stage3/tests/test_fast_action_mask.py`:

```python
from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from model.action_mask import ActionMask  # noqa: E402


class FastActionMaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.original = ActionMask()
        cls.fast = ActionMask(fast_get_steps=True)

    def assert_fast_matches_original(self, lidar: np.ndarray) -> None:
        expected = self.original.get_steps(lidar.copy())
        actual = self.fast.get_steps(lidar.copy())
        np.testing.assert_array_equal(actual, expected)

    def test_default_fast_get_steps_is_disabled(self) -> None:
        self.assertFalse(self.original.fast_get_steps)

    def test_fast_get_steps_matches_original_for_edge_lidar_samples(self) -> None:
        for value in (-5.0, 0.0, 0.01, 1.0, 9.99, 10.0, 25.0):
            with self.subTest(value=value):
                lidar = np.full((self.original.lidar_num,), value, dtype=float)
                self.assert_fast_matches_original(lidar)

    def test_fast_get_steps_matches_original_for_seeded_random_lidar_samples(self) -> None:
        rng = np.random.default_rng(20260619)
        for index in range(32):
            with self.subTest(index=index):
                lidar = rng.uniform(-2.0, 14.0, size=(self.original.lidar_num,))
                self.assert_fast_matches_original(lidar)


if __name__ == "__main__":
    unittest.main()
```

- [x] **Step 2: Run the new test to verify it fails**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_fast_action_mask -v
```

Expected: FAIL or ERROR because `ActionMask.__init__()` does not yet accept `fast_get_steps`.

---

### Task 2: Implement Default-Off Fast Action Mask

**Files:**
- Modify: `src/model/action_mask.py`
- Test: `tools/stage3/tests/test_fast_action_mask.py`

- [x] **Step 1: Add the constructor flag and dispatch**

Change the constructor signature and store the flag:

```python
class ActionMask():
    def __init__(self, VehicleBox=VehicleBox, n_iter=10, fast_get_steps=False) -> None:
        print('initializing action mask')
        self.fast_get_steps = fast_get_steps
```

Update `get_steps()` to dispatch:

```python
    def get_steps(self, raw_lidar_obs:np.ndarray):
        if self.fast_get_steps:
            return self._get_steps_fast(raw_lidar_obs)
        return self._get_steps_original(raw_lidar_obs)
```

- [x] **Step 2: Preserve the original body unchanged in `_get_steps_original`**

Move the current `get_steps()` body into:

```python
    def _get_steps_original(self, raw_lidar_obs:np.ndarray):
        '''
        raw_lidar_obs: the raw lidar obs which already substract the vehicle base.
        '''
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1,1,1)
        step_save = np.zeros_like(self.dist_star)
        step_save[self.dist_star<=dist_obs] = 1
        step_save[self.dist_star>dist_obs] = 0
        max_step = np.argmin(step_save, axis=-1)
        max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter

        step_len = np.min(max_step, axis=0)

        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
        return step_len
```

- [x] **Step 3: Add the optimized exact-output method**

Add:

```python
    def _get_steps_fast(self, raw_lidar_obs:np.ndarray):
        '''
        raw_lidar_obs: the raw lidar obs which already substract the vehicle base.
        '''
        lidar_obs = np.clip(raw_lidar_obs, 0, 10) + self.vehicle_lidar_base
        dist_obs = self._linear_interpolate(lidar_obs.reshape(-1), self.up_sample_rate).reshape(-1,1,1)
        step_valid = self.dist_star <= dist_obs
        max_step = np.argmin(step_valid, axis=-1)
        max_step[np.all(step_valid, axis=-1)] = self.n_iter

        step_len = np.min(max_step, axis=0)

        step_len = self.post_process(step_len)
        if np.sum(step_len) == 0:
            return np.clip(step_len, 0.01, 1)
        return step_len
```

- [x] **Step 4: Run the parity test to verify it passes**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_fast_action_mask -v
```

Expected: PASS.

---

### Task 3: Add Diagnostic Profiler Mode

**Files:**
- Modify: `tools/stage3/profile_stage3_components.py`
- Modify: `tools/stage3/tests/test_profile_stage3_components.py`

- [x] **Step 1: Write failing profiler tests**

Add assertions to `tools/stage3/tests/test_profile_stage3_components.py`:

```python
    def test_profile_action_mask_settings_accepts_known_modes(self):
        self.assertEqual(profile.profile_action_mask_settings("original")["fast_get_steps"], False)
        self.assertEqual(profile.profile_action_mask_settings("fast")["fast_get_steps"], True)

    def test_profile_action_mask_settings_rejects_unknown_mode(self):
        with self.assertRaises(ValueError):
            profile.profile_action_mask_settings("missing")
```

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_profile_stage3_components -v
```

Expected: FAIL because `profile_action_mask_settings` does not exist.

- [x] **Step 2: Implement the profiler mode**

Add:

```python
ACTION_MASK_MODES = {
    "original": {"fast_get_steps": False, "description": "Default HOPE ActionMask.get_steps behavior."},
    "fast": {"fast_get_steps": True, "description": "Opt-in exact-output fast ActionMask.get_steps path."},
}
```

Add:

```python
def profile_action_mask_settings(action_mask_mode: str) -> dict[str, Any]:
    try:
        return dict(ACTION_MASK_MODES[action_mask_mode])
    except KeyError as exc:
        choices = ", ".join(sorted(ACTION_MASK_MODES))
        raise ValueError(f"unknown action mask mode {action_mask_mode!r}; expected one of: {choices}") from exc
```

Add parser argument:

```python
parser.add_argument("--action-mask-mode", choices=sorted(ACTION_MASK_MODES), default="original")
```

Thread `action_mask_mode` through `_make_env`, `_make_agent`, and `run_diagnostic`. Apply only to diagnostic-created objects:

```python
def _set_action_mask_mode(target: Any, action_mask_mode: str) -> None:
    settings = profile_action_mask_settings(action_mask_mode)
    action_filter = getattr(target, "action_filter", None)
    if action_filter is not None and hasattr(action_filter, "fast_get_steps"):
        action_filter.fast_get_steps = bool(settings["fast_get_steps"])
```

- [x] **Step 3: Run profiler tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest tools.stage3.tests.test_profile_stage3_components -v
```

Expected: PASS.

---

### Task 4: Run Verification And Bounded Profile

**Files:**
- Create: `docs/research/stage3_env_step_detail_fast_action_mask_20260619.json`
- Create: `docs/research/stage3_env_step_detail_fast_action_mask_20260619.md`

- [x] **Step 1: Run Stage 3 unit tests**

Run:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -p "test_*.py" -v
```

Expected: PASS.

- [x] **Step 2: Run bounded fast action-mask profile**

Run:

```powershell
$env:SDL_VIDEODRIVER = "dummy"
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
.\.venv\Scripts\python.exe tools\stage3\profile_stage3_components.py --profile-mode command-only --detail env-step --action-mask-mode fast --episodes 16 --max-steps 160 --updates 20 --json docs\research\stage3_env_step_detail_fast_action_mask_20260619.json --markdown docs\research\stage3_env_step_detail_fast_action_mask_20260619.md
```

Expected: exit code 0, JSON/Markdown written. Treat as measurement only, not training-quality evidence.

- [x] **Step 3: Check protected-source scope**

Run:

```powershell
git diff -- src\train src\env
git diff -- src\model\action_mask.py
```

Expected: no `src/train` or `src/env` diff; only the approved `src/model/action_mask.py` opt-in change under protected source.

---

### Task 5: Document, Commit, And Push

**Files:**
- Create: `docs/research/2026-06-19-stage3-fast-action-mask-opt-in.md`
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`

- [x] **Step 1: Write research note**

Record:

```markdown
# Stage 3 Opt-In Fast Action Mask

- Scope: default-off source optimization in `src/model/action_mask.py`.
- Safety boundary: original `get_steps` behavior remains default; fast path must be explicitly enabled.
- Parity evidence: `tools/stage3/tests/test_fast_action_mask.py`.
- Measurement evidence: `stage3_env_step_detail_fast_action_mask_20260619.*`.
- Not a 20K candidate result: this is only admission evidence for a future 20K run.
```

- [x] **Step 2: Update planning files**

Update `task_plan.md`, `findings.md`, and `progress.md` with the implementation and verification results.

- [x] **Step 3: Run final hygiene checks**

Run:

```powershell
git diff --check -- src\model\action_mask.py tools docs task_plan.md findings.md progress.md
git status -sb
```

Expected: no whitespace errors; changed files match this plan.

- [x] **Step 4: Commit and push**

Run:

```powershell
git add src\model\action_mask.py tools\stage3\profile_stage3_components.py tools\stage3\tests\test_fast_action_mask.py tools\stage3\tests\test_profile_stage3_components.py docs\superpowers\plans\2026-06-19-hope-stage3-fast-action-mask.md docs\research\2026-06-19-stage3-fast-action-mask-opt-in.md docs\research\stage3_env_step_detail_fast_action_mask_20260619.json docs\research\stage3_env_step_detail_fast_action_mask_20260619.md task_plan.md findings.md progress.md
git commit -m "Add opt-in fast action mask"
git push
```

Expected: commit pushed to `origin/codex/stage3-resource-study`, updating the existing draft PR.

---

## Self-Review

- Spec coverage: preserves default HOPE action-mask behavior, adds opt-in optimized path, verifies exact output parity, and collects bounded profiler evidence before any 20K candidate run.
- Placeholder scan: no `TBD`, `TODO`, or unspecified tests.
- Type consistency: `fast_get_steps`, `_get_steps_original`, `_get_steps_fast`, `ACTION_MASK_MODES`, and `profile_action_mask_settings` are named consistently across tasks.
