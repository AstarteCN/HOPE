# HOPE RL-OGM Proxy Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a default-off Stage 4 proxy OGM observation path for HOPE, train/evaluate `ogm + target + action_mask` policies, and gate progress against OGM simulation KPIs without breaking HOPE's action mask, curriculum, Reeds-Shepp hybrid behavior, or existing training defaults.

**Architecture:** Add OGM as a first-class observation modality with an ego-centered proxy OGM rasterizer, default-off environment/wrapper/config/model support, and an independent Stage 4 training/evaluation toolchain. The original `src/train/train_HOPE_sac.py` remains the baseline path; Stage 4 uses an opt-in runner that copies HOPE training semantics and mutates experiment config locally.

**Tech Stack:** Python 3.13 project `.venv`, NumPy, Shapely, Gym spaces, PyTorch, TensorBoard event summaries, PowerShell launch/monitor wrappers, `unittest`, existing HOPE `src/` modules, existing Stage 3 tools.

---

## Scope Check

This plan implements one coherent sub-project: a proxy OGM simulation experiment path. It deliberately excludes real-world OGM dataset ingestion, real-vehicle validation, target-representation bugfixes, and lidar-free action-mask replacement. Those are separate future specs.

## File Structure

Create:

- `D:\Github\HOPE\src\env\ogm.py`
  OGM config, geometry normalization, ego/world transforms, and deterministic rasterization.
- `D:\Github\HOPE\tools\stage4\__init__.py`
  Stage 4 package marker.
- `D:\Github\HOPE\tools\stage4\stage4_ogm_targets.py`
  OGM paper KPI targets, baseline metadata, gate thresholds, and helper functions.
- `D:\Github\HOPE\tools\stage4\stage4_ogm_cases.py`
  Fixed `20 parallel + 50 perpendicular` simulation eval-set generation and loading.
- `D:\Github\HOPE\tools\stage4\eval_stage4_ogm_checkpoint.py`
  Checkpoint evaluator that reports PSR, ANGS, PL, and per-case records.
- `D:\Github\HOPE\tools\stage4\stage4_ogm_progress.py`
  TensorBoard/eval progress gate logic, including 20K baseline gate and 10K grace-stop policy.
- `D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py`
  Opt-in OGM training runner using HOPE training semantics.
- `D:\Github\HOPE\tools\stage4\launch_stage4_ogm.ps1`
  PowerShell launcher with manifest/resource monitoring.
- `D:\Github\HOPE\tools\stage4\monitor_stage4_ogm_progress.ps1`
  One-shot progress monitor for 20K and every additional 10K node.
- `D:\Github\HOPE\tools\stage4\tests\test_ogm_rasterizer.py`
- `D:\Github\HOPE\tools\stage4\tests\test_ogm_env_integration.py`
- `D:\Github\HOPE\tools\stage4\tests\test_ogm_network.py`
- `D:\Github\HOPE\tools\stage4\tests\test_stage4_cases.py`
- `D:\Github\HOPE\tools\stage4\tests\test_stage4_eval_metrics.py`
- `D:\Github\HOPE\tools\stage4\tests\test_stage4_progress_gates.py`
- `D:\Github\HOPE\docs\research\stage4_ogm_fixed_eval_cases_20260620.json`

Modify:

- `D:\Github\HOPE\src\configs.py`
  Add default-off OGM config constants and `ogm_shape` in actor/critic config.
- `D:\Github\HOPE\src\env\car_parking_base.py`
  Add default-off `use_ogm_observation`, observation space, and `render()` OGM key.
- `D:\Github\HOPE\src\env\env_wrapper.py`
  Transpose `ogm` from HWC to CHW like `img`.
- `D:\Github\HOPE\src\model\state_norm.py`
  Add `ogm: False`.
- `D:\Github\HOPE\src\model\network.py`
  Support `lidar_shape=None` and add an explicit OGM encoder branch.
- `D:\Github\HOPE\task_plan.md`
- `D:\Github\HOPE\findings.md`
- `D:\Github\HOPE\progress.md`

Do not modify:

- `D:\Github\HOPE\src\train\train_HOPE_sac.py` during OGM v1.
- `D:\Github\HOPE\src\model\ckpt\*`.
- Training logs, TensorBoard event files, generated checkpoints, or local environment folders.

## Common Commands

Run all commands from PowerShell:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v
git diff --check
git diff -- src\train\train_HOPE_sac.py src\model\ckpt
```

Expected protected-source check: no diff for `src\train\train_HOPE_sac.py` or `src\model\ckpt`.

---

### Task 1: Add Stage 4 Package, Targets, And Plan State

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\__init__.py`
- Create: `D:\Github\HOPE\tools\stage4\stage4_ogm_targets.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_stage4_progress_gates.py`
- Modify: `D:\Github\HOPE\task_plan.md`
- Modify: `D:\Github\HOPE\progress.md`

- [ ] **Step 1: Write the failing target constants and acceptance-window test**

Create `D:\Github\HOPE\tools\stage4\tests\test_stage4_progress_gates.py` with:

```python
from __future__ import annotations

import unittest

from tools.stage4.stage4_ogm_targets import (
    BASELINE_FAST_ACTION_MASK_20K,
    OGM_SIM_TARGETS,
    relative_band,
)


class Stage4TargetTests(unittest.TestCase):
    def test_ogm_sim_targets_match_prd_values(self) -> None:
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["psr"], 0.9933)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["angs"], 1.5)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Normal"]["pl"], 20.3)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["psr"], 0.977)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["angs"], 1.9)
        self.assertEqual(OGM_SIM_TARGETS["Sim-Complex"]["pl"], 23.6)

    def test_relative_band_uses_three_percent_window(self) -> None:
        low, high = relative_band(100.0)
        self.assertAlmostEqual(low, 97.0)
        self.assertAlmostEqual(high, 103.0)

    def test_fast_action_mask_20k_baseline_is_recorded(self) -> None:
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["run_dir"], "src/log/exp/sac_20260620_085208")
        self.assertEqual(BASELINE_FAST_ACTION_MASK_20K["checkpoint"], "SAC_19999.pt")
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["episodes_per_hour"], 1826.840564)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["env_steps_per_second"], 46.223481)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Normal"], 0.985)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Complex"], 0.945)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["Extrem"], 0.655)
        self.assertAlmostEqual(BASELINE_FAST_ACTION_MASK_20K["eval_success"]["DLP"], 0.960)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_progress_gates -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'tools.stage4.stage4_ogm_targets'`.

- [ ] **Step 3: Add Stage 4 target constants**

Create `D:\Github\HOPE\tools\stage4\__init__.py` as an empty file.

Create `D:\Github\HOPE\tools\stage4\stage4_ogm_targets.py` with:

```python
from __future__ import annotations

from typing import Mapping

RELATIVE_TOLERANCE = 0.03
TRAIN_EPISODE_MIN = 80_000
TRAIN_EPISODE_TARGET = 100_000
TRAIN_EPISODE_MAX = 120_000
FIRST_GATE_EPISODE = 20_000
PROGRESS_GATE_INTERVAL = 10_000

OGM_SIM_TARGETS: dict[str, dict[str, float]] = {
    "Sim-Normal": {"psr": 0.9933, "angs": 1.5, "pl": 20.3},
    "Sim-Complex": {"psr": 0.977, "angs": 1.9, "pl": 23.6},
}

BASELINE_FAST_ACTION_MASK_20K: dict[str, object] = {
    "id": "hope-fast-action-mask-20k",
    "run_dir": "src/log/exp/sac_20260620_085208",
    "checkpoint": "SAC_19999.pt",
    "wall_time_hours": 10.947863,
    "episodes_per_hour": 1826.840564,
    "env_steps_per_second": 46.223481,
    "eval_success": {
        "Normal": 0.985,
        "Complex": 0.945,
        "Extrem": 0.655,
        "DLP": 0.960,
    },
}


def relative_band(target: float, tolerance: float = RELATIVE_TOLERANCE) -> tuple[float, float]:
    return target * (1.0 - tolerance), target * (1.0 + tolerance)


def metric_within_relative_band(value: float, target: float, tolerance: float = RELATIVE_TOLERANCE) -> bool:
    low, high = relative_band(target, tolerance)
    return low <= value <= high


def all_ogm_targets_met(results: Mapping[str, Mapping[str, float]]) -> bool:
    for split, targets in OGM_SIM_TARGETS.items():
        observed = results.get(split, {})
        for metric, target in targets.items():
            if metric not in observed:
                return False
            if not metric_within_relative_band(float(observed[metric]), target):
                return False
    return True
```

- [ ] **Step 4: Run the test and verify it passes**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_progress_gates -v
```

Expected: PASS.

- [ ] **Step 5: Update planning files**

In `D:\Github\HOPE\task_plan.md`, add Phase 12:

```markdown
### Phase 12: Stage 4 OGM Proxy Implementation

- [ ] Add Stage 4 target constants and gates.
- [ ] Implement proxy OGM rasterizer.
- [ ] Integrate OGM observation as a default-off environment path.
- [ ] Add explicit OGM network/config/state-normalization support.
- [ ] Add fixed OGM-style simulation evaluation set.
- [ ] Add Stage 4 OGM checkpoint evaluation and progress gates.
- [ ] Add Stage 4 opt-in training launcher and monitor.
- [ ] Run 1K, 20K, and gated 10K-progress validation ladder.
- **Status:** implementation plan written; waiting for execution mode selection.
```

In `D:\Github\HOPE\progress.md`, append:

```markdown
### Stage 4 OGM Proxy Implementation Planning

- **Status:** implementation plan started.
- Added `docs/superpowers/plans/2026-06-20-hope-rl-ogm-proxy-integration.md`.
- The plan includes the user-requested validation policy: first gate at 20K episodes, then monitoring every 10K episodes with a one-window grace rule before stopping a stagnant or degrading run.
```

- [ ] **Step 6: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/__init__.py tools/stage4/stage4_ogm_targets.py tools/stage4/tests/test_stage4_progress_gates.py task_plan.md progress.md
git commit -m "Add Stage 4 OGM target constants"
```

---

### Task 2: Implement Proxy OGM Rasterizer

**Files:**
- Create: `D:\Github\HOPE\src\env\ogm.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_ogm_rasterizer.py`

- [ ] **Step 1: Write failing rasterizer tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_ogm_rasterizer.py` with:

```python
from __future__ import annotations

import math
import sys
import unittest
from pathlib import Path

import numpy as np
from shapely.geometry import LinearRing

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.ogm import OGMConfig, build_ego_ogm, world_to_grid  # noqa: E402
from env.vehicle import State  # noqa: E402


class ProxyOGMRasterizerTests(unittest.TestCase):
    def test_world_to_grid_places_forward_points_right_of_center(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        row, col = world_to_grid(ego_state=State([0.0, 0.0, 0.0]), x=2.0, y=0.0, config=config)
        self.assertEqual(row, 32)
        self.assertGreater(col, 32)

    def test_world_to_grid_places_left_points_above_center(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        row, col = world_to_grid(ego_state=State([0.0, 0.0, 0.0]), x=0.0, y=2.0, config=config)
        self.assertLess(row, 32)
        self.assertEqual(col, 32)

    def test_world_to_grid_respects_ego_heading(self) -> None:
        config = OGMConfig(size=64, resolution=0.5, channels=2)
        ego = State([0.0, 0.0, math.pi / 2.0])
        row, col = world_to_grid(ego_state=ego, x=0.0, y=2.0, config=config)
        self.assertEqual(row, 32)
        self.assertGreater(col, 32)

    def test_build_ego_ogm_marks_obstacle_and_target_channels(self) -> None:
        config = OGMConfig(size=64, resolution=0.25, channels=2)
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        ogm = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        self.assertEqual(ogm.shape, (64, 64, 2))
        self.assertEqual(ogm.dtype, np.float32)
        self.assertGreater(float(ogm[:, :, 0].sum()), 0.0)
        self.assertGreater(float(ogm[:, :, 1].sum()), 0.0)

    def test_build_ego_ogm_is_deterministic(self) -> None:
        config = OGMConfig(size=32, resolution=0.5, channels=2)
        ego = State([0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (2.0, -0.5), (2.0, 0.5), (1.0, 0.5)])
        target = LinearRing([(-1.0, -0.5), (-0.25, -0.5), (-0.25, 0.5), (-1.0, 0.5)])

        first = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)
        second = build_ego_ogm(ego_state=ego, obstacles=[obstacle], target_box=target, config=config)

        np.testing.assert_array_equal(first, second)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_rasterizer -v
```

Expected: FAIL with `ModuleNotFoundError: No module named 'env.ogm'`.

- [ ] **Step 3: Implement the rasterizer**

Create `D:\Github\HOPE\src\env\ogm.py` with:

```python
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable

import numpy as np
from shapely.geometry import LinearRing, Point, Polygon
from shapely.prepared import prep


@dataclass(frozen=True)
class OGMConfig:
    size: int = 64
    resolution: float = 1.0 / 3.0
    channels: int = 2
    obstacle_channel: int = 0
    target_channel: int = 1
    obstacle_value: float = 1.0
    target_value: float = 1.0
    dtype: type = np.float32

    def __post_init__(self) -> None:
        if self.size <= 0:
            raise ValueError("OGMConfig.size must be positive.")
        if self.resolution <= 0:
            raise ValueError("OGMConfig.resolution must be positive.")
        if self.channels < 2:
            raise ValueError("OGMConfig.channels must be at least 2.")


DEFAULT_OGM_CONFIG = OGMConfig()


def _ring_from_object(item: object) -> LinearRing:
    if isinstance(item, LinearRing):
        return item
    shape = getattr(item, "shape", None)
    if isinstance(shape, LinearRing):
        return shape
    raise TypeError(f"Unsupported OGM geometry object: {type(item)!r}")


def world_to_local(ego_state: object, x: float, y: float) -> tuple[float, float]:
    ego_x = float(ego_state.loc.x)
    ego_y = float(ego_state.loc.y)
    heading = float(ego_state.heading)
    dx = float(x) - ego_x
    dy = float(y) - ego_y
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    local_x = cos_h * dx + sin_h * dy
    local_y = -sin_h * dx + cos_h * dy
    return float(local_x), float(local_y)


def world_to_grid(ego_state: object, x: float, y: float, config: OGMConfig = DEFAULT_OGM_CONFIG) -> tuple[int, int]:
    local_x, local_y = world_to_local(ego_state, x, y)
    center = config.size // 2
    col = int(np.floor(local_x / config.resolution + center))
    row = int(np.floor(center - local_y / config.resolution))
    row = int(np.clip(row, 0, config.size - 1))
    col = int(np.clip(col, 0, config.size - 1))
    return row, col


def _cell_center_world(ego_state: object, row: int, col: int, config: OGMConfig) -> tuple[float, float]:
    center = config.size / 2.0
    local_x = (float(col) + 0.5 - center) * config.resolution
    local_y = (center - float(row) - 0.5) * config.resolution
    heading = float(ego_state.heading)
    cos_h = np.cos(heading)
    sin_h = np.sin(heading)
    world_x = float(ego_state.loc.x) + cos_h * local_x - sin_h * local_y
    world_y = float(ego_state.loc.y) + sin_h * local_x + cos_h * local_y
    return float(world_x), float(world_y)


def _rasterize_ring(
    grid: np.ndarray,
    ego_state: object,
    ring: LinearRing,
    channel: int,
    value: float,
    config: OGMConfig,
) -> None:
    polygon = Polygon(ring)
    if polygon.is_empty:
        return
    prepared = prep(polygon)
    coords = np.asarray(ring.coords, dtype=float)
    grid_points = [world_to_grid(ego_state, x, y, config) for x, y in coords]
    rows = [p[0] for p in grid_points]
    cols = [p[1] for p in grid_points]
    min_row = max(min(rows) - 2, 0)
    max_row = min(max(rows) + 2, config.size - 1)
    min_col = max(min(cols) - 2, 0)
    max_col = min(max(cols) + 2, config.size - 1)
    for row in range(min_row, max_row + 1):
        for col in range(min_col, max_col + 1):
            world_x, world_y = _cell_center_world(ego_state, row, col, config)
            point = Point(world_x, world_y)
            if prepared.contains(point) or polygon.touches(point):
                grid[row, col, channel] = value


def build_ego_ogm(
    ego_state: object,
    obstacles: Iterable[object],
    target_box: LinearRing,
    config: OGMConfig = DEFAULT_OGM_CONFIG,
) -> np.ndarray:
    grid = np.zeros((config.size, config.size, config.channels), dtype=config.dtype)
    for obstacle in obstacles:
        _rasterize_ring(
            grid=grid,
            ego_state=ego_state,
            ring=_ring_from_object(obstacle),
            channel=config.obstacle_channel,
            value=config.obstacle_value,
            config=config,
        )
    _rasterize_ring(
        grid=grid,
        ego_state=ego_state,
        ring=_ring_from_object(target_box),
        channel=config.target_channel,
        value=config.target_value,
        config=config,
    )
    return grid
```

- [ ] **Step 4: Run the rasterizer tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_rasterizer -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
cd D:\Github\HOPE
git add src/env/ogm.py tools/stage4/tests/test_ogm_rasterizer.py
git commit -m "Add proxy OGM rasterizer"
```

---

### Task 3: Integrate Default-Off OGM Observation Into Environment And Wrapper

**Files:**
- Modify: `D:\Github\HOPE\src\configs.py`
- Modify: `D:\Github\HOPE\src\env\car_parking_base.py`
- Modify: `D:\Github\HOPE\src\env\env_wrapper.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_ogm_env_integration.py`

- [ ] **Step 1: Write failing environment integration tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_ogm_env_integration.py` with:

```python
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402


class OGMEnvironmentIntegrationTests(unittest.TestCase):
    def test_default_env_does_not_advertise_ogm_observation_space(self) -> None:
        env = CarParking(render_mode="rgb_array", verbose=False)
        try:
            self.assertNotIn("ogm", env.observation_space)
        finally:
            env.close()

    def test_ogm_enabled_env_returns_hwc_ogm(self) -> None:
        env = CarParking(
            render_mode="rgb_array",
            verbose=False,
            use_img_observation=False,
            use_lidar_observation=True,
            use_action_mask=True,
            use_ogm_observation=True,
        )
        try:
            obs = env.reset(0, None, "Normal")
            self.assertIn("ogm", obs)
            self.assertEqual(obs["ogm"].shape, (64, 64, 2))
            self.assertEqual(obs["ogm"].dtype, np.float32)
            self.assertGreaterEqual(float(obs["ogm"].min()), 0.0)
            self.assertLessEqual(float(obs["ogm"].max()), 1.0)
            self.assertIsNotNone(obs["action_mask"])
            self.assertIsNotNone(obs["target"])
        finally:
            env.close()

    def test_wrapper_transposes_ogm_to_chw(self) -> None:
        raw_env = CarParking(
            render_mode="rgb_array",
            verbose=False,
            use_img_observation=False,
            use_lidar_observation=True,
            use_action_mask=True,
            use_ogm_observation=True,
        )
        env = CarParkingWrapper(raw_env)
        try:
            obs = env.reset(0, None, "Normal")
            self.assertEqual(env.observation_shape["ogm"], (2, 64, 64))
            self.assertEqual(obs["ogm"].shape, (2, 64, 64))
        finally:
            raw_env.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_env_integration -v
```

Expected: FAIL because `CarParking.__init__()` does not accept `use_ogm_observation`.

- [ ] **Step 3: Add default-off OGM config constants**

Modify `D:\Github\HOPE\src\configs.py` near the existing observation toggles:

```python
USE_LIDAR = True
USE_IMG = True
USE_ACTION_MASK = True
USE_OGM = False
OGM_SIZE = 64
OGM_RESOLUTION = 1.0 / 3.0
OGM_CHANNELS = 2
```

Modify `ACTOR_CONFIGS` and `CRITIC_CONFIGS` to include:

```python
'ogm_shape': (OGM_CHANNELS, OGM_SIZE, OGM_SIZE) if USE_OGM else None,
```

Do not change the existing default values for `USE_LIDAR`, `USE_IMG`, or `USE_ACTION_MASK`.

- [ ] **Step 4: Add environment OGM observation support**

Modify `D:\Github\HOPE\src\env\car_parking_base.py` imports:

```python
from env.ogm import OGMConfig, build_ego_ogm
```

Modify `CarParking.__init__()` signature:

```python
        use_lidar_observation: bool = USE_LIDAR,
        use_img_observation: bool = USE_IMG,
        use_action_mask: bool = USE_ACTION_MASK,
        use_ogm_observation: bool = USE_OGM,
```

Set the flag:

```python
        self.use_ogm_observation = use_ogm_observation
```

After image observation-space setup, add:

```python
        if self.use_ogm_observation:
            self.ogm_config = OGMConfig(
                size=OGM_SIZE,
                resolution=OGM_RESOLUTION,
                channels=OGM_CHANNELS,
            )
            self.observation_space["ogm"] = spaces.Box(
                low=0,
                high=1,
                shape=(OGM_SIZE, OGM_SIZE, OGM_CHANNELS),
                dtype=np.float32,
            )
```

Add a method:

```python
    def _get_ogm_observation(self):
        return build_ego_ogm(
            ego_state=self.vehicle.state,
            obstacles=self.map.obstacles,
            target_box=self.map.dest_box,
            config=self.ogm_config,
        )
```

Modify `render()` initial observation dict:

```python
        observation = {"img": None, "lidar": None, "target": None, "action_mask": None, "ogm": None}
```

Add before target assignment:

```python
        if self.use_ogm_observation:
            observation["ogm"] = self._get_ogm_observation()
```

- [ ] **Step 5: Add wrapper OGM transpose**

Modify `D:\Github\HOPE\src\env\env_wrapper.py`:

```python
def observation_rescale(obs):
    if obs['img'] is not None:
        obs['img'] = obs['img'].transpose((2, 0, 1))
    if obs.get('ogm') is not None:
        obs['ogm'] = obs['ogm'].transpose((2, 0, 1))
    return obs
```

Modify `CarParkingWrapper.__init__()`:

```python
        if 'ogm' in self.observation_shape:
            w, h, c = self.observation_shape['ogm']
            self.observation_shape['ogm'] = (c, w, h)
```

- [ ] **Step 6: Run environment integration tests**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_env_integration -v
```

Expected: PASS.

- [ ] **Step 7: Run Stage 3 tests to protect defaults**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v
```

Expected: PASS. Any failure caused by default OGM activation is a blocking regression.

- [ ] **Step 8: Commit**

```powershell
cd D:\Github\HOPE
git add src/configs.py src/env/car_parking_base.py src/env/env_wrapper.py tools/stage4/tests/test_ogm_env_integration.py
git commit -m "Add default-off OGM environment observation"
```

---

### Task 4: Add OGM Model, Config, StateNorm, And SAC Input Support

**Files:**
- Modify: `D:\Github\HOPE\src\model\state_norm.py`
- Modify: `D:\Github\HOPE\src\model\network.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_ogm_network.py`

- [ ] **Step 1: Write failing network and state-normalization tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_ogm_network.py` with:

```python
from __future__ import annotations

import sys
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from configs import ACTOR_CONFIGS, CRITIC_CONFIGS, N_DISCRETE_ACTION  # noqa: E402
from model.agent.sac_agent import SACAgent  # noqa: E402
from model.network import MultiObsEmbedding, SACCriticAdapter  # noqa: E402
from model.state_norm import DEFAULT_UPDATE_MODAL, StateNorm  # noqa: E402


def actor_config() -> dict:
    config = deepcopy(ACTOR_CONFIGS)
    config.update(
        {
            "n_modal": 3,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": (2, 64, 64),
            "action_mask_shape": N_DISCRETE_ACTION,
        }
    )
    return config


def critic_config() -> dict:
    config = deepcopy(CRITIC_CONFIGS)
    config.update(
        {
            "n_modal": 4,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": (2, 64, 64),
            "action_mask_shape": N_DISCRETE_ACTION,
        }
    )
    return config


class OGMNetworkTests(unittest.TestCase):
    def test_state_norm_knows_ogm_is_not_running_normalized(self) -> None:
        self.assertIn("ogm", DEFAULT_UPDATE_MODAL)
        self.assertFalse(DEFAULT_UPDATE_MODAL["ogm"])
        norm = StateNorm({"target": (5,), "action_mask": (N_DISCRETE_ACTION,), "ogm": (2, 64, 64)})
        obs = {
            "target": np.zeros((5,), dtype=np.float32),
            "action_mask": np.ones((N_DISCRETE_ACTION,), dtype=np.float32),
            "ogm": np.ones((2, 64, 64), dtype=np.float32),
        }
        normalized = norm.state_norm(obs.copy(), update=True)
        np.testing.assert_array_equal(normalized["ogm"], obs["ogm"])

    def test_actor_accepts_target_action_mask_and_ogm_without_lidar_or_img(self) -> None:
        net = MultiObsEmbedding(actor_config())
        obs = {
            "target": torch.zeros((1, 5), dtype=torch.float32),
            "action_mask": torch.ones((1, N_DISCRETE_ACTION), dtype=torch.float32),
            "ogm": torch.zeros((1, 2, 64, 64), dtype=torch.float32),
        }
        output = net(obs)
        self.assertEqual(tuple(output.shape), (1, 2))
        self.assertTrue(torch.isfinite(output).all())

    def test_critic_accepts_target_action_mask_ogm_and_action(self) -> None:
        critic = SACCriticAdapter(critic_config())
        obs = {
            "target": torch.zeros((1, 5), dtype=torch.float32),
            "action_mask": torch.ones((1, N_DISCRETE_ACTION), dtype=torch.float32),
            "ogm": torch.zeros((1, 2, 64, 64), dtype=torch.float32),
            "action": torch.zeros((1, 2), dtype=torch.float32),
        }
        output = critic(obs)
        self.assertEqual(tuple(output.shape), (1, 1))
        self.assertTrue(torch.isfinite(output).all())

    def test_sac_obs2tensor_ignores_extra_lidar_key_when_config_excludes_lidar(self) -> None:
        config = {
            "discrete": False,
            "observation_shape": {
                "target": (5,),
                "action_mask": (N_DISCRETE_ACTION,),
                "ogm": (2, 64, 64),
            },
            "action_dim": 2,
            "actor_layers": actor_config(),
            "critic_layers": critic_config(),
        }
        agent = SACAgent(config)
        obs = {
            "target": np.zeros((5,), dtype=np.float32),
            "action_mask": np.ones((N_DISCRETE_ACTION,), dtype=np.float32),
            "ogm": np.zeros((2, 64, 64), dtype=np.float32),
            "lidar": np.zeros((120,), dtype=np.float32),
        }
        tensor_obs = agent.obs2tensor(obs)
        self.assertIn("ogm", tensor_obs)
        self.assertNotIn("lidar", tensor_obs)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_network -v
```

Expected: FAIL because `DEFAULT_UPDATE_MODAL` lacks `ogm`, and `MultiObsEmbedding` does not support `ogm_shape` or `lidar_shape=None`.

- [ ] **Step 3: Update StateNorm**

Modify `D:\Github\HOPE\src\model\state_norm.py`:

```python
DEFAULT_UPDATE_MODAL = {'img': False, 'lidar': True, 'target': True, 'action_mask': False, 'ogm': False}
```

- [ ] **Step 4: Update MultiObsEmbedding for optional lidar and explicit OGM**

Modify `D:\Github\HOPE\src\model\network.py` inside `MultiObsEmbedding.__init__()`:

```python
        self.use_lidar = False if configs['lidar_shape'] is None else True
        self.use_img = False if configs['img_shape'] is None else True
        self.use_ogm = False if configs.get('ogm_shape') is None else True
```

Guard lidar embedding creation:

```python
        if configs['lidar_shape'] is not None:
            layers = [nn.Linear(configs['lidar_shape'], embed_size)]
            for _ in range(configs['n_embed_layers'] - 1):
                layers.append(activate_func)
                layers.append(nn.Linear(embed_size, embed_size))
            self.embed_lidar = nn.Sequential(*layers)
```

Add OGM embedding creation after image embedding:

```python
        if configs.get('ogm_shape') is not None:
            self.embed_ogm = ImgEncoder(
                configs['ogm_shape'],
                configs['k_img_conv'],
                embed_size,
                configs['img_conv_layers'],
                configs['img_linear_layers'],
            )
            self.re_embed_ogm = nn.Sequential(activate_func, nn.Linear(embed_size, embed_size))
```

Guard lidar orthogonal initialization:

```python
        if self.use_lidar:
            for layer_name, layer in self.embed_lidar.state_dict().items():
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
```

Add OGM orthogonal initialization:

```python
        if self.use_ogm:
            for layer_name, layer in self.re_embed_ogm.state_dict().items():
                gain = 1
                if layer_name.endswith("weight"):
                    nn.init.orthogonal_(layer, gain=gain)
                elif layer_name.endswith("bias"):
                    nn.init.constant_(layer, 0)
```

Modify `forward()` feature construction:

```python
        features = []
        if self.use_lidar:
            features.append(self.embed_lidar(x['lidar']))
        feature_target = self.embed_tgt(x['target'])
        features.append(feature_target)
        if self.use_action_mask:
            feature_am = self.embed_am(x['action_mask'])
            features.append(feature_am)
        if self.use_img:
            feature_img, _ = self.embed_img(x['img'])
            feature_img = self.re_embed_img(feature_img)
            features.append(feature_img)
        if self.use_ogm:
            feature_ogm, _ = self.embed_ogm(x['ogm'])
            feature_ogm = self.re_embed_ogm(feature_ogm)
            features.append(feature_ogm)
```

Keep the critic action feature append after these modality features.

- [ ] **Step 5: Run network tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_ogm_network -v
```

Expected: PASS.

- [ ] **Step 6: Run Stage 3 tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v
```

Expected: PASS. This protects the existing lidar/img/action-mask network path.

- [ ] **Step 7: Commit**

```powershell
cd D:\Github\HOPE
git add src/model/state_norm.py src/model/network.py tools/stage4/tests/test_ogm_network.py
git commit -m "Add explicit OGM network modality"
```

---

### Task 5: Add Fixed OGM-Style Evaluation Cases

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\stage4_ogm_cases.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_stage4_cases.py`
- Create: `D:\Github\HOPE\docs\research\stage4_ogm_fixed_eval_cases_20260620.json`

- [ ] **Step 1: Write failing fixed-case tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_stage4_cases.py` with:

```python
from __future__ import annotations

import unittest

from tools.stage4.stage4_ogm_cases import build_fixed_eval_cases, split_case_counts


class Stage4FixedCaseTests(unittest.TestCase):
    def test_fixed_eval_cases_have_20_parallel_and_50_perpendicular(self) -> None:
        cases = build_fixed_eval_cases()
        counts = split_case_counts(cases)
        self.assertEqual(counts["parallel"], 20)
        self.assertEqual(counts["perpendicular"], 50)
        self.assertEqual(len(cases), 70)

    def test_fixed_eval_cases_have_stable_ids_and_hope_case_ids(self) -> None:
        cases = build_fixed_eval_cases()
        self.assertEqual(cases[0]["case_uid"], "parallel_000")
        self.assertEqual(cases[0]["hope_case_id"], 1)
        self.assertEqual(cases[20]["case_uid"], "perpendicular_000")
        self.assertEqual(cases[20]["hope_case_id"], 0)
        self.assertEqual(len({case["seed"] for case in cases}), 70)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_cases -v
```

Expected: FAIL with missing `tools.stage4.stage4_ogm_cases`.

- [ ] **Step 3: Implement fixed-case generation**

Create `D:\Github\HOPE\tools\stage4\stage4_ogm_cases.py` with:

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Iterable


DEFAULT_CASE_FILE = Path("docs/research/stage4_ogm_fixed_eval_cases_20260620.json")
PARALLEL_COUNT = 20
PERPENDICULAR_COUNT = 50
BASE_SEED = 20260620


def build_fixed_eval_cases() -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for index in range(PARALLEL_COUNT):
        cases.append(
            {
                "case_uid": f"parallel_{index:03d}",
                "split": "Sim-Complex",
                "parking_type": "parallel",
                "hope_level": "Complex",
                "hope_case_id": 1,
                "seed": BASE_SEED + index,
            }
        )
    for index in range(PERPENDICULAR_COUNT):
        cases.append(
            {
                "case_uid": f"perpendicular_{index:03d}",
                "split": "Sim-Normal",
                "parking_type": "perpendicular",
                "hope_level": "Normal",
                "hope_case_id": 0,
                "seed": BASE_SEED + PARALLEL_COUNT + index,
            }
        )
    return cases


def split_case_counts(cases: Iterable[dict[str, object]]) -> dict[str, int]:
    counts = {"parallel": 0, "perpendicular": 0}
    for case in cases:
        counts[str(case["parking_type"])] += 1
    return counts


def write_cases(path: str | Path = DEFAULT_CASE_FILE) -> Path:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    cases = build_fixed_eval_cases()
    output_path.write_text(json.dumps(cases, indent=2, ensure_ascii=False), encoding="utf-8")
    return output_path


def load_cases(path: str | Path = DEFAULT_CASE_FILE) -> list[dict[str, object]]:
    input_path = Path(path)
    return json.loads(input_path.read_text(encoding="utf-8"))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=DEFAULT_CASE_FILE)
    args = parser.parse_args()
    path = write_cases(args.output)
    print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Generate the fixed eval-case JSON**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe .\tools\stage4\stage4_ogm_cases.py --output .\docs\research\stage4_ogm_fixed_eval_cases_20260620.json
```

Expected: prints `docs\research\stage4_ogm_fixed_eval_cases_20260620.json`.

- [ ] **Step 5: Run tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_cases -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/stage4_ogm_cases.py tools/stage4/tests/test_stage4_cases.py docs/research/stage4_ogm_fixed_eval_cases_20260620.json
git commit -m "Add fixed OGM simulation eval cases"
```

---

### Task 6: Add OGM Evaluation Metrics And Checkpoint Evaluator

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\eval_stage4_ogm_checkpoint.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_stage4_eval_metrics.py`

- [ ] **Step 1: Write failing metric tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_stage4_eval_metrics.py` with:

```python
from __future__ import annotations

import unittest

from tools.stage4.eval_stage4_ogm_checkpoint import (
    count_gear_shifts,
    summarize_eval_records,
)


class Stage4EvalMetricTests(unittest.TestCase):
    def test_count_gear_shifts_counts_speed_sign_changes_ignoring_zero(self) -> None:
        self.assertEqual(count_gear_shifts([1.0, 1.0, -1.0, -1.0, 0.0, 1.0]), 2)
        self.assertEqual(count_gear_shifts([0.0, 0.0, 1.0, 1.0]), 0)
        self.assertEqual(count_gear_shifts([-1.0, -1.0, -1.0]), 0)

    def test_summarize_eval_records_reports_psr_angs_and_pl_by_split(self) -> None:
        records = [
            {"split": "Sim-Normal", "success": True, "gear_shifts": 1, "path_length": 10.0},
            {"split": "Sim-Normal", "success": False, "gear_shifts": 3, "path_length": 30.0},
            {"split": "Sim-Complex", "success": True, "gear_shifts": 2, "path_length": 20.0},
        ]
        summary = summarize_eval_records(records)
        self.assertAlmostEqual(summary["Sim-Normal"]["psr"], 0.5)
        self.assertAlmostEqual(summary["Sim-Normal"]["angs"], 1.0)
        self.assertAlmostEqual(summary["Sim-Normal"]["pl"], 10.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["psr"], 1.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["angs"], 2.0)
        self.assertAlmostEqual(summary["Sim-Complex"]["pl"], 20.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_eval_metrics -v
```

Expected: FAIL with missing `tools.stage4.eval_stage4_ogm_checkpoint`.

- [ ] **Step 3: Implement metric helpers and evaluator skeleton**

Create `D:\Github\HOPE\tools\stage4\eval_stage4_ogm_checkpoint.py` with:

```python
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
from env.vehicle import Status  # noqa: E402
from model.agent.parking_agent import ParkingAgent  # noqa: E402
from model.agent.sac_agent import SAC as SACAgent  # noqa: E402
from parking_planner import RsPlanner  # noqa: E402
from configs import N_DISCRETE_ACTION, VALID_SPEED  # noqa: E402
from tools.stage4.stage4_ogm_cases import load_cases  # noqa: E402


def count_gear_shifts(speeds: Iterable[float]) -> int:
    last_sign = 0
    shifts = 0
    for speed in speeds:
        sign = 1 if speed > 0 else -1 if speed < 0 else 0
        if sign == 0:
            continue
        if last_sign != 0 and sign != last_sign:
            shifts += 1
        last_sign = sign
    return shifts


def summarize_eval_records(records: Iterable[Mapping[str, object]]) -> dict[str, dict[str, float]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for record in records:
        grouped[str(record["split"])].append(record)
    summary: dict[str, dict[str, float]] = {}
    for split, split_records in grouped.items():
        successes = [record for record in split_records if bool(record["success"])]
        psr = len(successes) / len(split_records) if split_records else 0.0
        if successes:
            angs = float(np.mean([float(record["gear_shifts"]) for record in successes]))
            pl = float(np.mean([float(record["path_length"]) for record in successes]))
        else:
            angs = float("inf")
            pl = float("inf")
        summary[split] = {"psr": psr, "angs": angs, "pl": pl}
    return summary


def build_ogm_agent_config(env: CarParkingWrapper) -> dict[str, object]:
    actor_layers = {
        "n_modal": 3,
        "lidar_shape": None,
        "target_shape": 5,
        "action_mask_shape": N_DISCRETE_ACTION,
        "img_shape": None,
        "ogm_shape": (2, 64, 64),
        "output_size": 2,
        "embed_size": 128,
        "hidden_size": 256,
        "n_hidden_layers": 3,
        "n_embed_layers": 2,
        "img_conv_layers": [4, 8],
        "img_linear_layers": [256],
        "k_img_conv": 3,
        "orthogonal_init": True,
        "use_tanh_output": True,
        "use_tanh_activate": True,
        "attention_configs": {"depth": 1, "heads": 8, "dim_head": 32, "mlp_dim": 128, "hidden_dim": 128},
    }
    critic_layers = dict(actor_layers)
    critic_layers.update({"n_modal": 4, "output_size": 1, "use_tanh_output": False, "input_action_dim": 2})
    return {
        "discrete": False,
        "observation_shape": {
            "target": env.observation_shape["target"],
            "action_mask": env.observation_shape["action_mask"],
            "ogm": env.observation_shape["ogm"],
        },
        "action_dim": env.action_space.shape[0],
        "actor_layers": actor_layers,
        "critic_layers": critic_layers,
    }


def evaluate_checkpoint(checkpoint_path: Path, cases_path: Path, output_json: Path) -> dict[str, object]:
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
    raw_env = CarParking(
        render_mode="rgb_array",
        verbose=False,
        use_img_observation=False,
        use_lidar_observation=True,
        use_action_mask=True,
        use_ogm_observation=True,
    )
    env = CarParkingWrapper(raw_env)
    try:
        rl_agent = SACAgent(build_ogm_agent_config(env))
        rl_agent.load(str(checkpoint_path), params_only=True)
        step_ratio = env.vehicle.kinetic_model.step_len * env.vehicle.kinetic_model.n_step * VALID_SPEED[1]
        parking_agent = ParkingAgent(rl_agent, RsPlanner(step_ratio))
        records = []
        for case in load_cases(cases_path):
            np.random.seed(int(case["seed"]))
            obs = env.reset(int(case["hope_case_id"]), None, str(case["hope_level"]))
            parking_agent.reset()
            done = False
            step_num = 0
            path_length = 0.0
            speeds = []
            last_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
            while not done:
                step_num += 1
                action, _ = parking_agent.get_action(obs)
                speeds.append(float(action[1]))
                next_obs, reward, done, info = env.step(action)
                obs = next_obs
                current_xy = (env.vehicle.state.loc.x, env.vehicle.state.loc.y)
                path_length += float(np.linalg.norm(np.array(last_xy) - np.array(current_xy)))
                last_xy = current_xy
                if info["path_to_dest"] is not None:
                    parking_agent.set_planner_path(info["path_to_dest"])
            records.append(
                {
                    "case_uid": case["case_uid"],
                    "split": case["split"],
                    "success": info["status"] == Status.ARRIVED,
                    "status": str(info["status"]),
                    "step_num": step_num,
                    "gear_shifts": count_gear_shifts(speeds),
                    "path_length": path_length,
                }
            )
        result = {"records": records, "summary": summarize_eval_records(records)}
        output_json.parent.mkdir(parents=True, exist_ok=True)
        output_json.write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
        return result
    finally:
        raw_env.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True, type=Path)
    parser.add_argument("--cases", required=True, type=Path)
    parser.add_argument("--output-json", required=True, type=Path)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.cases, args.output_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run metric tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_eval_metrics -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/eval_stage4_ogm_checkpoint.py tools/stage4/tests/test_stage4_eval_metrics.py
git commit -m "Add Stage 4 OGM evaluation metrics"
```

---

### Task 7: Add 20K And Every-10K Progress Gate Logic

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\stage4_ogm_progress.py`
- Modify: `D:\Github\HOPE\tools\stage4\tests\test_stage4_progress_gates.py`

- [ ] **Step 1: Add failing tests for the user-requested gate policy**

Append to `D:\Github\HOPE\tools\stage4\tests\test_stage4_progress_gates.py`:

```python
from tools.stage4.stage4_ogm_progress import decide_progress_gate, recommend_recovery_action


class Stage4ProgressGateTests(unittest.TestCase):
    def test_20k_gate_allows_weaker_than_hope_baseline_when_trend_improves(self) -> None:
        current = {
            "episode": 20000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": 0.08, "mean_psr_delta": 0.05, "step_num_improvement": 0.08},
            "summary": {"Sim-Normal": {"psr": 0.70, "angs": 3.0, "pl": 35.0}},
        }
        decision = decide_progress_gate(current=current, previous=None, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "continue")
        self.assertIn("early_ogm_can_lag_hope_20k_baseline", decision["notes"])

    def test_first_stagnant_gate_gets_one_10k_grace_window(self) -> None:
        current = {
            "episode": 30000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.01, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}},
        }
        previous = {"episode": 20000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=False)
        self.assertEqual(decision["decision"], "grace-10k")

    def test_second_consecutive_stagnant_gate_stops_before_100k(self) -> None:
        current = {
            "episode": 40000,
            "has_nonfinite": False,
            "checkpoint_exists": True,
            "trend": {"avg_reward_delta": -0.02, "mean_psr_delta": 0.0, "step_num_improvement": 0.0},
            "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.2, "pl": 46.0}},
        }
        previous = {"episode": 30000, "summary": {"Sim-Normal": {"psr": 0.60, "angs": 4.0, "pl": 45.0}}}
        decision = decide_progress_gate(current=current, previous=previous, previous_bad_gate=True)
        self.assertEqual(decision["decision"], "stop")
        self.assertIn("second_consecutive_bad_progress_gate", decision["reasons"])

    def test_recovery_restarts_when_observation_or_rasterizer_changes(self) -> None:
        self.assertEqual(recommend_recovery_action(["ogm_rasterizer_changed"]), "restart-from-scratch")
        self.assertEqual(recommend_recovery_action(["reward_logging_only"]), "resume-from-checkpoint")
```

- [ ] **Step 2: Run the progress-gate tests to verify they fail**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_progress_gates -v
```

Expected: FAIL with missing `tools.stage4.stage4_ogm_progress`.

- [ ] **Step 3: Implement progress gate logic**

Create `D:\Github\HOPE\tools\stage4\stage4_ogm_progress.py` with:

```python
from __future__ import annotations

from typing import Any, Mapping


def _trend_is_improving(trend: Mapping[str, float]) -> bool:
    positive = 0
    if float(trend.get("avg_reward_delta", 0.0)) > 0.02:
        positive += 1
    if float(trend.get("mean_psr_delta", 0.0)) > 0.02:
        positive += 1
    if float(trend.get("step_num_improvement", 0.0)) > 0.05:
        positive += 1
    return positive >= 2


def _trend_is_bad(trend: Mapping[str, float]) -> bool:
    return not _trend_is_improving(trend)


def decide_progress_gate(
    current: Mapping[str, Any],
    previous: Mapping[str, Any] | None,
    previous_bad_gate: bool,
) -> dict[str, Any]:
    reasons: list[str] = []
    notes: list[str] = []
    episode = int(current.get("episode", 0))
    if bool(current.get("has_nonfinite", False)):
        reasons.append("nonfinite_tensorboard_or_eval_metric")
    if not bool(current.get("checkpoint_exists", False)):
        reasons.append("missing_checkpoint")
    if reasons:
        return {"decision": "stop", "reasons": reasons, "notes": notes, "episode": episode}

    trend = current.get("trend", {})
    if episode == 20_000:
        notes.append("early_ogm_can_lag_hope_20k_baseline")
        if _trend_is_improving(trend):
            return {"decision": "continue", "reasons": [], "notes": notes, "episode": episode}
        return {"decision": "grace-10k", "reasons": ["weak_20k_trend"], "notes": notes, "episode": episode}

    if _trend_is_bad(trend):
        if previous_bad_gate:
            reasons.append("second_consecutive_bad_progress_gate")
            return {"decision": "stop", "reasons": reasons, "notes": notes, "episode": episode}
        return {"decision": "grace-10k", "reasons": ["first_bad_progress_gate"], "notes": notes, "episode": episode}

    return {"decision": "continue", "reasons": [], "notes": notes, "episode": episode}


def recommend_recovery_action(change_reasons: list[str]) -> str:
    restart_reasons = {
        "ogm_rasterizer_changed",
        "observation_semantics_changed",
        "reward_semantics_changed",
        "network_shape_changed",
        "replay_buffer_contaminated",
        "state_norm_semantics_changed",
    }
    if any(reason in restart_reasons for reason in change_reasons):
        return "restart-from-scratch"
    return "resume-from-checkpoint"
```

- [ ] **Step 4: Run progress-gate tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_progress_gates -v
```

Expected: PASS.

- [ ] **Step 5: Record the validation policy in findings**

Append to `D:\Github\HOPE\findings.md`:

```markdown
- Stage 4 OGM validation policy: use 20K episodes as the first gate against `hope-fast-action-mask-20k`. The OGM run may be weaker than the HOPE 20K baseline early, but TensorBoard/eval trends must improve. After 20K, monitor every 10K episodes. If a gate shows stagnation, regression, or clearly unreachable trajectory, allow exactly one more 10K grace window; if the next gate does not improve, stop before forcing 100K/120K and debug using TensorBoard, eval, and implementation evidence. Restart from scratch when observation/rasterizer/reward/network/state-norm semantics changed; resume from checkpoint only for logging, monitoring, or evaluation-only fixes.
```

- [ ] **Step 6: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/stage4_ogm_progress.py tools/stage4/tests/test_stage4_progress_gates.py findings.md
git commit -m "Add Stage 4 OGM progress gates"
```

---

### Task 8: Add Opt-In OGM Training Runner

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py`
- Create: `D:\Github\HOPE\tools\stage4\tests\test_stage4_runner_config.py`

- [ ] **Step 1: Write failing runner-config tests**

Create `D:\Github\HOPE\tools\stage4\tests\test_stage4_runner_config.py` with:

```python
from __future__ import annotations

import os
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

from env.car_parking_base import CarParking  # noqa: E402
from env.env_wrapper import CarParkingWrapper  # noqa: E402
from tools.stage4.train_HOPE_sac_ogm import build_ogm_training_config  # noqa: E402


class Stage4RunnerConfigTests(unittest.TestCase):
    def test_build_ogm_training_config_uses_only_target_action_mask_and_ogm(self) -> None:
        raw_env = CarParking(
            render_mode="rgb_array",
            verbose=False,
            use_img_observation=False,
            use_lidar_observation=True,
            use_action_mask=True,
            use_ogm_observation=True,
        )
        env = CarParkingWrapper(raw_env)
        try:
            config = build_ogm_training_config(env)
            self.assertEqual(set(config["observation_shape"].keys()), {"target", "action_mask", "ogm"})
            self.assertIsNone(config["actor_layers"]["lidar_shape"])
            self.assertIsNone(config["actor_layers"]["img_shape"])
            self.assertEqual(config["actor_layers"]["ogm_shape"], (2, 64, 64))
            self.assertEqual(config["actor_layers"]["n_modal"], 3)
            self.assertEqual(config["critic_layers"]["n_modal"], 4)
        finally:
            raw_env.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run the test to verify it fails**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_runner_config -v
```

Expected: FAIL with missing `tools.stage4.train_HOPE_sac_ogm`.

- [ ] **Step 3: Implement OGM runner config helpers**

Create `D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py` by copying the training loop from `src/train/train_HOPE_sac.py` and making these exact OGM-specific changes:

```python
from copy import deepcopy


def build_ogm_training_config(env):
    actor_params = deepcopy(ACTOR_CONFIGS)
    critic_params = deepcopy(CRITIC_CONFIGS)
    actor_params.update(
        {
            "n_modal": 3,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
            "action_mask_shape": env.observation_shape["action_mask"][0],
        }
    )
    critic_params.update(
        {
            "n_modal": 4,
            "lidar_shape": None,
            "img_shape": None,
            "ogm_shape": env.observation_shape["ogm"],
            "action_mask_shape": env.observation_shape["action_mask"][0],
            "input_action_dim": env.action_space.shape[0],
        }
    )
    return {
        "discrete": False,
        "observation_shape": {
            "target": env.observation_shape["target"],
            "action_mask": env.observation_shape["action_mask"],
            "ogm": env.observation_shape["ogm"],
        },
        "action_dim": env.action_space.shape[0],
        "hidden_size": 64,
        "activation": "tanh",
        "dist_type": "gaussian",
        "save_params": False,
        "actor_layers": actor_params,
        "critic_layers": critic_params,
    }
```

In the runner's environment construction, use:

```python
raw_env = CarParking(
    fps=100,
    verbose=verbose,
    render_mode=None if args.visualize else "rgb_array",
    use_img_observation=False,
    use_lidar_observation=True,
    use_action_mask=True,
    use_ogm_observation=True,
)
env = CarParkingWrapper(raw_env)
```

Keep the original HOPE loop semantics:

- `SceneChoose`
- `DlpCaseChoose`
- random action warmup while memory fills
- SAC update cadence `total_step_num % 10 == 0`
- checkpoint cadence every 2000 episodes
- `SAC_best.pt`
- TensorBoard tags already used by HOPE
- action mask, RS planner, reward, dynamics, terminal status

Add a startup print:

```python
print("Stage 4 OGM runner policy inputs: target + action_mask + ogm")
```

Do not load `autoencoder.pt` because RGB BEV is disabled for the OGM policy.

- [ ] **Step 4: Run runner-config test**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_stage4_runner_config -v
```

Expected: PASS.

- [ ] **Step 5: Run all Stage 4 unit tests**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v
```

Expected: PASS.

- [ ] **Step 6: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/train_HOPE_sac_ogm.py tools/stage4/tests/test_stage4_runner_config.py
git commit -m "Add opt-in OGM SAC training runner"
```

---

### Task 9: Add Stage 4 Launch And Monitor Wrappers

**Files:**
- Create: `D:\Github\HOPE\tools\stage4\launch_stage4_ogm.ps1`
- Create: `D:\Github\HOPE\tools\stage4\monitor_stage4_ogm_progress.ps1`
- Modify: `D:\Github\HOPE\progress.md`

- [ ] **Step 1: Create launcher script**

Create `D:\Github\HOPE\tools\stage4\launch_stage4_ogm.ps1` with this behavior:

```powershell
param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [int]$TrainEpisode = 100000,
    [int]$EvalEpisode = 200,
    [string]$ChangedKnobsJson = '{"policy_inputs":"target+action_mask+ogm","rgb_bev_policy":false,"internal_lidar_for_action_mask":true}'
)

$ErrorActionPreference = 'Stop'

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$SrcDir = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'src')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$TrainScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage4\train_HOPE_sac_ogm.py')).Path
$MonitorScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\monitor_stage3_resources.ps1')).Path
$ExpDir = Join-Path -Path $SrcDir -ChildPath 'log\exp'
[System.IO.Directory]::CreateDirectory($ExpDir) | Out-Null

$env:SDL_VIDEODRIVER = 'dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$safeRunName = $RunName -replace '[^A-Za-z0-9_.-]', '_'
$prefix = "${safeRunName}_${stamp}"
$stdout = Join-Path -Path $ExpDir -ChildPath "$prefix.stdout.log"
$stderr = Join-Path -Path $ExpDir -ChildPath "$prefix.stderr.log"
$resourceCsv = Join-Path -Path $ExpDir -ChildPath "$prefix.resources.csv"
$manifest = Join-Path -Path $ExpDir -ChildPath "$prefix.meta.json"

$args = @(
    $TrainScript,
    '--train_episode', "$TrainEpisode",
    '--eval_episode', "$EvalEpisode",
    '--visualize=',
    '--verbose='
)

$workload = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $SrcDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

$monitor = Start-Process -FilePath powershell -ArgumentList @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $MonitorScript,
    '-ProcessId', "$($workload.Id)",
    '-OutputCsv', $resourceCsv
) -WindowStyle Hidden -PassThru

$meta = [ordered]@{
    schema_version = 1
    run_name = $RunName
    candidate_type = 'stage4_ogm_proxy'
    train_episode = $TrainEpisode
    eval_episode = $EvalEpisode
    workload_pid = $workload.Id
    monitor_pid = $monitor.Id
    stdout_path = $stdout
    stderr_path = $stderr
    resource_csv_path = $resourceCsv
    manifest_path = $manifest
    changed_knobs = ($ChangedKnobsJson | ConvertFrom-Json)
    first_gate_episode = 20000
    progress_gate_interval = 10000
    max_train_episode = 120000
    created_at = (Get-Date).ToUniversalTime().ToString('o')
}
$meta | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Output $manifest
```

- [ ] **Step 2: Create progress monitor script**

Create `D:\Github\HOPE\tools\stage4\monitor_stage4_ogm_progress.ps1` with:

```powershell
param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)][int]$GateEpisode
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$SummaryScript = Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\tensorboard_stage3_summary.py'
$GateJson = Join-Path -Path $RepoRoot -ChildPath "docs\research\stage4_ogm_gate_${GateEpisode}.tensorboard.json"
$GateMd = Join-Path -Path $RepoRoot -ChildPath "docs\research\stage4_ogm_gate_${GateEpisode}.tensorboard.md"

& $Python $SummaryScript $RunDir --min-episodes $GateEpisode --json $GateJson --markdown $GateMd

$checkpointEpisode = $GateEpisode - 1
$checkpoint = Join-Path -Path $RunDir -ChildPath "SAC_${checkpointEpisode}.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Expected checkpoint missing at gate ${GateEpisode}: $checkpoint"
}

Write-Output "TensorBoard summary: $GateJson"
Write-Output "Checkpoint: $checkpoint"
```

This script only exports and checks gate artifacts. The execution agent must combine it with `eval_stage4_ogm_checkpoint.py` and `stage4_ogm_progress.py` to decide continue, grace, or stop.

- [ ] **Step 3: Run PowerShell parser checks**

```powershell
cd D:\Github\HOPE
$null = [scriptblock]::Create((Get-Content -LiteralPath .\tools\stage4\launch_stage4_ogm.ps1 -Raw))
$null = [scriptblock]::Create((Get-Content -LiteralPath .\tools\stage4\monitor_stage4_ogm_progress.ps1 -Raw))
```

Expected: no parser error.

- [ ] **Step 4: Commit**

```powershell
cd D:\Github\HOPE
git add tools/stage4/launch_stage4_ogm.ps1 tools/stage4/monitor_stage4_ogm_progress.ps1 progress.md
git commit -m "Add Stage 4 OGM launch wrappers"
```

---

### Task 10: OGM Plumbing Smoke Verification

**Files:**
- Modify: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-smoke-report.md`

- [ ] **Step 1: Run full unit test suite for Stage 4 and Stage 3**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v
```

Expected: both commands PASS.

- [ ] **Step 2: Run a tiny OGM training smoke**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
D:\Github\HOPE\.venv\Scripts\python.exe D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py --train_episode 20 --eval_episode 1 --visualize= --verbose=
```

Expected:

- A new `D:\Github\HOPE\src\log\exp\sac_*` directory appears.
- TensorBoard event file exists.
- No non-finite actor/critic loss is printed.
- No generated checkpoint is expected at 20 episodes.

- [ ] **Step 3: Write smoke report**

Create `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-smoke-report.md` with:

```markdown
# Stage 4 OGM Proxy Smoke Report

Date: 2026-06-20

## Commands

- Unit tests: `.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v`
- Stage 3 regression tests: `.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v`
- Tiny OGM training smoke: `D:\Github\HOPE\.venv\Scripts\python.exe D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py --train_episode 20 --eval_episode 1 --visualize= --verbose=`

## Result

- Stage 4 unit tests: record the exact `unittest` pass/fail summary.
- Stage 3 regression tests: record the exact `unittest` pass/fail summary.
- OGM smoke run directory: record the exact `D:\Github\HOPE\src\log\exp\sac_*` path created by the smoke command.
- Notes: record any warnings emitted by Pygame, Gym, TensorBoard, or PyTorch.

## Boundaries

This smoke validates OGM plumbing only. It is not a 20K quality gate and not an OGM-paper reproduction result.
```

- [ ] **Step 4: Commit**

```powershell
cd D:\Github\HOPE
git add docs/research/2026-06-20-stage4-ogm-proxy-smoke-report.md
git commit -m "Record Stage 4 OGM smoke verification"
```

---

### Task 11: 1K Diagnostic Run

**Files:**
- Modify: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-1k-diagnostic.md`

- [ ] **Step 1: Launch 1K diagnostic**

```powershell
cd D:\Github\HOPE
.\tools\stage4\launch_stage4_ogm.ps1 -RunName stage4_ogm_proxy_1k -TrainEpisode 1000 -EvalEpisode 10
```

Expected:

- Manifest path is printed.
- Workload and resource monitor PIDs are present in the manifest.
- A new `sac_*` run directory appears under `D:\Github\HOPE\src\log\exp`.

- [ ] **Step 2: After 1K finishes, export TensorBoard summary**

```powershell
cd D:\Github\HOPE
$Manifest = Get-ChildItem -LiteralPath .\src\log\exp -Filter 'stage4_ogm_proxy_1k_*.meta.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Meta = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
$RunDir = $Meta.run_dir
.\.venv\Scripts\python.exe .\tools\stage3\tensorboard_stage3_summary.py $RunDir --min-episodes 1000 --json .\docs\research\stage4_ogm_proxy_1k.tensorboard.json --markdown .\docs\research\stage4_ogm_proxy_1k.tensorboard.md
```

Expected:

- `training_budget_met=true`
- `hard_reject_has_nonfinite=false`
- Scalar summaries include `avg_reward`, `total_reward`, `actor_loss`, `critic_loss`, `alpha`, `step_num`.

- [ ] **Step 3: Write 1K diagnostic report**

Create `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-1k-diagnostic.md` with:

```markdown
# Stage 4 OGM Proxy 1K Diagnostic

Date: 2026-06-20

## Run

- Manifest: record `$Manifest.FullName`.
- Run directory: record `$RunDir`.
- Checkpoint: record `none expected at 1K`.
- Resource CSV: record `$Meta.resource_csv_path`.

## TensorBoard

- Training budget met: record the value from `stage4_ogm_proxy_1k.tensorboard.json`.
- Hard-reject nonfinite: record the value from `stage4_ogm_proxy_1k.tensorboard.json`.
- Avg reward trend: record the latest and recent mean from `avg_reward`.
- Step-num trend: record the latest and recent mean from `step_num`.
- Actor loss latest: record the latest finite actor loss if present.
- Critic loss latest: record the latest finite critic loss if present.

## Resource Summary

- Sample count: record the resource CSV sample count.
- Avg process CPU: record the process CPU average.
- Avg whole-GPU utilization: record the whole-GPU utilization average.
- Peak GPU memory: record the peak whole-GPU memory.

## Decision

Proceed to 20K gate only if `hard_reject_has_nonfinite=false`, the run completed without process failure, and OGM observations remained finite.
```

- [ ] **Step 4: Commit**

```powershell
cd D:\Github\HOPE
git add docs/research/2026-06-20-stage4-ogm-proxy-1k-diagnostic.md docs/research/stage4_ogm_proxy_1k.tensorboard.json docs/research/stage4_ogm_proxy_1k.tensorboard.md
git commit -m "Record Stage 4 OGM 1K diagnostic"
```

---

### Task 12: 20K First Gate Against Fast Action-Mask Baseline

**Files:**
- Modify: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-20k-gate.md`
- Generated local artifacts under `D:\Github\HOPE\docs\research\stage4_ogm_proxy_20k.*`

- [ ] **Step 1: Launch 20K OGM run**

```powershell
cd D:\Github\HOPE
.\tools\stage4\launch_stage4_ogm.ps1 -RunName stage4_ogm_proxy_20k -TrainEpisode 20000 -EvalEpisode 70
```

Expected: manifest printed and new run directory created.

- [ ] **Step 2: At 20K, export TensorBoard gate summary**

```powershell
cd D:\Github\HOPE
$Manifest = Get-ChildItem -LiteralPath .\src\log\exp -Filter 'stage4_ogm_proxy_20k_*.meta.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Meta = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
$RunDir = $Meta.run_dir
.\tools\stage4\monitor_stage4_ogm_progress.ps1 -ManifestPath $Manifest.FullName -RunDir $RunDir -GateEpisode 20000
```

Expected:

- `docs\research\stage4_ogm_gate_20000.tensorboard.json`
- `docs\research\stage4_ogm_gate_20000.tensorboard.md`
- `SAC_19999.pt` exists.

- [ ] **Step 3: Evaluate `SAC_19999.pt` on fixed OGM cases**

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
$Manifest = Get-ChildItem -LiteralPath .\src\log\exp -Filter 'stage4_ogm_proxy_20k_*.meta.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Meta = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
$RunDir = $Meta.run_dir
.\.venv\Scripts\python.exe .\tools\stage4\eval_stage4_ogm_checkpoint.py --checkpoint (Join-Path -Path $RunDir -ChildPath 'SAC_19999.pt') --cases .\docs\research\stage4_ogm_fixed_eval_cases_20260620.json --output-json .\docs\research\stage4_ogm_proxy_20k.eval.json
```

Expected: JSON contains `records` and `summary` with `Sim-Normal` and `Sim-Complex`.

- [ ] **Step 4: Apply the 20K gate policy**

Use `tools.stage4.stage4_ogm_progress.decide_progress_gate()` from this Python probe:

```powershell
cd D:\Github\HOPE
@'
import json
from pathlib import Path
from tools.stage4.stage4_ogm_progress import decide_progress_gate

tb = json.loads(Path("docs/research/stage4_ogm_gate_20000.tensorboard.json").read_text(encoding="utf-8"))
ev = json.loads(Path("docs/research/stage4_ogm_proxy_20k.eval.json").read_text(encoding="utf-8"))
avg_reward = tb["scalars"].get("avg_reward", {})
step_num = tb["scalars"].get("step_num", {})
summary = ev["summary"]
mean_psr = sum(split["psr"] for split in summary.values()) / max(len(summary), 1)
current = {
    "episode": 20000,
    "has_nonfinite": bool(tb.get("hard_reject_has_nonfinite", False)),
    "checkpoint_exists": True,
    "trend": {
        "avg_reward_delta": float(avg_reward.get("trend", {}).get("delta", float(avg_reward.get("mean_last100", 0.0)) - float(avg_reward.get("mean_first500", 0.0)))),
        "mean_psr_delta": mean_psr,
        "step_num_improvement": max(0.0, (float(step_num.get("mean_first500", 0.0)) - float(step_num.get("mean_last100", 0.0))) / max(float(step_num.get("mean_first500", 1.0)), 1.0)),
    },
    "summary": summary,
}
print(json.dumps(decide_progress_gate(current=current, previous=None, previous_bad_gate=False), indent=2, ensure_ascii=False))
'@ | .\.venv\Scripts\python.exe -
```

Decision interpretation:

- `continue`: proceed to 30K.
- `grace-10k`: proceed exactly one more 10K and re-check at 30K.
- `stop`: stop immediately and debug.

The OGM 20K checkpoint may be weaker than `hope-fast-action-mask-20k` on HOPE Normal/Complex/Extrem/DLP because OGM is a new representation. The gate still compares speed/resources and training health to `hope-fast-action-mask-20k`, but quality acceptance at 20K is based on improving trend and absence of hard reject conditions, not matching HOPE 20K absolute success.

- [ ] **Step 5: Write the 20K gate report**

Create `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-20k-gate.md` with:

```markdown
# Stage 4 OGM Proxy 20K Gate

Date: 2026-06-20

## Baseline Reference

- Baseline: `hope-fast-action-mask-20k`
- Baseline checkpoint: `src/log/exp/sac_20260620_085208/SAC_19999.pt`
- Baseline throughput: `1826.840564 episodes/hour`, `46.223481 env steps/second`
- Baseline external eval: Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`

## OGM Run

- Manifest: record `$Manifest.FullName`.
- Run directory: record `$RunDir`.
- Checkpoint: record the exact `SAC_19999.pt` path.
- Wall time: record computed manifest/TensorBoard wall time.
- Episodes/hour: record computed episodes/hour.
- Env steps/second: record computed environment steps/second.

## OGM Fixed-Set Evaluation

- Sim-Normal PSR: record `summary["Sim-Normal"]["psr"]`.
- Sim-Normal ANGS: record `summary["Sim-Normal"]["angs"]`.
- Sim-Normal PL: record `summary["Sim-Normal"]["pl"]`.
- Sim-Complex PSR: record `summary["Sim-Complex"]["psr"]`.
- Sim-Complex ANGS: record `summary["Sim-Complex"]["angs"]`.
- Sim-Complex PL: record `summary["Sim-Complex"]["pl"]`.

## TensorBoard Trend

- Avg reward trend: record the computed recent-minus-early delta.
- Mean PSR trend: record the computed mean fixed-set PSR value used as the first trend anchor.
- Step-num improvement: record the computed positive fraction.
- Nonfinite hard reject: record the TensorBoard hard-reject flag.

## Decision

- Gate decision: record `continue`, `grace-10k`, or `stop`.
- Reason: record the exact reasons from `decide_progress_gate()`.
- Next action: record the next run action implied by the decision.
```

- [ ] **Step 6: Commit**

```powershell
cd D:\Github\HOPE
git add docs/research/2026-06-20-stage4-ogm-proxy-20k-gate.md docs/research/stage4_ogm_gate_20000.tensorboard.json docs/research/stage4_ogm_gate_20000.tensorboard.md docs/research/stage4_ogm_proxy_20k.eval.json
git commit -m "Record Stage 4 OGM 20K gate"
```

---

### Task 13: Full-State Stage 4 Continuation Before Further Segmented KPI Gates

**Context update, 2026-06-22:** User inspection of TensorBoard DLP curves showed that each 10K segment has a similar local shape. Code inspection confirmed the current continuation is model-checkpoint based: it restores SAC model/optimizer/state-normalization checkpoint data and global episode numbering, but resets replay memory, RNG state, `SceneChoose`, `DlpCaseChoose`, `total_step_num`, and local training histories. This is useful for diagnostic gates, but it is not equivalent to one continuous long run.

**Files:**
- Modify: `D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py`
- Modify: `D:\Github\HOPE\tools\stage4\launch_stage4_ogm.ps1`
- Modify/Create tests under `D:\Github\HOPE\tools\stage4\tests\`
- Modify: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-long-run.md`
- Modify: `D:\Github\HOPE\task_plan.md`
- Modify: `D:\Github\HOPE\findings.md`
- Modify: `D:\Github\HOPE\progress.md`

- [x] **Step 1: Write failing tests for full-state serialization**

Add tests proving Stage 4 can serialize and restore:

- SAC checkpoint path or embedded SAC state.
- Replay memory contents or an explicitly versioned replay snapshot.
- `total_step_num`.
- `SceneChoose.scene_record` and `SceneChoose.success_record`.
- `DlpCaseChoose.case_record` and `DlpCaseChoose.case_success_rate`.
- NumPy RNG state.
- Torch RNG state and CUDA RNG state when CUDA is available.
- `best_success_rate`.
- Local histories needed for TensorBoard/report continuity.

Expected red state: current runner has no full-state checkpoint mechanism.

- [x] **Step 2: Implement explicit full-state checkpoint helpers**

Add Stage4-only helpers rather than changing original `src/train` defaults. Prefer a versioned artifact name such as `stage4_state_<episode>.pt` or `stage4_state_latest.pt` next to the existing `SAC_<episode>.pt`.

The full-state artifact must be explicit and reviewable. It should not silently alter original HOPE SAC checkpoints or author checkpoints.

- [x] **Step 3: Add full-state resume CLI and launcher support**

Add explicit flags, for example:

```powershell
.\tools\stage4\launch_stage4_ogm.ps1 -RunName stage4_ogm_proxy_60k_fullstate -TrainEpisode 60000 -StartEpisode 50000 -ResumeState D:\Github\HOPE\src\log\exp\...\stage4_state_49999.pt
```

Validation rules:

- `-ResumeState` implies `StartEpisode > 0`.
- `-ResumeState` and model-only `-ResumeCheckpoint` must not be confused in manifests.
- Manifest must record `continuation_type="full_state"` for full-state runs.
- If only `-ResumeCheckpoint` is used, manifest must keep `continuation_type="checkpoint_based"` and reports must caveat it as diagnostic.

- [x] **Step 4: Verify resumed state equivalence with a bounded deterministic test**

Run a short bounded training path and compare:

- Single uninterrupted path from episode N to N+M.
- Interrupted path saved at N and resumed from full-state artifact to N+M.

Use deterministic seeds and a small bounded budget. The verification does not need to prove floating-point bit identity across every operation if CUDA nondeterminism prevents it, but it must prove the restored counters, chooser histories, replay length, RNG progression, checkpoint naming, and TensorBoard step continuity behave as intended.

Result, 2026-06-22: `tools/stage4/verify_stage4_full_state_resume.py` generated `docs/research/2026-06-22-stage4-full-state-resume-verification.json` and `.md`. The bounded deterministic runner-state simulation passed: direct 8-episode path and interrupted/resumed path matched restored counters, replay length, chooser histories, RNG progression including CUDA, checkpoint naming, and TensorBoard global steps. This does not prove bitwise real SAC learning-trajectory equivalence.

- [x] **Step 5: Decide current-run recovery policy**

After the current 50K diagnostic gate:

- If no Stage 4 semantic code changed and full-state continuation is ready, prefer launching the next gate from the latest full-state artifact.
- If only model checkpoints exist for the old segments, either label the next segment diagnostic or restart a clean uninterrupted/full-state-tracked run from scratch.
- Do not claim the 30K/40K/50K model-only ladder is equivalent to a continuous 50K learning process.

Decision, 2026-06-22: the old 50K checkpoint-based segment completed and is diagnostic only. It produced `SAC_49999.pt`, but no trustworthy full-state artifact spanning the prior 0K-50K process exists because the segment was launched before Task 13 full-state support. Its fixed eval was Sim-Normal PSR `1.000`, ANGS `3.200`, PL `14.431`; Sim-Complex PSR `0.900`, ANGS `39.667`, PL `45.010`. The old model-only gate policy would return `grace-10k`, and the full-state boundary now supersedes automatic checkpoint-only continuation. Do not launch a 60K model-only segment from `SAC_49999.pt`; the next KPI-reproduction run should be clean/full-state-tracked or explicitly full-state-resumed from a matching `stage4_state_<episode>.pt`.

- [x] **Step 6: Update docs and progress state**

Update the long-run report and root planning files with the full-state continuation decision, tests, and any restart/resume choice.

---

### Task 14: Long-Run 10K Gate Monitoring To 80K-120K

**Files:**
- Modify: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-long-run.md`

- [ ] **Step 1: Continue or relaunch the OGM long run according to 20K decision**

If the 20K decision is `continue`, use the existing process if it is still running. If it stopped at exactly 20K, relaunch with full-state continuation when a matching `stage4_state_<episode>.pt` exists. Model-only checkpoint-based continuation may be used only as a diagnostic bridge and must preserve global episode numbering with `-StartEpisode 20000`; it does not restore replay buffer, RNG, curriculum/scenario chooser state, DLP chooser state, reward history, or `total_step_num` from the checkpoint:

```powershell
cd D:\Github\HOPE
$State20K = 'D:\Github\HOPE\src\log\exp\sac_ogm_stage4_ogm_proxy_fullstate_20k_...\stage4_state_19999.pt'
.\tools\stage4\launch_stage4_ogm.ps1 -RunName stage4_ogm_proxy_long_fullstate -TrainEpisode 30000 -EvalEpisode 70 -ResumeState $State20K -StartEpisode 20000 -ChangedKnobsJson '{"resume_from":"stage4_state_19999.pt","start_episode":20000,"policy_inputs":"target+action_mask+ogm","continuation_type":"full_state"}'
```

Current update, 2026-06-22: no matching full-state artifact exists for the old 20K/30K/40K/50K checkpoint-only ladder. Do not use the old `SAC_49999.pt` as a KPI-reproduction continuation source. The next production-quality long run should either restart cleanly from episode 0 with full-state snapshots enabled, or resume only from a `stage4_state_<episode>.pt` produced by that clean/full-state-tracked run.

If OGM rasterizer, observation semantics, reward semantics, network shape, or state normalization changed after the 20K gate, restart from scratch:

```powershell
cd D:\Github\HOPE
.\tools\stage4\launch_stage4_ogm.ps1 -RunName stage4_ogm_proxy_long_restart -TrainEpisode 100000 -EvalEpisode 70 -ChangedKnobsJson '{"restart_reason":"semantic_fix","policy_inputs":"target+action_mask+ogm"}'
```

- [ ] **Step 2: At every 10K node, export TensorBoard and checkpoint data**

Run at `30000`, `40000`, `50000`, `60000`, `70000`, `80000`, and then continue to `90000`, `100000`, `110000`, `120000` only if trend remains promising:

```powershell
cd D:\Github\HOPE
$Manifest = Get-ChildItem -LiteralPath .\src\log\exp -Filter 'stage4_ogm_proxy_long*.meta.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Meta = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
$RunDir = $Meta.run_dir
.\tools\stage4\monitor_stage4_ogm_progress.ps1 -ManifestPath $Manifest.FullName -RunDir $RunDir -GateEpisode 30000
```

Change `-GateEpisode` for each gate.

- [ ] **Step 3: Evaluate each gate checkpoint on fixed OGM cases**

For 30K:

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
$Manifest = Get-ChildItem -LiteralPath .\src\log\exp -Filter 'stage4_ogm_proxy_long*.meta.json' | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$Meta = Get-Content -LiteralPath $Manifest.FullName -Raw | ConvertFrom-Json
$RunDir = $Meta.run_dir
.\.venv\Scripts\python.exe .\tools\stage4\eval_stage4_ogm_checkpoint.py --checkpoint (Join-Path -Path $RunDir -ChildPath 'SAC_29999.pt') --cases .\docs\research\stage4_ogm_fixed_eval_cases_20260620.json --output-json .\docs\research\stage4_ogm_proxy_30k.eval.json
```

Use `SAC_39999.pt`, `SAC_49999.pt`, and so on for later gates.

- [ ] **Step 4: Apply the one-window grace rule**

For each gate:

1. Build the `current` object with episode, nonfinite flag, checkpoint flag, trend deltas, and fixed-set summary.
2. Use `previous_bad_gate=false` if the previous gate was `continue`.
3. Use `previous_bad_gate=true` if the previous gate was `grace-10k`.
4. If current decision is `grace-10k`, continue exactly one more 10K gate.
5. If the next decision is still `grace-10k` or `stop`, terminate the run and write a debug report instead of forcing the run to 100K or 120K.

Stop examples:

- 30K gate is stagnant -> continue to 40K.
- 40K gate is still stagnant -> stop and debug.
- 50K gate regresses badly -> continue to 60K only if no hard reject exists.
- 60K gate is still regressed -> stop and debug.

- [ ] **Step 5: Choose restart versus resume after a stopped gate**

Use this rule:

- Restart from scratch when fixes touch `src/env/ogm.py`, OGM observation shape/encoding, reward semantics, action semantics, network input shape, state-normalization semantics, or replay-buffer meaning.
- Resume from checkpoint only when fixes touch logging, monitor scripts, report formatting, eval parser bugs, or resource-monitor PID tracking.

- [ ] **Step 6: Maintain the long-run report**

Create or update `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-long-run.md` at each gate:

```markdown
# Stage 4 OGM Proxy Long-Run Gate Log

## Gate Table

| Gate | Checkpoint | Decision | Sim-Normal PSR | Sim-Normal ANGS | Sim-Normal PL | Sim-Complex PSR | Sim-Complex ANGS | Sim-Complex PL | Avg Reward Trend | Step Trend | Notes |
| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |

## Stop/Continue Decisions

- 20K: record decision and exact reason list.
- 30K: record decision and exact reason list.
- 40K: record decision and exact reason list.
- 50K: record decision and exact reason list.
- 60K: record decision and exact reason list.
- 70K: record decision and exact reason list.
- 80K: record decision and exact reason list.
- 90K: record decision and exact reason list.
- 100K: record decision and exact reason list.
- 110K: record decision and exact reason list.
- 120K: record decision and exact reason list.

## Debug Notes

- TensorBoard anomalies: record non-finite, flat reward, rising step count, or scalar gaps if observed.
- Eval anomalies: record PSR regression, ANGS growth, PL growth, or split collapse if observed.
- Resource anomalies: record monitor exits, missing samples, or process/GPU trend shifts if observed.
- Recovery action: record `restart-from-scratch` or `resume-from-checkpoint`.
```

- [ ] **Step 7: Commit after each gate**

At every completed gate:

```powershell
cd D:\Github\HOPE
$Episode = 30000
$Label = '30k'
git add docs/research/2026-06-20-stage4-ogm-proxy-long-run.md "docs/research/stage4_ogm_gate_${Episode}.tensorboard.json" "docs/research/stage4_ogm_gate_${Episode}.tensorboard.md" "docs/research/stage4_ogm_proxy_${Label}.eval.json"
git commit -m "Record Stage 4 OGM ${Label} gate"
```

For later gates, set `$Episode` to `40000`, `50000`, `60000`, `70000`, `80000`, `90000`, `100000`, `110000`, or `120000`, and set `$Label` to `40k`, `50k`, `60k`, `70k`, `80k`, `90k`, `100k`, `110k`, or `120k`.

---

### Task 15: Final OGM KPI Comparison And Documentation

**Files:**
- Create: `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-final-report.md`
- Modify: `D:\Github\HOPE\task_plan.md`
- Modify: `D:\Github\HOPE\findings.md`
- Modify: `D:\Github\HOPE\progress.md`

- [ ] **Step 1: Select the final checkpoint**

Select the first checkpoint between `SAC_79999.pt` and `SAC_119999.pt` that meets all hard gates:

- Training episode is between 80K and 120K.
- Fixed eval set has Sim-Normal and Sim-Complex metrics.
- PSR, ANGS, and PL are inside relative `+/-3%` of OGM paper targets for both splits.
- TensorBoard hard-reject nonfinite flag is false.
- No two consecutive bad progress gates were ignored.

- [ ] **Step 2: Write final report**

Create `D:\Github\HOPE\docs\research\2026-06-20-stage4-ogm-proxy-final-report.md` with:

```markdown
# Stage 4 OGM Proxy Final Report

Date: 2026-06-20

## Final Checkpoint

- Run directory: record the final selected run directory.
- Checkpoint: record the final selected checkpoint path.
- Training episodes: record the checkpoint episode plus one.
- Wall time: record the final measured wall time.
- Episodes/hour: record final episodes/hour.
- Env steps/second: record final environment steps/second.

## Hard KPI Comparison

| Split | Metric | OGM Paper Target | Accepted Band | Observed | Pass |
| --- | --- | ---: | ---: | ---: | --- |
| Sim-Normal | PSR | 0.9933 | 0.963501 - 1.023099 | record observed value | record pass/fail |
| Sim-Normal | ANGS | 1.5 | 1.455 - 1.545 | record observed value | record pass/fail |
| Sim-Normal | PL | 20.3 | 19.691 - 20.909 | record observed value | record pass/fail |
| Sim-Complex | PSR | 0.977 | 0.94769 - 1.00631 | record observed value | record pass/fail |
| Sim-Complex | ANGS | 1.9 | 1.843 - 1.957 | record observed value | record pass/fail |
| Sim-Complex | PL | 23.6 | 22.892 - 24.308 | record observed value | record pass/fail |

## Compatibility Metrics

- HOPE Normal: record the compatibility success rate if evaluated.
- HOPE Complex: record the compatibility success rate if evaluated.
- Speed/resource comparison to `hope-fast-action-mask-20k`: record wall-time, episodes/hour, env steps/second, process CPU, whole-GPU utilization, and GPU memory comparison.

## Dataset Boundary

No real-world OGM dataset was available in this repository. Real-world dataset and real-vehicle KPI rows are excluded from this acceptance.

## Decision

- Final decision: record `pass`, `investigate`, or `fail`.
- Reason: record the exact KPI and gate reasons.
- Next recommended experiment: record the next experiment name and why it follows from this result.
```

- [ ] **Step 3: Update planning files**

In `D:\Github\HOPE\task_plan.md`, mark Phase 12 complete only if the final report exists and the accepted gate decision is recorded.

In `D:\Github\HOPE\findings.md`, append the final KPI result and any failed-gate/restart decisions.

In `D:\Github\HOPE\progress.md`, append the final run summary, checkpoint path, and next recommendation.

- [ ] **Step 4: Final verification**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v
git diff --check
git diff -- src\train\train_HOPE_sac.py src\model\ckpt
```

Expected:

- Stage 4 tests PASS.
- Stage 3 tests PASS.
- `git diff --check` prints no errors.
- Protected diff check prints no source diff for original training script or checkpoints.

- [ ] **Step 5: Commit**

```powershell
cd D:\Github\HOPE
git add docs/research/2026-06-20-stage4-ogm-proxy-final-report.md task_plan.md findings.md progress.md
git commit -m "Record Stage 4 OGM final report"
```

---

## Self-Review Checklist For The Execution Agent

Before starting Task 1:

- [ ] Confirm the branch is `codex/stage3-resource-study` or a new `codex/` worktree branch.
- [ ] Confirm `origin` is `https://github.com/AstarteCN/HOPE.git`.
- [ ] Confirm `upstream` push URL is disabled or not used.
- [ ] Confirm no training artifacts, TensorBoard logs, generated checkpoints, `.venv`, or large data files are staged.

After each task:

- [ ] Run the task's explicit tests.
- [ ] Run `git status -sb`.
- [ ] Commit only intentional source, tool, test, doc, and planning files.

Before long training:

- [ ] Ensure all tests pass.
- [ ] Ensure OGM is default-off for normal HOPE paths.
- [ ] Ensure `src/train/train_HOPE_sac.py` is unchanged.
- [ ] Ensure `hope-fast-action-mask-20k` baseline details are present in the gate report.

## Execution Handoff

Recommended execution mode: Subagent-Driven. Dispatch a fresh subagent for Tasks 1-9, review each commit, then execute Tasks 10-15 with explicit user-visible checkpoints before 1K, 20K, full-state continuation rollout, and every 10K long-run gate.
