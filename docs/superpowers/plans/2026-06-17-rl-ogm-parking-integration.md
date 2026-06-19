# RL-OGM Parking Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add an incremental RL-OGM-Parking variant to this HOPE repository, train it locally, and compare it against HOPE-style baselines without breaking the existing HOPE code paths.

**Architecture:** Keep the existing Gym/Pygame environment, vehicle model, Reeds-Shepp hybrid planner, SAC agent, and action-mask machinery. Add an explicit OGM observation path, generated first from existing Shapely map geometry as a local proxy for LiDAR OGM, then wire it into a SAC training/evaluation variant with OGM-paper metrics.

**Tech Stack:** Python 3.8+, NumPy, Shapely, OpenCV, Gym, Pygame, PyTorch, TensorBoard, existing HOPE SAC/PPO code.

---

## Current Research Inputs

Read first:

- `docs/research/2026-06-17-rl-ogm-paper-code-research.md`
- `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`
- `raw_paper/2502.18846_RL-OGM-Parking_Lidar_OGM-Based_Hybrid_Reinforcement_Learning_Planner_for_Autonomous_Parking.pdf`

Do not implement against guesses from memory. The OGM paper does not specify every grid parameter, so the initial local implementation intentionally uses a 64x64 grid to match the existing HOPE BEV encoder.

## File Structure

Create:

- `tests/test_ogm.py`: unit tests for OGM rasterization and local ego-frame crops.
- `tests/test_env_ogm_observation.py`: environment smoke tests for the new `ogm` observation.
- `tests/test_eval_metrics.py`: metric tests for gear shifts and path length.
- `src/env/ogm.py`: OGM configuration and rasterization helpers.
- `src/evaluation/ogm_metrics.py`: PSR, ANGS, PL, and AOT-style metric helpers.
- `src/train/train_RL_OGM_sac.py`: SAC training entry point for OGM experiments.
- `src/evaluation/eval_ogm_mix_scene.py`: evaluation entry point that reports OGM-paper metrics.

Modify:

- `src/configs.py`: add OGM toggles and shapes while preserving existing defaults for HOPE.
- `src/env/car_parking_base.py`: add optional OGM observation generation.
- `src/env/env_wrapper.py`: transpose OGM tensors to CHW.
- `src/model/network.py`: add explicit `ogm` embedding path.
- `src/model/agent/sac_agent.py`: ensure `obs2tensor()` and observation slicing support `ogm` without special cases.
- `src/evaluation/eval_utils.py`: keep existing behavior; optionally call shared metric helpers from `ogm_metrics.py`.

Do not modify:

- `src/train/train_HOPE_sac.py` except for import-compatible refactors that are strictly required.
- `src/train/train_HOPE_ppo.py`.
- Existing checkpoints in `src/model/ckpt/`.

## Task 0: Baseline Sanity Check

**Files:**

- Read-only: `src/train/train_HOPE_sac.py`
- Read-only: `src/evaluation/eval_mix_scene.py`

- [ ] **Step 1: Verify imports from the existing environment**

Run:

```powershell
cd D:\Github\HOPE\src
python -c "from env.car_parking_base import CarParking; from env.env_wrapper import CarParkingWrapper; e=CarParking(render_mode='rgb_array', verbose=False); w=CarParkingWrapper(e); obs=w.reset(); print(sorted(obs.keys())); print({k: None if v is None else v.shape for k,v in obs.items()}); e.close()"
```

Expected: command exits 0 and prints keys including `action_mask`, `img`, `lidar`, and `target`.

- [ ] **Step 2: Run a short pretrained evaluation smoke test**

Run:

```powershell
cd D:\Github\HOPE\src
python .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 5 --visualize False
```

Expected: command exits 0 and prints an evaluation result. Do not compare paper metrics from only 5 episodes.

- [ ] **Step 3: Record baseline output**

Append the command, checkpoint, date, and printed success rate to a new experiment log:

```text
docs/research/ogm_experiment_log.md
```

Expected entry format:

```markdown
## 2026-06-17 HOPE_SAC0 smoke evaluation

Command: `python .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 5 --visualize False`

Result: paste the printed success rate and any warnings.
```

- [ ] **Step 4: Commit only the experiment log if it was created**

Run:

```powershell
git add docs/research/ogm_experiment_log.md
git commit -m "docs: record HOPE baseline smoke evaluation"
```

Expected: a docs-only commit.

## Task 1: Add OGM Rasterization Utility

**Files:**

- Create: `tests/test_ogm.py`
- Create: `src/env/ogm.py`

- [ ] **Step 1: Write failing unit tests**

Create `tests/test_ogm.py`:

```python
import unittest

import numpy as np
from shapely.geometry import LinearRing

from env.ogm import OGMConfig, rasterize_local_ogm
from env.vehicle import State


class OGMTests(unittest.TestCase):
    def test_rasterize_local_ogm_marks_obstacle_cells(self):
        cfg = OGMConfig(size=64, resolution=0.2)
        ego = State([0.0, 0.0, 0.0, 0.0, 0.0])
        obstacle = LinearRing([(1.0, -0.5), (1.5, -0.5), (1.5, 0.5), (1.0, 0.5)])

        ogm = rasterize_local_ogm(
            ego_state=ego,
            obstacles=[obstacle],
            target_box=None,
            vehicle_box=None,
            config=cfg,
        )

        self.assertEqual(ogm.shape, (3, 64, 64))
        self.assertGreater(float(ogm[0].sum()), 0.0)
        self.assertTrue(np.all((ogm >= 0.0) & (ogm <= 1.0)))

    def test_rasterize_local_ogm_marks_target_channel(self):
        cfg = OGMConfig(size=64, resolution=0.2)
        ego = State([0.0, 0.0, 0.0, 0.0, 0.0])
        target = LinearRing([(2.0, -1.0), (3.0, -1.0), (3.0, 1.0), (2.0, 1.0)])

        ogm = rasterize_local_ogm(
            ego_state=ego,
            obstacles=[],
            target_box=target,
            vehicle_box=None,
            config=cfg,
        )

        self.assertEqual(ogm.shape, (3, 64, 64))
        self.assertEqual(float(ogm[0].sum()), 0.0)
        self.assertGreater(float(ogm[1].sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_ogm -v
```

Expected: FAIL or ERROR because `env.ogm` does not exist.

- [ ] **Step 3: Implement the OGM utility**

Create `src/env/ogm.py`:

```python
from dataclasses import dataclass
from typing import Iterable, Optional

import cv2
import numpy as np
from shapely.affinity import affine_transform
from shapely.geometry import LinearRing


@dataclass(frozen=True)
class OGMConfig:
    size: int = 64
    resolution: float = 0.2
    channels: int = 3


def _shape_to_ring(shape):
    if hasattr(shape, "shape"):
        return shape.shape
    return shape


def _ego_transform_matrix(ego_state):
    x = ego_state.loc.x
    y = ego_state.loc.y
    theta = ego_state.heading
    c = np.cos(theta)
    s = np.sin(theta)
    return [c, s, -s, c, -x * c - y * s, x * s - y * c]


def _world_ring_to_grid_points(ring: LinearRing, ego_state, config: OGMConfig):
    ego_ring = affine_transform(ring, _ego_transform_matrix(ego_state))
    coords = np.asarray(ego_ring.coords[:-1], dtype=np.float32)
    half = config.size / 2.0
    gx = np.floor(coords[:, 0] / config.resolution + half).astype(np.int32)
    gy = np.floor(half - coords[:, 1] / config.resolution).astype(np.int32)
    points = np.stack([gx, gy], axis=1)
    return points.reshape((-1, 1, 2))


def _draw_rings(channel: np.ndarray, rings: Iterable[LinearRing], ego_state, config: OGMConfig):
    for raw_ring in rings:
        ring = _shape_to_ring(raw_ring)
        points = _world_ring_to_grid_points(ring, ego_state, config)
        cv2.fillPoly(channel, [points], 1.0)


def rasterize_local_ogm(
    ego_state,
    obstacles,
    target_box: Optional[LinearRing],
    vehicle_box: Optional[LinearRing],
    config: OGMConfig,
) -> np.ndarray:
    ogm = np.zeros((config.channels, config.size, config.size), dtype=np.float32)
    _draw_rings(ogm[0], obstacles, ego_state, config)
    if target_box is not None:
        _draw_rings(ogm[1], [target_box], ego_state, config)
    if vehicle_box is not None and config.channels > 2:
        _draw_rings(ogm[2], [vehicle_box], ego_state, config)
    return np.clip(ogm, 0.0, 1.0)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_ogm -v
```

Expected: both tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/env/ogm.py tests/test_ogm.py
git commit -m "feat: add local OGM rasterizer"
```

## Task 2: Add Optional OGM Observation To The Environment

**Files:**

- Create: `tests/test_env_ogm_observation.py`
- Modify: `src/configs.py`
- Modify: `src/env/car_parking_base.py`
- Modify: `src/env/env_wrapper.py`

- [ ] **Step 1: Write failing environment tests**

Create `tests/test_env_ogm_observation.py`:

```python
import unittest

from env.car_parking_base import CarParking
from env.env_wrapper import CarParkingWrapper


class EnvOGMObservationTests(unittest.TestCase):
    def test_unwrapped_env_can_return_ogm(self):
        env = CarParking(render_mode="rgb_array", verbose=False, use_ogm_observation=True)
        try:
            obs = env.reset(level="Normal")
            self.assertIn("ogm", obs)
            self.assertEqual(obs["ogm"].shape, (64, 64, 3))
        finally:
            env.close()

    def test_wrapper_transposes_ogm_to_chw(self):
        env = CarParking(render_mode="rgb_array", verbose=False, use_ogm_observation=True)
        wrapped = CarParkingWrapper(env)
        try:
            obs = wrapped.reset(level="Normal")
            self.assertIn("ogm", obs)
            self.assertEqual(obs["ogm"].shape, (3, 64, 64))
            self.assertEqual(wrapped.observation_shape["ogm"], (3, 64, 64))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_env_ogm_observation -v
```

Expected: FAIL or ERROR because `use_ogm_observation` is not accepted.

- [ ] **Step 3: Add OGM config values**

In `src/configs.py`, add these environment constants near the existing observation constants:

```python
USE_OGM = False
OGM_SIZE = 64
OGM_RESOLUTION = 0.2
OGM_CHANNELS = 3
```

Then add model config fields:

```python
'ogm_shape': (OGM_CHANNELS, OGM_SIZE, OGM_SIZE) if USE_OGM else None,
```

When counting modalities, update `n_modal` to include `int(USE_OGM)`.

- [ ] **Step 4: Add OGM observation in `CarParking`**

In `src/env/car_parking_base.py`:

- Import `OGMConfig` and `rasterize_local_ogm`.
- Add constructor argument `use_ogm_observation: bool = USE_OGM`.
- Store `self.use_ogm_observation`.
- Add an `observation_space['ogm']` with shape `(OGM_SIZE, OGM_SIZE, OGM_CHANNELS)`.
- Add `_get_ogm_observation()` that calls `rasterize_local_ogm()` and transposes CHW to HWC for unwrapped environment consistency.
- Include `ogm` in the observation dictionary returned by `render()`.

The method body should follow this shape:

```python
def _get_ogm_observation(self):
    ogm = rasterize_local_ogm(
        ego_state=self.vehicle.state,
        obstacles=[obs.shape for obs in self.map.obstacles],
        target_box=self.map.dest_box,
        vehicle_box=self.vehicle.box,
        config=self.ogm_config,
    )
    return ogm.transpose((1, 2, 0))
```

- [ ] **Step 5: Update wrapper transposition**

In `src/env/env_wrapper.py`, update `observation_rescale()`:

```python
if obs.get('ogm') is not None:
    obs['ogm'] = obs['ogm'].transpose((2, 0, 1))
```

In `CarParkingWrapper.__init__()`, apply the same HWC to CHW shape conversion for `ogm` that is already used for `img`.

- [ ] **Step 6: Run tests**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_ogm tests.test_env_ogm_observation -v
```

Expected: all tests pass.

- [ ] **Step 7: Commit**

Run:

```powershell
git add src/configs.py src/env/car_parking_base.py src/env/env_wrapper.py tests/test_env_ogm_observation.py
git commit -m "feat: expose optional OGM observations"
```

## Task 3: Add Explicit OGM Encoder Support

**Files:**

- Modify: `src/model/network.py`
- Modify: `src/configs.py`
- Modify: `src/model/agent/sac_agent.py` only if tensor conversion assumes a fixed key set.

- [ ] **Step 1: Add a smoke test for actor forward**

Create or extend `tests/test_env_ogm_observation.py` with:

```python
def test_actor_config_accepts_ogm_observation(self):
    import torch
    from configs import ACTOR_CONFIGS
    from model.network import MultiObsEmbedding

    configs = dict(ACTOR_CONFIGS)
    configs["ogm_shape"] = (3, 64, 64)
    configs["img_shape"] = None
    configs["n_modal"] = 3 + int(configs["action_mask_shape"] is not None)

    model = MultiObsEmbedding(configs)
    obs = {
        "lidar": torch.zeros((2, configs["lidar_shape"]), dtype=torch.float32),
        "target": torch.zeros((2, configs["target_shape"]), dtype=torch.float32),
        "action_mask": torch.ones((2, configs["action_mask_shape"]), dtype=torch.float32),
        "ogm": torch.zeros((2, 3, 64, 64), dtype=torch.float32),
    }

    out = model(obs)
    self.assertEqual(out.shape, (2, configs["output_size"]))
```

- [ ] **Step 2: Run the test and verify it fails**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_env_ogm_observation -v
```

Expected: FAIL because `MultiObsEmbedding` does not read `ogm_shape`.

- [ ] **Step 3: Implement OGM embedding**

In `src/model/network.py`, mirror the image encoder path:

```python
self.use_ogm = False if configs.get('ogm_shape') is None else True
```

Create:

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

In `forward()`:

```python
if self.use_ogm:
    feature_ogm, _ = self.embed_ogm(x['ogm'])
    feature_ogm = self.re_embed_ogm(feature_ogm)
    features.append(feature_ogm)
```

Also include OGM layers in `orthogonal_init()` with the same pattern as `re_embed_img`.

- [ ] **Step 4: Update configs without changing HOPE defaults**

Add `ogm_shape` to both actor and critic configs:

```python
'ogm_shape': (OGM_CHANNELS, OGM_SIZE, OGM_SIZE) if USE_OGM else None,
```

Keep `USE_OGM = False` by default so existing checkpoints and scripts are not invalidated.

- [ ] **Step 5: Run model and environment tests**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_ogm tests.test_env_ogm_observation -v
```

Expected: all tests pass.

- [ ] **Step 6: Commit**

Run:

```powershell
git add src/model/network.py src/configs.py tests/test_env_ogm_observation.py
git commit -m "feat: add OGM encoder modality"
```

## Task 4: Add OGM-Paper Evaluation Metrics

**Files:**

- Create: `tests/test_eval_metrics.py`
- Create: `src/evaluation/ogm_metrics.py`
- Modify: `src/evaluation/eval_ogm_mix_scene.py` in Task 6.

- [ ] **Step 1: Write failing metric tests**

Create `tests/test_eval_metrics.py`:

```python
import unittest

from evaluation.ogm_metrics import count_gear_shifts, path_length


class OGMMetricTests(unittest.TestCase):
    def test_count_gear_shifts_ignores_zero_and_counts_sign_changes(self):
        self.assertEqual(count_gear_shifts([1.0, 1.0, -1.0, -0.5, 0.5]), 2)
        self.assertEqual(count_gear_shifts([1.0, 0.0, 1.0, -1.0]), 1)

    def test_path_length_sums_xy_segments(self):
        states = [(0.0, 0.0), (3.0, 4.0), (6.0, 8.0)]
        self.assertAlmostEqual(path_length(states), 10.0)


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_eval_metrics -v
```

Expected: ERROR because `evaluation.ogm_metrics` does not exist.

- [ ] **Step 3: Implement metrics**

Create `src/evaluation/ogm_metrics.py`:

```python
import numpy as np


def _sign(value):
    if value > 0:
        return 1
    if value < 0:
        return -1
    return 0


def count_gear_shifts(speeds):
    last = 0
    shifts = 0
    for speed in speeds:
        current = _sign(speed)
        if current == 0:
            continue
        if last != 0 and current != last:
            shifts += 1
        last = current
    return shifts


def path_length(xy_points):
    if len(xy_points) < 2:
        return 0.0
    points = np.asarray(xy_points, dtype=np.float64)
    deltas = points[1:] - points[:-1]
    return float(np.sum(np.linalg.norm(deltas, axis=1)))
```

- [ ] **Step 4: Run tests**

Run:

```powershell
cd D:\Github\HOPE
$env:PYTHONPATH="D:\Github\HOPE\src"
python -m unittest tests.test_eval_metrics -v
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
git add src/evaluation/ogm_metrics.py tests/test_eval_metrics.py
git commit -m "feat: add OGM evaluation metrics"
```

## Task 5: Add OGM SAC Training Entry Point

**Files:**

- Create: `src/train/train_RL_OGM_sac.py`
- Modify: no existing HOPE training script unless duplicate code is later extracted in a separate refactor.

- [ ] **Step 1: Copy the SAC training script**

Run:

```powershell
Copy-Item -LiteralPath 'D:\Github\HOPE\src\train\train_HOPE_sac.py' -Destination 'D:\Github\HOPE\src\train\train_RL_OGM_sac.py'
```

- [ ] **Step 2: Change only the OGM experiment defaults**

In `src/train/train_RL_OGM_sac.py`:

- Instantiate `CarParking(..., use_ogm_observation=True, use_img_observation=False)`.
- Build local actor and critic config dictionaries that set:

```python
actor_params = dict(ACTOR_CONFIGS)
critic_params = dict(CRITIC_CONFIGS)
actor_params['img_shape'] = None
critic_params['img_shape'] = None
actor_params['ogm_shape'] = (OGM_CHANNELS, OGM_SIZE, OGM_SIZE)
critic_params['ogm_shape'] = (OGM_CHANNELS, OGM_SIZE, OGM_SIZE)
actor_params['n_modal'] = 2 + int(USE_ACTION_MASK) + 1
critic_params['n_modal'] = 2 + int(USE_ACTION_MASK) + 1
```

- Set `img_encoder_checkpoint = None`.
- Change log directory prefix from `sac_` to `rl_ogm_sac_`.

- [ ] **Step 3: Add a very short smoke run**

Run:

```powershell
cd D:\Github\HOPE\src
python .\train\train_RL_OGM_sac.py --train_episode 2 --eval_episode 2 --visualize False --verbose False
```

Expected: command exits 0, creates `src/log/exp/rl_ogm_sac_<timestamp>/`, and does not modify HOPE checkpoints.

- [ ] **Step 4: Commit**

Run:

```powershell
git add src/train/train_RL_OGM_sac.py
git commit -m "feat: add RL-OGM SAC training entry"
```

## Task 6: Add OGM Evaluation Entry Point

**Files:**

- Create: `src/evaluation/eval_ogm_mix_scene.py`
- Modify: none required outside this file if shared metrics from Task 4 are used.

- [ ] **Step 1: Copy existing mixed-scene evaluator**

Run:

```powershell
Copy-Item -LiteralPath 'D:\Github\HOPE\src\evaluation\eval_mix_scene.py' -Destination 'D:\Github\HOPE\src\evaluation\eval_ogm_mix_scene.py'
```

- [ ] **Step 2: Enable OGM observation in the evaluator**

In `src/evaluation/eval_ogm_mix_scene.py`, instantiate the environment with:

```python
raw_env = CarParking(
    fps=100,
    verbose=args.verbose,
    render_mode='rgb_array' if not args.visualize else None,
    use_ogm_observation=True,
    use_img_observation=False,
)
```

Mirror the actor/critic config overrides from Task 5 before creating the SAC agent.

- [ ] **Step 3: Run a checkpoint-load smoke test after Task 5 has produced a checkpoint**

Run:

```powershell
cd D:\Github\HOPE\src
python .\evaluation\eval_ogm_mix_scene.py .\log\exp\<rl_ogm_sac_run>\SAC_best.pt --eval_episode 5 --visualize False
```

Expected: command exits 0 and prints PSR-style success rate. If no best checkpoint was saved during a short smoke run, use the latest periodic checkpoint from the run directory.

- [ ] **Step 4: Commit**

Run:

```powershell
git add src/evaluation/eval_ogm_mix_scene.py
git commit -m "feat: add RL-OGM evaluation entry"
```

## Task 7: Local Training Protocol

**Files:**

- Read-only: `src/train/train_RL_OGM_sac.py`
- Read-only: `src/evaluation/eval_ogm_mix_scene.py`
- Create or append: `docs/research/ogm_experiment_log.md`

- [ ] **Step 1: Confirm CUDA availability**

Run:

```powershell
cd D:\Github\HOPE\src
python -c "import torch; print('cuda', torch.cuda.is_available()); print(torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')"
```

Expected: prints CUDA status. CPU is acceptable for smoke tests but likely too slow for full 100k episode training.

- [ ] **Step 2: Run OGM smoke training**

Run:

```powershell
cd D:\Github\HOPE\src
python .\train\train_RL_OGM_sac.py --train_episode 100 --eval_episode 20 --visualize False --verbose True
```

Expected: training completes, TensorBoard logs are written, and evaluation directories are created.

- [ ] **Step 3: Run medium training**

Run:

```powershell
cd D:\Github\HOPE\src
python .\train\train_RL_OGM_sac.py --train_episode 5000 --eval_episode 200 --visualize False --verbose True
```

Expected: success-rate curves should trend upward. Do not expect paper-level results at 5k episodes.

- [ ] **Step 4: Run paper-scale local training**

Run:

```powershell
cd D:\Github\HOPE\src
python .\train\train_RL_OGM_sac.py --train_episode 100000 --eval_episode 2000 --visualize False --verbose True
```

Expected: comparable training budget to HOPE's reported setup. The OGM paper trained for 8 hours on an RTX 4090; local runtime depends on GPU.

- [ ] **Step 5: Record results**

Append to `docs/research/ogm_experiment_log.md`:

```markdown
## RL-OGM SAC full training

Command: `python .\train\train_RL_OGM_sac.py --train_episode 100000 --eval_episode 2000 --visualize False --verbose True`

Hardware:

Checkpoint:

Normal PSR:
Complex PSR:
Extrem PSR:
DLP PSR:
Average path length:
Average gear shifts:

Notes:
```

- [ ] **Step 6: Commit docs and reproducibility notes**

Run:

```powershell
git add docs/research/ogm_experiment_log.md
git commit -m "docs: record RL-OGM training results"
```

## Task 8: Real OGM Dataset Adapter

**Files:**

- Create: `tests/test_ogm_dataset.py`
- Create: `src/env/ogm_dataset.py`
- Modify: `src/env/parking_map_dlp.py` only if a shared map interface is extracted.

- [ ] **Step 1: Define local dataset schema**

Use `.npz` files with these arrays:

```text
global_ogm: uint8 or float32, shape (H, W) or (C, H, W)
resolution: float32 scalar, meters per cell
origin: float32 shape (3,), world x, y, heading for grid origin
start: float32 shape (3,), vehicle start x, y, heading
target: float32 shape (3,), target x, y, heading
obstacle_polygons: object array of polygon coordinate arrays, optional but required for exact collision checks
```

- [ ] **Step 2: Write dataset loading tests**

Create a temporary `.npz` inside the test and assert the loader returns typed fields with expected shapes.

- [ ] **Step 3: Implement loader**

Implement `load_ogm_case(path)` and `OGMCase` dataclass in `src/env/ogm_dataset.py`.

- [ ] **Step 4: Integrate only after proxy OGM training works**

Do not make real OGM data mandatory for normal local training. Add a separate `MAP_LEVEL = 'ogm_dataset'` mode only when data exists.

## Reproduction Targets

Use these as reference targets, not guarantees:

- HOPE paper: current HOPE(SAC) reports 99.4%+ in normal categories and 94%+ across all reported categories.
- RL-OGM paper simulation: Hybrid RL reports 99.33% PSR in Sim-Normal, 97.7% in Sim-Complex, and 87.2% in Real-World dataset scenarios.
- RL-OGM real vehicle: 100% long-distance perpendicular, 85% long-distance parallel, 60% narrow dead-end across 20 attempts each.

Local reproduction can only target the first two families unless real OGM maps and vehicle logs are added.

## Completion Criteria

- Existing HOPE pretrained evaluation still runs.
- OGM utility tests pass.
- Environment returns `ogm` only when requested.
- Default HOPE configs remain checkpoint-compatible.
- `train_RL_OGM_sac.py` can complete a 100-episode smoke run.
- `eval_ogm_mix_scene.py` can report PSR, ANGS, and PL for an OGM checkpoint.
- Full-training results are recorded in `docs/research/ogm_experiment_log.md`.
