# Stage 4 OGM Debug Viewer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a non-invasive Stage 4 OGM debug viewer that exports HOPE/OGM replay traces and renders a two-column visual debugging tool for parking planning, OGM, lidar, vehicle geometry, RL trajectory, and RS reference trajectory.

**Architecture:** Add a standalone trace exporter under `tools/stage4/` and a static browser viewer under `tools/stage4/debug_viewer/`. The exporter reads evaluation-time HOPE data without modifying training loops. The viewer consumes JSON traces and renders everything in rear-axle local coordinates. Training processes, active logs, checkpoints, and original HOPE source paths stay untouched except for read-only imports.

**Tech Stack:** Python 3 project environment, stdlib `argparse/json/dataclasses`, existing HOPE modules (`src/configs.py`, `src/env/*`, `src/model/*`), plain HTML/CSS/JavaScript Canvas 2D, Python `unittest`, optional Node syntax checks if `node` is available.

**Execution Mode:** The user already approved subagent-driven implementation. After this plan is saved, use a fresh subagent for each implementation task and run a dedicated review subagent before completion.

---

## Non-Negotiable Constraints

- [ ] Do not stop, restart, attach to, or modify the ongoing OGM training process.
- [ ] Do not edit original training entry points under `src/train/`.
- [ ] Do not overwrite or regenerate checkpoints under `src/model/ckpt/`.
- [ ] Do not stage unrelated dirty files from the active training tracker.
- [ ] Do not commit large exported traces, TensorBoard logs, checkpoints, or raw training artifacts.
- [ ] Keep viewer timing aligned to HOPE: one viewer step equals one HOPE environment interaction step, `0.5 s`. Do not interpolate to `20 ms`.
- [ ] Use rear axle center as the coordinate origin. Use `x` forward and `y` left in the local map.
- [ ] Label Stage 4 OGM grid semantics as the current HOPE proxy OGM unless a later source provides exact RL-OGM grid parameters.

---

## Confirmed Technical Constants

Read constants from `src/configs.py` in code rather than duplicating values in multiple modules. The current values are:

- Vehicle length: `4.69 m`
- Vehicle width: `1.94 m`
- Wheelbase: `2.8 m`
- Front hang: `0.96 m`
- Rear hang: `0.93 m`
- Body rectangle in rear-axle frame: `x=[-0.93, 3.76]`, `y=[-0.97, 0.97]`
- Max velocity: `2.5 m/s`
- Max steering angle: `0.75 rad`
- Lidar: `120` rays, `10.0 m` max range, `2*pi/120` angular spacing
- OGM proxy grid: `64 x 64`, `1/3 m` per cell, `2` channels
- OGM channel `0`: obstacle occupancy
- OGM channel `1`: target occupancy
- Viewer simulation step: `0.5 s`
- Visual trajectory waypoint spacing: max `0.4 m`

---

## Expected Trace JSON Shape

Create one stable JSON schema for exporter and viewer:

```json
{
  "schema_version": "stage4-ogm-debug-v1",
  "source": {
    "checkpoint": "src/model/ckpt/...",
    "created_by": "tools/stage4/debug_trace_export.py",
    "created_at": "2026-06-22T00:00:00+08:00",
    "git_commit": "optional"
  },
  "vehicle": {
    "length_m": 4.69,
    "width_m": 1.94,
    "wheelbase_m": 2.8,
    "front_hang_m": 0.96,
    "rear_hang_m": 0.93,
    "max_velocity_mps": 2.5,
    "max_steer_rad": 0.75
  },
  "grid": {
    "size": 64,
    "resolution_m": 0.3333333333333333,
    "origin": "rear_axle_center",
    "x_forward": true,
    "y_left": true,
    "channels": ["obstacle", "target"]
  },
  "case": {
    "scene": "Sim-Normal",
    "case_id": 0,
    "case_uid": "optional",
    "description": "optional"
  },
  "static_geometry": {
    "obstacle_polygons": [[[0.0, 0.0], [1.0, 0.0]]],
    "target_polygon": [[0.0, 0.0]],
    "map_bounds": [-10.67, -10.67, 10.67, 10.67]
  },
  "frames": [
    {
      "step": 0,
      "sim_time_s": 0.0,
      "ego": {"x": 0.0, "y": 0.0, "heading": 0.0, "speed": 0.0, "steer": 0.0},
      "drive_direction": "stopped",
      "action": {"speed": 0.0, "steer": 0.0},
      "collision_distance_m": 1.2,
      "terminal": false,
      "terminal_reason": null,
      "reward": 0.0,
      "ogm": {
        "size": 64,
        "obstacle": [[0, 0]],
        "target": [[0, 0]]
      },
      "lidar": {
        "distances_m": [10.0],
        "hit_points": [[10.0, 0.0]]
      },
      "rl_trajectory": [[0.0, 0.0, 0.0]],
      "rs_trajectory": [[0.0, 0.0, 0.0]],
      "status": {
        "success": false,
        "outbound": false,
        "collision": false,
        "path_to_dest_m": 0.0,
        "target_distance_m": 0.0,
        "heading_error_rad": 0.0
      }
    }
  ]
}
```

Keep the schema permissive enough for partial data. The viewer must render fixture traces even if checkpoint-only fields are absent.

---

## Task 1: Add Trace Schema And Geometry Helpers

**Files:**

- Create `tools/stage4/debug_trace_schema.py`
- Create `tools/stage4/tests/test_debug_trace_schema.py`

**Implementation:**

- [ ] Add plain dictionaries/dataclasses for vehicle, grid, layer defaults, frame, and trace metadata.
- [ ] Import constants from `src/configs.py`.
- [ ] Implement rear-axle-frame vehicle geometry:

```python
def vehicle_params_from_configs() -> dict:
    return {
        "length_m": float(LENGTH),
        "width_m": float(WIDTH),
        "wheelbase_m": float(WHEEL_BASE),
        "front_hang_m": float(FRONT_HANG),
        "rear_hang_m": float(REAR_HANG),
        "max_velocity_mps": float(VALID_SPEED[1]),
        "max_steer_rad": float(VALID_STEER[1]),
    }

def vehicle_body_corners_local(params: dict) -> list[list[float]]:
    half_width = params["width_m"] / 2.0
    return [
        [-params["rear_hang_m"], -half_width],
        [params["wheelbase_m"] + params["front_hang_m"], -half_width],
        [params["wheelbase_m"] + params["front_hang_m"], half_width],
        [-params["rear_hang_m"], half_width],
    ]
```

- [ ] Implement `densify_polyline(points, max_spacing_m=0.4)` for `[x, y]` and `[x, y, heading]` points. Preserve the first and last waypoint.
- [ ] Implement `project_lidar_hits(distances_m, vehicle_boundary_m=None)` using `2*pi/LIDAR_NUM` ray spacing. If `vehicle_boundary_m` is provided, add it back before projection because HOPE subtracts vehicle boundary from lidar observations.
- [ ] Implement `ogm_default_grid()` returning size, resolution, origin, and channels from current Stage 4 proxy constants.
- [ ] Implement `default_layer_groups()` with the approved first-level and second-level layer toggles:
  - `environment_model`: map bounds, obstacle polygons, target slot, OGM obstacle, OGM target
  - `perception`: lidar rays, lidar hit points, minimum-distance marker, action-mask feasible sectors
  - `vehicle_model`: body, front wheels, rear wheels, front axle, rear axle, rear-axle center, heading arrow
  - `trajectory_control`: RL trajectory, RL waypoints, RS trajectory, RS waypoints, current-step marker
  - `debug_annotations`: executed action vector, terminal status marker, collision/outbound marker, reward label, case metadata

**Tests:**

- [ ] Vehicle params equal `src/configs.py` values.
- [ ] Body corners produce rear axle at `x=0` and front axle at `x=2.8`.
- [ ] Densifier produces segment spacing `<= 0.4 m` and preserves endpoints.
- [ ] Lidar projection returns 120 points for 120 distances and places ray `0` at positive local `x`.
- [ ] Layer defaults include all required groups and RS/RL toggles separately.

**Run:**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_debug_trace_schema -v
```

---

## Task 2: Add Fixture Trace Builder And Writer

**Files:**

- Modify `tools/stage4/debug_trace_schema.py`
- Create `tools/stage4/debug_trace_export.py`
- Create `tools/stage4/tests/test_debug_trace_export.py`
- Create `tools/stage4/tests/fixtures/stage4_debug_trace_minimal.json` only if the file remains small and deterministic.

**Implementation:**

- [ ] Add `build_trace(source, case, static_geometry, frames)` that wraps metadata, vehicle params, grid params, layer defaults, and frames.
- [ ] Add `write_trace_json(trace, output_path)` using UTF-8 and stable indentation.
- [ ] Add `build_fixture_trace(frame_count=6)` with:
  - Simple obstacle polygon and target polygon.
  - RL and RS trajectories with visibly different lines.
  - OGM obstacle and target channel arrays.
  - Lidar distances and hit points.
  - Vehicle state, direction, collision distance, reward, and status.
- [ ] Add CLI fixture mode:

```powershell
.\.venv\Scripts\python.exe .\tools\stage4\debug_trace_export.py --fixture --output .\tools\stage4\tests\fixtures\stage4_debug_trace_minimal.json
```

- [ ] The fixture mode must not import torch or load checkpoints.

**Tests:**

- [ ] Fixture trace validates `schema_version`, `vehicle`, `grid`, `case`, `frames`.
- [ ] `sim_time_s` increments exactly by `0.5`.
- [ ] RS and RL trajectories are present as separate arrays.
- [ ] JSON writer creates parent directories and round-trips through `json.load`.
- [ ] Running fixture CLI succeeds without checkpoint arguments.

**Run:**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_debug_trace_export -v
.\.venv\Scripts\python.exe .\tools\stage4\debug_trace_export.py --fixture --output .\tools\stage4\tests\fixtures\stage4_debug_trace_minimal.json
```

---

## Task 3: Add Non-Invasive HOPE Replay Exporter

**Files:**

- Modify `tools/stage4/debug_trace_export.py`
- Create or extend `tools/stage4/tests/test_debug_trace_export.py`

**Implementation:**

- [ ] Keep `--fixture` as the default low-risk smoke path.
- [ ] Add checkpoint replay CLI arguments:

```text
--checkpoint
--output
--cases-json
--case-index
--case-uid
--max-steps
--visualize False
```

- [ ] Use existing Stage 4 case definitions from `tools/stage4/stage4_ogm_cases.py` when `--cases-json` is absent.
- [ ] Use existing OGM wrapper/evaluation code as read-only references. Reuse current Stage 4 OGM agent construction patterns instead of inventing a second model-loading path.
- [ ] Set `SDL_VIDEODRIVER=dummy` inside the exporter when replaying unless the caller has already set it.
- [ ] For each replay frame collect:
  - Ego rear axle pose from `env.vehicle.state`.
  - Speed/steer/action from the current state and selected action.
  - Current OGM proxy channels from the Stage 4 OGM observation path.
  - Lidar distances and projected hit points.
  - Static obstacle polygons and target polygon from environment/map geometry when available.
  - RL trajectory as executed rear-axle poses up to the current step.
  - RS reference path from `info["path_to_dest"]` when available, densified to `<= 0.4 m`.
  - Direction label: `forward`, `reverse`, or `stopped`.
  - Collision/outbound/success/terminal flags from `info` and environment signals.
  - Collision distance proxy from minimum lidar hit distance or available collision metric.
- [ ] If the model outputs path points in a coordinate convention other than rear axle center, add a mapping function and document the mapping in `source.notes`. Current HOPE `State.loc` is rear axle center, so the normal path should be identity.
- [ ] The replay exporter must never train, optimize, or save model checkpoints.

**Tests:**

- [ ] Unit-test argument parsing and fixture mode.
- [ ] Unit-test `direction_from_speed`.
- [ ] Unit-test `extract_rs_trajectory_from_info` with fake `path_to_dest`.
- [ ] Unit-test that fixture mode avoids importing torch-heavy replay modules until needed. This can be a structural test around lazy imports.

**Manual Smoke Command:**

Run only after unit tests pass. Use a small `--max-steps` replay to avoid interfering with the active training run:

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
.\.venv\Scripts\python.exe .\tools\stage4\debug_trace_export.py --checkpoint .\src\model\ckpt\HOPE_SAC0.pt --case-index 0 --max-steps 5 --output .\tools\stage4\debug_viewer\sample_trace.json
```

If GPU contention is a concern during active training, skip the checkpoint smoke and rely on fixture trace until the training tracker says the GPU is free.

---

## Task 4: Add Static Viewer Scaffold

**Files:**

- Create `tools/stage4/debug_viewer/index.html`
- Create `tools/stage4/debug_viewer/styles.css`
- Create `tools/stage4/debug_viewer/viewer_core.js`
- Create `tools/stage4/debug_viewer/app.js`
- Create `tools/stage4/tests/test_debug_viewer_static.py`

**Implementation:**

- [ ] Build a two-column debug console:
  - Left: full-height map/canvas workspace.
  - Right: control/status/layer sidebar.
- [ ] Add controls:
  - File input for trace JSON.
  - Play/pause button.
  - Single-step button.
  - Optional previous-step button.
  - `+10` fast-forward button.
  - Numeric step jump input.
  - Current step and total step display.
- [ ] Add right-side status signals:
  - Current driving direction.
  - Collision distance.
  - Speed.
  - Steering angle.
  - Step index.
  - Simulation time.
  - Reward.
  - Terminal/success/collision/outbound.
  - Distance to target.
  - Heading error.
  - RS/RL trajectory lengths.
  - Case scene/case id.
- [ ] Add hierarchical layer panel exactly matching `default_layer_groups()`.
- [ ] Load `sample_trace.json` automatically when available, and show a useful empty state when it is absent.
- [ ] Do not add a landing page or marketing copy. The first screen is the usable debug tool.

**Tests:**

- [ ] Static test confirms all required files exist.
- [ ] Static test confirms `index.html` references `styles.css`, `viewer_core.js`, and `app.js`.
- [ ] Static test confirms required control ids exist.
- [ ] Static test confirms layer group labels exist in the JS defaults or HTML.

**Run:**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest tools.stage4.tests.test_debug_viewer_static -v
```

---

## Task 5: Implement Canvas Rendering And Interaction

**Files:**

- Modify `tools/stage4/debug_viewer/viewer_core.js`
- Modify `tools/stage4/debug_viewer/app.js`
- Modify `tools/stage4/debug_viewer/styles.css`
- Create `tools/stage4/tests/debug_viewer_core_node_test.js`

**Implementation:**

- [ ] In `viewer_core.js`, implement pure functions:
  - `clampStep(index, total)`
  - `worldToCanvas(point, viewport)`
  - `gridCellToLocal(row, col, grid)`
  - `normalizeLayerGroups(groups)`
  - `trajectoryLength(points)`
  - `densifyPolyline(points, maxSpacing)`
  - `statusRowsForFrame(trace, frame)`
- [ ] In `app.js`, implement:
  - Trace load from file input.
  - Auto-load `sample_trace.json` if fetch succeeds.
  - Playback timer with one frame advance every `500 ms`.
  - Pause at last frame.
  - Step, previous-step, `+10`, and jump controls.
  - Layer checkbox event handling.
  - Responsive canvas resize.
- [ ] Render map elements with stable colors and independent layer toggles:
  - OGM obstacle cells.
  - OGM target cells.
  - Static obstacle polygons.
  - Target slot polygon.
  - Lidar rays.
  - Lidar hit points.
  - Vehicle body rectangle.
  - Front wheels and rear wheels.
  - Front and rear axles.
  - Rear-axle center point.
  - Heading arrow.
  - RL trajectory and waypoints.
  - RS trajectory and waypoints.
  - Current-step marker.
  - Collision/min-distance marker.
- [ ] Keep visual hierarchy readable:
  - Map background: dark neutral grid.
  - Obstacles: red/orange.
  - Target: green.
  - Lidar: cyan/blue.
  - RL: yellow.
  - RS: white or purple, clearly distinct from RL.
  - Vehicle: high-contrast outline and wheels.
- [ ] Make the sidebar dense and engineering-focused. Avoid hero UI, decorative cards, and explanatory feature text.
- [ ] Ensure text does not overflow controls on typical desktop and narrow widths.

**Tests:**

- [ ] Node test validates pure functions from `viewer_core.js`.
- [ ] `node --check` passes for both JS files when Node is available.
- [ ] Manual browser smoke opens the viewer and verifies fixture trace controls/layers.

**Run:**

```powershell
cd D:\Github\HOPE
node --check .\tools\stage4\debug_viewer\viewer_core.js
node --check .\tools\stage4\debug_viewer\app.js
node .\tools\stage4\tests\debug_viewer_core_node_test.js
```

If `node` is unavailable in PATH, use the Codex bundled Node runtime or record that JS runtime verification was skipped and rely on browser smoke plus static tests.

---

## Task 6: Add Usage Documentation And Safe Workflow Notes

**Files:**

- Create `docs/research/2026-06-22-stage4-ogm-debug-viewer-usage.md`
- Optionally create `tools/stage4/debug_viewer/README.md` if the viewer directory needs local launch notes.

**Implementation:**

- [ ] Document fixture trace generation.
- [ ] Document checkpoint replay export with small `--max-steps`.
- [ ] Document how to open the viewer:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m http.server 8765 -d .\tools\stage4\debug_viewer
```

Then open:

```text
http://127.0.0.1:8765/
```

- [ ] Document the non-interference rule: exporter is read-only replay/evaluation, not training.
- [ ] Document that large exported traces should remain local and untracked.
- [ ] Document current proxy OGM assumption and exact constants.
- [ ] Document layer menu structure and required status signals.

**Tests:**

- [ ] Docs include fixture command, checkpoint command, server command, and non-interference note.
- [ ] Docs do not instruct users to edit `src/train/` or active training monitor files.

---

## Task 7: Full Verification And Review

**Files:**

- No new feature files unless review finds a concrete defect.

**Verification Commands:**

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v
git diff --check
git status -sb
```

Optional checkpoint replay smoke only when it will not contend with active training:

```powershell
cd D:\Github\HOPE
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
.\.venv\Scripts\python.exe .\tools\stage4\debug_trace_export.py --checkpoint .\src\model\ckpt\HOPE_SAC0.pt --case-index 0 --max-steps 5 --output .\tools\stage4\debug_viewer\sample_trace.json
```

Static local server smoke:

```powershell
cd D:\Github\HOPE
.\.venv\Scripts\python.exe -m http.server 8765 -d .\tools\stage4\debug_viewer
```

**Review Requirements:**

- [ ] Run a spec-compliance review subagent against `docs/superpowers/specs/2026-06-22-stage4-ogm-debug-viewer-design.md`.
- [ ] Run a code-quality review subagent against all changed files.
- [ ] Fix all critical review findings.
- [ ] Confirm final diff does not include active training tracker files unless intentionally created by this task.
- [ ] Confirm no files under `src/model/ckpt/` are changed.
- [ ] Confirm no training loop files under `src/train/` are changed.

---

## Final Deliverables

- [ ] Trace schema/helpers: `tools/stage4/debug_trace_schema.py`
- [ ] Trace exporter: `tools/stage4/debug_trace_export.py`
- [ ] Static viewer: `tools/stage4/debug_viewer/index.html`, `styles.css`, `viewer_core.js`, `app.js`
- [ ] Small deterministic fixture trace or fixture generator.
- [ ] Tests under `tools/stage4/tests/`.
- [ ] Usage documentation under `docs/research/`.
- [ ] Clear final report with exact commands run, skipped verification if any, and local viewer URL or launch command.
