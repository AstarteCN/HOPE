# Stage 4 OGM Debug Viewer Design

Date: 2026-06-22
Status: design approved in brainstorming; awaiting implementation plan
Repository: `D:\Github\HOPE`

## 1. Goal

Build a non-invasive Stage 4 OGM debug tool for analyzing parking planning behavior after checkpoints are produced. The tool must not attach to, pause, or mutate the active training process. It should replay selected checkpoints and cases offline, export a trace JSON, then render that trace in a local Web UI.

The main debugging questions are:

- What did the OGM layer contain at each step?
- How do HOPE lidar-ray observations compare against the OGM grid?
- What trajectory did the RL policy execute?
- What RS reference trajectory was available?
- Which status, action, reward, and safety signals explain a success, timeout, collision, long path, or excessive gear shifting?

## 2. Approved Direction

The selected implementation route is:

1. Export an offline trace JSON from a checkpoint and fixed cases.
2. Load the trace JSON in a local Web viewer.
3. Keep data extraction and UI rendering decoupled so later exporters can target Rerun or Foxglove/MCAP without rewriting replay logic.

This was chosen over a direct Rerun-only implementation because the user wants a specific two-column interface, custom playback controls, and selective layer toggles.

## 3. Non-Goals

- Do not interrupt the active 40K+ OGM training continuation.
- Do not change training code or training semantics.
- Do not replace the current Stage 4 gate/eval scripts.
- Do not require Product Design plugin workflows before the first engineering implementation.
- Do not integrate CARLA, Autoware, Apollo, ROS, or Foxglove in this first version.
- Do not resample the viewer timeline into 20 ms substeps.

## 4. Source Evidence And Vehicle Model

Use the vehicle parameters already present in both the HOPE paper and local code:

- Vehicle length: `4.69 m`
- Vehicle width: `1.94 m`
- Wheelbase: `2.8 m`
- Front hang: `0.96 m`
- Rear hang: `0.93 m`
- Maximum velocity: `2.5 m/s`
- Maximum steering angle: `0.75 rad`

Local source anchors:

- `src/configs.py` defines `WHEEL_BASE`, `FRONT_HANG`, `REAR_HANG`, `LENGTH`, `WIDTH`, and `VehicleBox`.
- `src/env/vehicle.py` states that the kinematic single-track model uses the center of the rear wheels as the local coordinate-system origin.

Therefore the debug viewer should not fall back to Passat-like generic dimensions. It should use the HOPE training vehicle parameters.

## 5. Coordinate System

The local debug map coordinate frame is ego-centered:

- Origin: vehicle rear-axle center.
- Positive x-axis: vehicle heading/front direction.
- Positive y-axis: vehicle left.
- Vehicle footprint: rectangle from `x=-0.93` to `x=3.76`, and `y=-0.97` to `y=0.97`.
- Rear axle: line segment across the vehicle width at `x=0`.
- Front axle: line segment across the vehicle width at `x=2.8`.
- Rear-axle center marker: point at `(0, 0)`.

Current HOPE `State.loc` is already the rear-axle center. If a future external trace source uses vehicle center or another reference point, the exporter must map it to rear-axle-center coordinates before writing the trace.

## 6. Time Model

One viewer step equals one HOPE environment interaction step.

- Display duration per viewer step: `500 ms`.
- Source rationale: HOPE reports `Delta t = 0.5 s`; local code has `NUM_STEP = 10` and `STEP_LENGTH = 0.05`, so one interaction step is `0.5 s`.
- No 20 ms interpolation is required for playback.

Trajectory waypoint densification is separate from the timeline. It is only used for visual continuity.

## 7. OGM Grid And Occupancy Semantics

Use the current Stage 4 proxy OGM configuration:

- Grid size: `64 x 64`.
- Resolution: `1 / 3 m` per cell.
- Span: approximately `21.33 m x 21.33 m`.
- Channels: `2`.
- Channel 0: obstacle occupancy.
- Channel 1: target-slot occupancy.

The OGM paper describes LiDAR-derived Occupancy Grid Maps, global OGM for training scenario generation, and local OGM for inference, but does not expose enough PDF detail to recover exact grid size, resolution, or channel encoding. The viewer should therefore label the grid as the current repository's proxy OGM assumptions.

## 8. Lidar Projection

The viewer must show HOPE's lidar-ray observations on the same local grid for comparison with OGM occupancy.

Trace exporter responsibilities:

- Read `obs["lidar"]`, which has `120` rays and a max range of `10 m`.
- Use ray angle `theta_i = i * 2*pi / LIDAR_NUM`.
- Convert each lidar distance to an ego-frame hit point.
- Keep lidar hit points as a separate overlay layer, not as an OGM channel mutation.
- Include minimum lidar distance and the nearest hit point as safety signals.

The local lidar values are already adjusted by the vehicle boundary in `LidarSimlator.get_observation()`. Reports and UI labels should make this explicit.

## 9. Trajectories

The trace must include both RL and RS trajectories.

### RL Trajectory

RL trajectory is the actual executed vehicle trajectory during replay, recorded from rear-axle-center states after each environment step.

### RS Reference Trajectory

RS trajectory is the reference route from the rule-based planner when available:

- Prefer `info["path_to_dest"]` from the environment step.
- If the `ParkingAgent` has an active RS route, record the active route metadata and sampled points when accessible.
- Record whether the executed action source at each step is `RL` or `RS`.

### Waypoint Spacing

All displayed trajectories should have waypoint spacing of at most `0.4 m`.

- If source trajectory points are farther apart, densify linearly for visualization.
- Densification must not add extra viewer time steps.
- Keep original source points separately when useful for debug.

## 10. Trace JSON

Add a trace schema that can be generated independently of the viewer.

Expected top-level fields:

- `schema_version`
- `source`: checkpoint path, case file path, exporter command, git info if cheap to capture
- `vehicle`: dimensions and kinematic limits
- `grid`: size, resolution, channel names, origin semantics
- `case`: case id, case uid, split, parking type, HOPE level, seed
- `static_geometry`: map bounds, obstacles, start box, target box
- `frames`: per-step records
- `summary`: terminal status, success, step count, path length, gear shifts, total reward

Expected per-frame fields:

- `step_index`
- `sim_time_s`
- `ego_state`: rear-axle center x/y/heading, speed, steering
- `vehicle_geometry`: body corners, front axle, rear axle, rear-axle center
- `ogm`: obstacle and target channels, preferably as compact nested arrays or compressed sidecar when large
- `lidar`: ray endpoints, hit points, min distance
- `action`: steer, speed, source `RL` or `RS`
- `action_mask`: raw vector and compact summary
- `trajectory`: RL executed path up to current step, RS reference path when available
- `reward`: total reward and components
- `status`: current environment status and terminal flag
- `signals`: derived debug signals such as direction, distance to target, heading error, collision/outbound flags

## 11. Web Viewer Layout

The UI is a two-column tool surface.

Left column:

- Primary OGM grid map.
- Ego vehicle rectangle and wheel/axle geometry.
- Trajectory overlays.
- Lidar and action-mask overlays.
- Step playback controls close to the map.

Right column:

- Layer-control panel.
- Current state signals.
- Vehicle/action/policy information.
- Case/checkpoint metadata.

This should feel like a dense engineering debug console, not a marketing page.

## 12. Playback Controls

Required controls:

- Play / pause.
- Single-step forward.
- Single-step backward if cheap to implement.
- Fast-forward `10` steps.
- Jump to a typed step index.
- Current step display.
- Simulation time display as `step_index * 0.5 s`.

The UI should clamp jumps to valid frame indices and keep the map/status panel synchronized.

## 13. Layer Controls

The right-side control panel must include first-level groups and second-level toggles.

Recommended initial groups:

### Environment Model

- Map bounds.
- Obstacle polygons.
- Target slot.
- OGM obstacle channel.
- OGM target channel.

### Perception

- Lidar rays.
- Lidar hit points.
- Minimum-distance marker.
- Action-mask feasible direction sectors.

### Vehicle Model

- Body rectangle.
- Front wheels.
- Rear wheels.
- Front axle.
- Rear axle.
- Rear-axle center.
- Heading arrow.

### Trajectory Control

- RL trajectory.
- RL waypoints.
- RS trajectory.
- RS waypoints.
- Current-step marker.

### Debug Annotations

- Executed action vector.
- Terminal status marker.
- Collision/outbound marker.
- Reward label.
- Case metadata.

Defaults should prioritize readability:

- On by default: OGM obstacle, target slot, ego body, rear-axle center, RL trajectory, current-step marker.
- Off by default: dense lidar rays, action-mask sectors, waypoint dots, reward labels.

## 14. Status Signals

The right panel should include at least:

- Current driving direction: `forward`, `reverse`, or `stopped`.
- Collision distance or nearest obstacle distance.
- Minimum lidar distance.
- Speed.
- Steering angle.
- Heading.
- Rear-axle center coordinate.
- Gear-shift count.
- Cumulative path length.
- Distance to target.
- Heading error to target.
- Current action source: `RL` or `RS`.
- Current steer/speed action.
- Action-mask summary: valid count, max allowed bin, nearest blocked bin if useful.
- Reward and reward components.
- Environment status.
- Case id, split, checkpoint.

## 15. Product Design Plugin Decision

The Product Design plugin can help later with visual polish, UX alternatives, prototype sharing, and design QA. It is not the main workflow for this first implementation because:

- The core risk is correctness of HOPE/OGM data, coordinates, vehicle geometry, and trace synchronization.
- The UI requirements are already concrete.
- The Product Design plugin would add extra ideation gates and visual-source selection before implementation.

Decision: do not route the first engineering implementation through Product Design. Reconsider Product Design after the first working viewer exists, for a focused UI refinement or design QA pass.

## 16. Implementation Boundaries

Implementation should be opt-in and tool-only:

- Prefer new files under `tools/stage4/`.
- Do not edit active training scripts unless a later plan proves it is required.
- Do not write into active run directories except by reading checkpoint/case inputs.
- Generated trace files should live under `docs/research/` or a local ignored output path, depending on size.
- Large traces should not be committed unless the user explicitly asks.

## 17. Test Strategy

Unit tests should cover:

- Vehicle parameters match `src/configs.py`.
- Vehicle rectangle uses rear-axle center origin.
- Front/rear axle and wheel markers are placed correctly.
- OGM grid metadata matches current Stage 4 config.
- Lidar rays project to expected local hit points.
- Trace frame time equals `step_index * 0.5`.
- Waypoint densification keeps segment length at or below `0.4 m`.
- RL and RS trajectory fields are independently toggleable in schema.
- Layer-control config contains the approved first-level and second-level menu structure.
- Trace exporter can emit at least one small fixed-case trace with finite values.

Viewer tests should cover:

- Loading a minimal trace.
- Step jump clamping.
- Play/pause state changes.
- Toggling second-level layers independently.
- Keeping right-panel signals synchronized with the selected frame.

## 18. Next Step

Convert this approved design into an implementation plan. The plan should start with trace schema and exporter tests, then build the Web viewer from a small fixture trace before using real checkpoints.
