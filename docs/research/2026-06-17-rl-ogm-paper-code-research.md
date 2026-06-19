# RL-OGM-Parking Research Notes

Date: 2026-06-17

Scope: compare the local papers in `raw_paper/`, map the current HOPE codebase, and identify an incremental path for introducing the OGM design from `2502.18846_RL-OGM-Parking_Lidar_OGM-Based_Hybrid_Reinforcement_Learning_Planner_for_Autonomous_Parking.pdf`.

Boundary for this research pass: no source code changes. These notes are intentionally limited to paper/code study and future planning.

## Sources Read

- `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`
- `raw_paper/2502.18846_RL-OGM-Parking_Lidar_OGM-Based_Hybrid_Reinforcement_Learning_Planner_for_Autonomous_Parking.pdf`
- `readme.md`
- `src/configs.py`
- `src/env/car_parking_base.py`
- `src/env/env_wrapper.py`
- `src/env/lidar_simulator.py`
- `src/env/observation_processor.py`
- `src/env/parking_map_normal.py`
- `src/env/parking_map_dlp.py`
- `src/model/network.py`
- `src/model/action_mask.py`
- `src/model/agent/parking_agent.py`
- `src/train/train_HOPE_sac.py`
- `src/evaluation/eval_utils.py`

## Executive Summary

The 2502 RL-OGM-Parking paper is best understood as an OGM/perception-driven successor to the 2405 HOPE paper, not as a replacement for HOPE's planner core. It keeps the hybrid planning idea: a learning-based RL planner, a rule-based Reeds-Shepp planner, and an action mask that prevents unsafe actions. The key new design is the perception and environment representation: LiDAR/IMU data is converted into Occupancy Grid Maps (OGMs), and the same OGM-style input is used during simulation training and real-world inference.

The current repository already implements most of the HOPE core: Gym/Pygame simulator, synthetic and DLP map generators, simulated lidar distance vector, BEV image observation, action mask, transformer-style multimodal fusion, SAC/PPO agents, RS fallback/switching, and training/evaluation loops. The largest missing piece for RL-OGM is an explicit OGM observation pipeline and OGM-backed real-world scenario dataset.

The recommended incremental path is:

1. Preserve the current HOPE baseline and reproduce a short local smoke run.
2. Add an OGM observation generated from the existing Shapely map geometry as a local proxy for LiDAR OGM.
3. Train a SAC-based `RL_OGM` variant against the same HOPE scenarios and DLP cases.
4. Add RL-OGM metrics: parking success rate, average number of gear shifts, path length, and optionally operation time.
5. Only after the proxy OGM path works, add a real OGM dataset format for occupancy grids exported from LiDAR mapping.

This should let local training start without waiting for real sensor logs, while keeping a clean path toward the 2502 paper's real-world OGM setup.

## Paper Comparison

| Area | HOPE, arXiv 2405.20579 | RL-OGM-Parking, arXiv 2502.18846 | Implementation implication |
| --- | --- | --- | --- |
| Main problem | Robust path planning across diverse parking scenarios. | Robust autonomous parking with reduced sim-to-real gap. | Keep HOPE planner core; change perception representation first. |
| Planner architecture | Hybrid policy: RL policy plus Reeds-Shepp policy. | Hybrid planner: RL planner plus Reeds-Shepp planner. | Existing `ParkingAgent` and `find_rs_path()` are reusable. |
| RL algorithm | PPO and SAC are both evaluated; repository supports both. | Paper text describes SAC for the RL planner. | Start with SAC only for OGM reproduction. |
| Observation input | Four inputs: obstacle distance vector `lt`, target representation `Ptgt`, action mask `fam`, and low-resolution BEV image `IBEV`. | LiDAR/IMU-derived OGM plus target position. The OGM is used in training and inference. | Add an OGM modal input; target remains. Decide whether to keep lidar/action-mask as auxiliary inputs during transition. |
| Perception representation | Simulated lidar vector and simulator-rendered BEV image. | Global OGM for training scenario generation; local OGM for inference. | Implement geometry-derived global/local OGM first, then real OGM dataset import. |
| Sim-to-real strategy | Simulator and DLP real-world-derived scenarios; real-world demo exists but observation rendering is still simulator-like. | Uses LiDAR OGM to align simulated and real observations, reducing sim-to-real mismatch. | Avoid relying on RGB/Pygame color semantics for the OGM model. |
| Action mask | Computes max safe step at discretized steering angles using obstacle distance vector. | Same idea: restrict maximum safe velocity for steering angles. | Keep `ActionMask`; either continue deriving it from lidar or derive equivalent ray distances from OGM. |
| Dataset | Random Normal/Complex/Extreme scenarios plus DLP dataset. HOPE reports 100,000 training episodes and 2,000 test trials per category. | Simulation dataset with 20 parallel and 50 perpendicular scenarios, plus real-world OGM maps from underground garages. | Current `data/dlp.data` is geometry-based, not OGM-based. It can seed proxy OGM generation. |
| Metrics | Planning success rate, time cost, training curves, ablations. | PSR, ANGS, PL in simulation; AOT and ANGS for real vehicle tests. | Add gear-shift and path-length evaluation logs. |
| Reported simulation results | HOPE(SAC) reaches 99.4%+ in normal cases and 94%+ across all reported categories. | Hybrid RL reports 99.33% PSR in Sim-Normal, 97.7% in Sim-Complex, 87.2% in Real-World dataset scenarios. | Local reproduction should compare both current HOPE metrics and OGM-paper metrics. |
| Reported real-world results | Demonstrates vertical, parallel, and dead-end parking in a garage. | 20 trials per real scenario: 100% PSR long-distance perpendicular, 85% long-distance parallel, 60% narrow dead-end. | Cannot fully reproduce without real vehicle/OGM data; local target is simulation and DLP proxy reproduction. |

## HOPE Paper Notes

The HOPE paper's state representation is explicitly multimodal:

- `lt`: vector-based nearest obstacle distances at angular bins.
- `Ptgt`: target position tuple `(d, cos(theta_t), sin(theta_t), cos(phi_t), sin(phi_t))`.
- `fam`: action mask vector representing max valid step size at steering angles.
- `IBEV`: low-resolution BEV image of drivable area, target parking spot, and ego trajectory.

The method uses a transformer-style fusion network over encoded modality tokens. The appendix gives parameters that match this repository closely: lidar dimension 120, action mask dimension 42, BEV size 64 px, `Tmax=200`, `gamma=0.98`, actor learning rate `5e-6`, replay buffer/batch size `8192`, and `drs=10.0`.

The action mask is not a small detail. The HOPE ablation reports large success drops without action-mask post-processing, especially in complex/extreme parallel parking. Any OGM integration should preserve action-mask behavior during early development.

## RL-OGM Paper Notes

The 2502 paper's main contribution is OGM alignment across training and inference:

- Multiple frames of LiDAR point clouds and IMU data are filtered, then processed by LiDAR-IMU odometry.
- Training uses a global point cloud map registered by estimated poses, then splatted into a 2D global occupancy grid.
- Inference uses multiple recent frames fused into a local keyframe map, then converted into a real-time local OGM.
- The hybrid RL planner consumes OGMs and target positions.

The paper does not provide enough detail in the PDF to recover every OGM implementation parameter. It does not clearly specify the grid resolution, grid dimensions, channel semantics, or exact network topology for the OGM encoder. For local reproduction, the safest initial assumption is to use a 64x64 OGM tensor because the existing HOPE image encoder already expects 64x64 BEV input.

## Current Code Map

### Configuration

`src/configs.py` centralizes vehicle dimensions, scenario difficulty parameters, observation toggles, model shapes, and rewards.

Key values:

- `MAP_LEVEL = 'Normal'`
- `OBS_W = 256`, `OBS_H = 256`, downsampled to 64x64 by `Obs_Processor`.
- `LIDAR_RANGE = 10.0`, `LIDAR_NUM = 120`.
- `TOLERANT_TIME = 200`.
- `USE_LIDAR = True`, `USE_IMG = True`, `USE_ACTION_MASK = True`.
- `RS_MAX_DIST = 10`.
- `N_DISCRETE_ACTION = 42`.
- `GAMMA = 0.98`, `BATCH_SIZE = 8192`, `LR = 5e-6`.
- Actor/critic `n_modal = 2 + int(USE_IMG) + int(USE_ACTION_MASK)`, corresponding to lidar, target, optional action mask, optional image.

### Environment

`src/env/car_parking_base.py` defines `CarParking`, a Gym environment. Its `render()` method returns:

- `img`: processed BEV-like Pygame RGB crop, if image observation is enabled.
- `lidar`: simulated ray distances from `LidarSimlator`, if lidar observation is enabled.
- `action_mask`: max safe step values from `ActionMask`, if enabled.
- `target`: relative target representation.

`src/env/lidar_simulator.py` computes vector lidar distances from Shapely obstacle geometry in the ego frame.

`src/env/observation_processor.py` changes the white background to black, resizes the RGB image to 64x64, and scales values to `[0, 1]`.

`src/env/env_wrapper.py` transposes image observations from HWC to CHW and applies reward shaping and action rescaling.

### Map Sources

`src/env/parking_map_normal.py` generates bay and parallel parking cases by randomized rules. Difficulty is controlled by parking-lot dimensions, wall distances, and obstacle counts.

`src/env/parking_map_dlp.py` loads `../data/dlp.data`, which contains start, destination, and Shapely obstacle geometry. This is not an OGM dataset, but it is useful for generating proxy OGMs from real-world-derived geometry.

### Hybrid Policy

`src/env/car_parking_base.py` checks for a feasible RS path when the vehicle is within `RS_MAX_DIST` of the destination. If found, it returns `info['path_to_dest']`.

`src/model/agent/parking_agent.py` wraps an RL agent and an `RsPlanner`. When an RS route is active, actions come from the RS route; otherwise they come from the RL policy. This matches HOPE and remains useful for RL-OGM.

### Network

`src/model/network.py` defines `MultiObsEmbedding`. It creates separate embeddings for lidar, target, action mask, optional image, and optional critic action input. It then stacks feature tokens into `AttentionNetwork` if attention is enabled.

The existing image branch can encode a 64x64 tensor. For OGM integration there are two practical choices:

1. Reuse the `img` key and feed OGM as a 3-channel image. This is fastest but hides semantic differences.
2. Add an explicit `ogm` modality and encoder. This is cleaner and easier to ablate.

The recommended path is explicit `ogm`, with an initial encoder copied from the image encoder pattern.

### Training And Evaluation

`src/train/train_HOPE_sac.py` is the best starting point for local RL-OGM reproduction because the 2502 paper describes SAC.

The training loop already:

- Selects among Normal, Complex, Extrem, and DLP scenes.
- Uses a warm-up period while replay memory fills.
- Logs reward, action std, entropy alpha, scene success rates, and step counts.
- Saves best checkpoints and periodic checkpoints.
- Runs evaluation on DLP, Extrem, Complex, and Normal at the end.

`src/evaluation/eval_utils.py` already records success, step counts, reward, and path length. It does not currently record gear shifts or operation time in the OGM paper's metric format.

## Findings That Matter For Future Work

1. The fastest useful OGM prototype can be built from existing Shapely geometry. It will not reproduce LiDAR noise or LIO mapping, but it will test the network/training path.
2. `data/dlp.data` is geometry, not occupancy grids. Treat it as a source for proxy OGM generation, not as a reproduction of the 2502 real-world OGM dataset.
3. The current action mask depends on lidar-style ray distances. For the first OGM variant, keeping the lidar vector solely for action-mask computation is acceptable, even if the policy input shifts to OGM. Later, derive the mask from OGM ray-casting for stricter alignment.
4. The current `target` code appears to return `cos(rel_dest_heading)` twice, while the paper describes `(cos(phi), sin(phi))`. This should be investigated in a separate bugfix pass before or during OGM implementation. It was not changed in this research pass.
5. The code uses `Extrem` rather than `Extreme` as a scenario level. Future files should preserve the existing spelling unless a migration is intentionally planned.
6. Full reproduction of the 2502 real-world numbers is not possible from this repository alone because the real OGM maps, LiDAR/IMU logs, and vehicle platform are not present.

## Recommended Development Strategy

Use a staged strategy:

### Stage 0: Baseline Audit

Run current HOPE pretrained evaluation and a short training smoke test. Capture the current metrics before changing any model input.

### Stage 1: Proxy OGM

Create an `ogm` observation by rasterizing current obstacle geometry, target box, and optionally ego vehicle or trajectory into a local ego-centered grid. This makes training and inference input look like an occupancy grid while requiring no external sensor data.

### Stage 2: OGM SAC Variant

Add explicit OGM model configuration and a separate `train_RL_OGM_sac.py` script. Keep HOPE training scripts untouched.

### Stage 3: OGM Metrics

Extend evaluation to report PSR, ANGS, PL, and optionally AOT-like elapsed time. Compare against the 2502 paper's table structure.

### Stage 4: Real OGM Dataset Adapter

Define a simple local dataset format such as `.npz` with:

- `global_ogm`: float or uint8 occupancy grid.
- `resolution`: meters per cell.
- `origin`: world-frame origin.
- `start`: `(x, y, heading)`.
- `target`: `(x, y, heading)`.
- `obstacle_polygons` if geometry is available for collision checks.

This keeps training reproducible even without the original authors' garage data.

## Open Questions

1. Should the first OGM policy remove lidar from the policy input entirely, or keep lidar as an auxiliary token while introducing OGM?
2. Should the old RGB BEV image be disabled for OGM runs, or retained for an ablation?
3. Is the goal "match 2502 simulation tables as closely as possible" or "obtain a strong local OGM-trained planner on this repo's available scenarios"?
4. Do we want to fix the target representation `cos/cos` issue before OGM work, or preserve exact checkpoint compatibility first?
