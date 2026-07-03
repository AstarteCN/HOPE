# RL-OGM Integration Research Against Current HOPE Code

Date: 2026-06-20

Scope: research how the RL-OGM-Parking paper can be integrated into the current HOPE repository structure, using the two local papers and current code state. This is a design/research note only. It does not approve or implement Stage 4 OGM source work.

## Sources

- `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`
- `raw_paper/2502.18846_RL-OGM-Parking_Lidar_OGM-Based_Hybrid_Reinforcement_Learning_Planner_for_Autonomous_Parking.pdf`
- `docs/research/2026-06-17-rl-ogm-paper-code-research.md`
- `docs/superpowers/plans/2026-06-17-rl-ogm-parking-integration.md`
- `task_plan.md`, `findings.md`, `progress.md`
- `docs/research/2026-06-20-stage3-post-fast-action-mask-optimization-handoff.md`
- Current source files under `src/` and Stage 3 tooling under `tools/stage3/`

## Current Boundary

Do not implement OGM-specific source code yet. The repository instructions still say OGM work belongs after Stages 1-3 are complete and after the user explicitly approves Stage 4. The current code state also has an active fast action-mask 20K candidate run, so OGM work should wait until that candidate is closed and reported.

The useful result of this pass is therefore a Stage 4-ready integration design: where OGM should attach, what should remain invariant, what tests/gates should exist, and which earlier OGM plan details need updating.

## Paper-Level Reading

### HOPE Invariants To Preserve

The HOPE paper is not just "SAC plus parking." Its core method is:

- hybrid RL policy plus Reeds-Shepp policy
- action-mask mechanism
- transformer-style fusion of multiple observation tokens
- difficulty-ranked Normal/Complex/Extreme and DLP-like scenario diversity
- evaluation by scene success rates and path-planning behavior

The paper states four state inputs: obstacle distance vector, target position vector, action mask vector, and BEV image. Its ablation results make the risk hierarchy clear:

- Removing action-mask post-processing is severe, especially for complex parallel and extreme cases.
- Removing action-mask input is also damaging.
- Removing BEV is much less damaging for SAC in the reported ablation table.
- Restricting training difficulty hurts generalization to hard cases.

This means the first OGM integration should treat BEV/image representation as replaceable or augmentable, but should not remove action-mask behavior, target representation, RS switching, or difficulty scheduling.

### RL-OGM Contribution

The RL-OGM paper keeps the same broad hybrid planning idea: a rule-based RS planner plus an RL planner using SAC, with an action mask for safer and more efficient exploration. Its new contribution is the perception representation:

- Multiple LiDAR frames plus IMU are filtered and processed through LIO.
- Training uses a global point-cloud map splatted into a global OGM.
- Inference uses recent frames fused into a local OGM.
- The planner consumes OGM plus the target position.
- The purpose is sim-to-real alignment: the policy sees OGM-like inputs in both simulation and real deployment.

The PDF does not specify enough implementation detail to exactly reproduce grid resolution, channels, unknown/free-space encoding, network topology, or dataset format. A local implementation must therefore be explicit about assumptions and should start with a proxy OGM generated from the current simulator geometry.

### Metrics To Add

RL-OGM simulation metrics:

- PSR: parking success rate
- ANGS: average number of gear shifts
- PL: path length on successful attempts

RL-OGM real-vehicle metrics:

- PSR
- AOT: average operation time
- ANGS

The current `eval_utils.py` already records success, step count, reward, and path length. It does not record gear shifts or AOT-style timing.

## Current Code Map

### Environment And Observation

`src/env/car_parking_base.py` is the central attachment point:

- `CarParking.__init__()` accepts `use_lidar_observation`, `use_img_observation`, and `use_action_mask`.
- `render()` currently returns `img`, `lidar`, `target`, and `action_mask`.
- `_get_img_observation()` uses Pygame rotate/crop and `Obs_Processor` to produce a 64x64 RGB-like BEV tensor.
- `_get_lidar_observation()` derives a 120-ray distance vector from Shapely obstacle geometry.
- `_get_targt_repr()` returns a 5-vector target representation, but currently uses `cos(rel_dest_heading)` twice. This should be investigated separately; do not combine that bugfix with first OGM work.
- `find_rs_path()` and `is_traj_valid()` implement RS feasibility probing.

`src/env/env_wrapper.py` transposes only `img` from HWC to CHW. Any OGM observation must receive the same shape handling.

`src/env/lidar_simulator.py` and the Shapely map geometry are useful for proxy OGM generation. The geometry source is stronger than rasterizing from Pygame pixels because it avoids color semantics and matches collision/mask geometry.

### Map Sources

`src/env/parking_map_normal.py` generates bay and parallel parking cases from difficulty parameters.

`src/env/parking_map_dlp.py` loads `data/dlp.data`, which contains start, destination, obstacles, and possibly trajectory data. This is not a real OGM dataset, but it is the best local source for "real-world-like" geometry. Treat it as proxy OGM material, not as proof of the RL-OGM real-world dataset.

### Network

`src/model/network.py` defines `MultiObsEmbedding`:

- current token branches: lidar, target, action mask, optional image, optional critic action
- current image branch: `ImgEncoder`
- current fusion: attention token stack when `USE_ATTENTION=True`

This is a good match for an explicit `ogm` token because OGM can be encoded as another image-like modality. Reusing the `img` key is faster but blurs semantics and makes ablations harder.

`src/configs.py` currently controls observation toggles and model shapes:

- `USE_LIDAR=True`
- `USE_IMG=True`
- `USE_ACTION_MASK=True`
- actor/critic `n_modal = 2 + int(USE_IMG) + int(USE_ACTION_MASK)`
- `img_shape=(3,64,64)`

Future OGM work needs default-off config fields, for example `USE_OGM=False`, `OGM_SIZE=64`, `OGM_CHANNELS`, and `ogm_shape`.

### Agent, Replay, And State Normalization

`src/model/agent/sac_agent.py` mostly supports dynamic observation keys because `obs2tensor()` loops over `configs.observation_shape`. Adding `ogm` should work if every observation contains that key and config shapes match.

However, `src/model/state_norm.py` has `DEFAULT_UPDATE_MODAL = {'img':False, 'lidar':True, 'target':True, 'action_mask':False}`. If `ogm` is added to `observation_shape` without extending this map, state normalization will likely fail. OGM should default to `False` for running mean/std updates, like `img`.

`ReplayMemory` stores Python objects in deques. Adding 64x64 OGM tensors will increase memory and sample-copy cost. This is acceptable for a first smoke but should be measured before long OGM training.

### Training And Evaluation

`src/train/train_HOPE_sac.py` owns:

- scene scheduling (`SceneChoose`)
- DLP case scheduling (`DlpCaseChoose`)
- single-env transition loop
- replay warmup and SAC update cadence
- TensorBoard scalar names
- checkpoint naming
- final eval per scene

Do not edit this script for first OGM experiments. A separate OGM runner should reuse or extract behavior deliberately, preserving scalar names where possible.

`tools/stage3/` now contains manifest, launch, resource monitor, TensorBoard summary, parity, eval, and 20K comparison tooling. OGM should reuse this style rather than inventing an untracked ad hoc training run.

## Recommended Integration Strategy

### Phase 0: Finish Current Stage 3 Candidate

Before implementing OGM:

1. Close the current fast action-mask 20K candidate.
2. Export TensorBoard/resource/eval summaries.
3. Decide whether fast action mask passes, is speed-neutral, needs investigation, or fails.
4. Keep the command-only 20K baseline and 36.5K local baseline as the anchor for all OGM comparisons.

### Phase 1: Write Stage 4 OGM Spec And Gate

Create a Stage 4 design/spec before source changes. It should define:

- whether first OGM policy is `ogm + target + action_mask + hidden lidar-for-mask`
- the OGM grid assumptions
- metric definitions
- test gates
- baseline comparison targets
- artifact hygiene and checkpoint/log naming

Recommended first variant:

```text
policy input: target + action_mask + ogm
environment helper input: lidar remains available for current action-mask generation
disabled policy input: RGB BEV image
unchanged: reward, vehicle dynamics, RS switching, scene scheduler, DLP scheduler
```

This studies the OGM paper's perception idea without changing the HOPE planner contract.

### Phase 2: Proxy OGM Rasterizer

Add a new utility, likely `src/env/ogm.py`, only after Stage 4 approval.

Input:

- ego state
- obstacle LinearRings
- target box
- optional vehicle footprint or recent trajectory
- grid config

Output:

- local ego-centered occupancy tensor

Recommended initial grid:

- `size=64`
- resolution around `0.3125` to `0.3333` m/cell
- span about 20-21.3 m, matching the current 64px BEV crop and 10m lidar range better than a 0.2 m/cell grid

Recommended initial channels:

- channel 0: occupied obstacles
- channel 1: target slot or goal region, for HOPE-compatible local training
- channel 2: ego footprint or recent trajectory, only if ablated and documented

Strict RL-OGM later should test occupancy-only OGM plus target vector, because the paper describes OGM plus target rather than target embedded only in the grid.

Unit tests:

- obstacle cells are marked
- target channel is marked
- ego-centered transform rotates/translates correctly
- out-of-grid polygons clip safely
- output range and dtype are stable

### Phase 3: Optional Environment Observation

Add `use_ogm_observation=False` to `CarParking.__init__()`.

Keep defaults unchanged. When enabled:

- add `observation_space['ogm']`
- compute OGM from geometry during `render()`
- return `ogm` alongside existing keys
- transpose `ogm` in `CarParkingWrapper`
- include `ogm` in `observation_shape`

Important: do not remove `lidar` yet. In the first OGM variant, lidar can remain as a helper for existing action-mask generation while being excluded from the policy input if desired. A later strict variant can derive the action mask from OGM ray-casting.

Parity tests:

- with `use_ogm_observation=False`, all old observation keys and values match current behavior
- with OGM enabled, old keys still match
- OGM is deterministic under fixed reset/action sequences

### Phase 4: Explicit OGM Network Branch

Add `ogm_shape` to actor and critic config dictionaries, default `None`.

In `MultiObsEmbedding`:

- add `self.use_ogm`
- add an image-like OGM encoder branch
- add OGM token to the attention stack
- include OGM branch in orthogonal initialization

Do not overload `img` unless only doing a throwaway smoke. Explicit `ogm` is better for checkpoint compatibility, ablation, and future strict OGM work.

Config traps:

- `n_modal` must include OGM only when active
- `StateNorm.DEFAULT_UPDATE_MODAL` needs `ogm: False`
- actor/critic configs must be copied before per-run modification; do not mutate global `ACTOR_CONFIGS`/`CRITIC_CONFIGS` in a way that leaks between runs

### Phase 5: OGM Runner And Evaluation

Create an opt-in Stage 4 runner instead of editing `train_HOPE_sac.py`.

Reasonable locations:

- reusable source: `src/env/ogm.py`, optional config fields, network support
- experiment orchestration: `tools/stage4/` or `tools/ogm/`
- training entry point: `src/train/train_RL_OGM_sac.py` only if clearly separated from HOPE
- launch/report wrappers: mirror `tools/stage3/launch_stage3_20k.ps1` and `compare_stage3_20k.py`

The OGM runner should preserve:

- `SceneChoose` behavior or a shared scheduler module
- `DlpCaseChoose` behavior
- warmup rule and update cadence unless explicitly changed
- TensorBoard scalar names for existing metrics
- checkpoint naming and evaluation directories

Add OGM metrics:

- gear-shift count by sign changes in commanded speed, ignoring zeros
- path length on successful attempts
- AOT as simulated time (`step_num * 0.5s`) and optionally wall-clock time, clearly labeled

### Phase 6: Training Ladder

Use the same discipline as Stage 3:

1. unit tests
2. scripted environment parity
3. tiny smoke only for command/runtime
4. 1K smoke for obvious non-finite and plumbing checks
5. 20K OGM candidate gate
6. 36.5K local equivalence gate if 20K is promising
7. 100K paper-scale run only after earlier gates

For the first 20K OGM run, compare on two axes:

- HOPE-axis: Normal/Complex/Extrem/DLP success rates, TensorBoard health, step_num, reward, wall-clock throughput
- OGM-axis: PSR, ANGS, PL, and AOT-style timing

Do not claim RL-OGM real-world reproduction without real OGM maps, LiDAR/IMU logs, and deployment data.

### Phase 7: Real OGM Dataset Adapter

Only after proxy OGM training works, define a dataset adapter.

Suggested `.npz` schema:

```text
global_ogm: uint8 or float32, shape (H, W) or (C, H, W)
resolution: float32 scalar
origin: float32 shape (3,)
start: float32 shape (3,)
target: float32 shape (3,)
obstacle_polygons: optional object array, needed for exact collision checks
metadata: JSON string or object-like sidecar
```

The key open issue is collision and action mask. A pure occupancy grid is enough for perception input, but the current simulator still needs geometry for collision checks, lidar/mask generation, and RS path validation. If real OGM data lacks polygons, the adapter must either reconstruct approximate geometry or label the experiment as a different environment model.

## Approach Options

### Option A: Explicit OGM Modality With Hidden Lidar For Mask

Recommended first.

- Add `ogm` as a new observation key and network token.
- Disable `img` in the OGM policy variant.
- Keep lidar generation for current action-mask computation.
- Keep target vector and RS switching.

Pros: closest useful test of RL-OGM perception alignment while preserving HOPE behavior.

Cons: not a strict OGM-only planner because lidar still supports the mask.

### Option B: Reuse `img` Key For OGM

Fastest prototype.

- Feed OGM through `img`.
- Avoid network changes at first.

Pros: fewer code edits.

Cons: hides semantics, confuses checkpoints/logs, makes ablations messy, and risks accidentally changing the original image path.

Use only for a throwaway prototype, not as the main integration.

### Option C: Strict OGM Planner

Later-stage research.

- Policy input is OGM plus target.
- Action mask is derived from OGM ray-casting.
- No policy lidar input.
- Potentially no simulator-rendered BEV.

Pros: closest to RL-OGM paper intent.

Cons: highest risk because OGM-to-mask and OGM-to-collision semantics can diverge from Shapely geometry. Needs separate parity and quality gates.

## Main Risks

| Risk | Why It Matters | Mitigation |
|---|---|---|
| OGM paper underspecifies grid details | Exact reproduction from PDF alone is impossible | Document grid assumptions and run ablations |
| DLP is geometry, not real OGM | Cannot reproduce real-world RL-OGM numbers | Label as DLP proxy; add dataset adapter later |
| StateNorm key mismatch | Adding `ogm` can break normalization | Add `ogm: False` to update-modal defaults |
| Hidden mutation of global configs | Actor/critic config dicts are globals | Copy config dicts per runner before mutation |
| OGM memory/copy overhead | Replay stores Python objects and images | Measure 1K/20K throughput; consider later tensorized replay |
| Mask semantics drift | HOPE and RL-OGM both rely on action masks | Keep current lidar-derived mask first; later OGM mask parity |
| Target representation bug | Current code repeats `cos(rel_dest_heading)` | Investigate separately; do not combine with first OGM pass |
| Training distribution drift | Scene scheduling is a core HOPE result driver | Reuse scheduler and report scene mix |
| Baseline confusion | OGM can be slower while better sim-to-real | Report HOPE-axis and OGM-axis metrics separately |

## Concrete Future Implementation Sketch

After explicit Stage 4 approval:

1. Add tests for OGM rasterization and default-off behavior.
2. Add `src/env/ogm.py`.
3. Add default-off OGM config fields.
4. Add optional `ogm` observation in `CarParking`.
5. Add wrapper transposition and state norm support.
6. Add explicit OGM branch to `MultiObsEmbedding`.
7. Add metric helpers for gear shifts and AOT-style timing.
8. Add opt-in OGM SAC runner and evaluator.
9. Add Stage 4 manifest/report tooling based on Stage 3.
10. Run smoke, parity, 20K gate, then decide whether to pursue 36.5K or 100K.

## Recommendation

The best integration path is not to "port the OGM paper" wholesale. The current HOPE repository already contains the hybrid RL/RS planner, action mask, scene generators, DLP geometry, SAC training loop, and multimodal attention network. The OGM paper should be integrated as a new perception representation and dataset path:

```text
HOPE planner contract + OGM observation branch + OGM metrics + opt-in Stage 4 runner
```

The first approved OGM experiment should be a proxy OGM SAC variant that keeps HOPE reward, action, terminal, RS, action-mask, and scene scheduling semantics unchanged. Only after that variant is measurable should the project attempt strict OGM-only mask derivation or real OGM dataset ingestion.
