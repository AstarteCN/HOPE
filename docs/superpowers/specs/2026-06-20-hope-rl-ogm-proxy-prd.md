# HOPE RL-OGM Proxy Integration PRD

Date: 2026-06-20
Status: design approved in brainstorming; awaiting implementation plan
Repository: `D:\Github\HOPE`

## 1. Background

The current repository is the user's fork of HOPE. Stage 1 through Stage 3 work established an isolated local environment, validated author checkpoints, retrained local HOPE baselines, and completed a safe-speed milestone. The accepted local speed baseline is the `hope-fast-action-mask-20k` run:

- Run directory: `src/log/exp/sac_20260620_085208`
- Checkpoint: `SAC_19999.pt`
- Wall time to 20K checkpoint: `10.947863 h`
- Throughput: `1826.840564 episodes/hour`, `46.223481 env steps/second`
- Matched 200-episode eval: Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`, mean `0.88625`
- User acceptance: speedup and 20K training quality accepted as the current local baseline

The next stage is to integrate the core contribution of the RL-OGM-Parking paper into HOPE while keeping HOPE as the base planner and preserving its verified local baseline.

## 2. Sources

This PRD is based on:

- `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`
- `raw_paper/2502.18846_RL-OGM-Parking_Lidar_OGM-Based_Hybrid_Reinforcement_Learning_Planner_for_Autonomous_Parking.pdf`
- `docs/research/2026-06-20-rl-ogm-integration-current-code-research.md`
- `docs/research/2026-06-20-stage3-fast-action-mask-20k-candidate-report.md`
- `docs/research/2026-06-20-stage3-post-fast-action-mask-optimization-handoff.md`
- Current HOPE source under `src/`
- Existing Stage 3 manifest, monitor, comparison, and reporting tools under `tools/stage3/`

## 3. Product Goal

Build a Stage 4 OGM experiment path that implements the RL-OGM paper's core simulation-side technical idea inside the HOPE project:

1. Add an explicit OGM observation modality to HOPE.
2. Train a hybrid RL parking policy using `ogm + target + action_mask` as policy inputs.
3. Preserve HOPE's hybrid RL plus Reeds-Shepp design, action-mask behavior, curriculum and scene scheduling intent, reward/action/terminal semantics, and local training framework boundaries.
4. Evaluate with simulation KPIs aligned to the RL-OGM paper.
5. Reach or approach the RL-OGM paper's simulation KPI values within the accepted tolerance and training budget.

## 4. Non-Goals And Hard Boundaries

This PRD does not include:

- Real vehicle testing.
- Real-world OGM dataset KPI acceptance unless a usable real-world OGM dataset is provided later.
- Replacing HOPE's action mask.
- Removing the internal lidar path needed by the current HOPE action-mask implementation.
- Changing HOPE's reward function, action semantics, terminal conditions, vehicle dynamics, Reeds-Shepp hybrid planner semantics, or curriculum and scene scheduling strategy as part of OGM v1.
- Fixing the current target representation bug in `CarParking._get_targt_repr()` where the final pair is `cos(phi), cos(phi)` instead of `cos(phi), sin(phi)`. This remains a known risk and future independent experiment.
- Treating speed as a hard success criterion. The accepted fast action-mask 20K run is the local throughput baseline, but OGM success is judged primarily on simulation KPI quality.

OGM source implementation must not begin until this PRD is approved by the user and converted into an executable implementation plan.

## 5. Accepted Design Decisions

The user approved these decisions during brainstorming:

| Decision | Accepted choice |
| --- | --- |
| Overall route | Proxy OGM First with strict simulation KPI gates |
| Policy inputs | `ogm + target + action_mask` |
| RGB BEV image | Disabled for the OGM policy path |
| Lidar | Kept internally for existing action-mask generation; not a policy token in OGM v1 |
| Hard evaluation set | New OGM-style fixed simulation set: `20 parallel + 50 perpendicular` |
| Compatibility metrics | Continue reporting HOPE Normal and Complex |
| Simulation KPI tolerance | Relative `+/-3%` against RL-OGM paper values |
| Training budget | `100K +/-20%` episodes, so `80K` to `120K` episodes |
| Real vehicle KPI | Excluded from current acceptance |
| Real-world OGM dataset KPI | Excluded unless a usable dataset becomes available |
| Target representation bug | Do not fix in OGM v1; record as risk |

## 6. KPI Targets

Hard acceptance uses simulation metrics from the RL-OGM paper's reported Hybrid RL results.

| Eval split | Target PSR | Target ANGS | Target PL |
| --- | ---: | ---: | ---: |
| Sim-Normal | `99.33%` | `1.5` | `20.3 m` |
| Sim-Complex | `97.7%` | `1.9` | `23.6 m` |

Acceptance window:

- PSR, ANGS, and PL must each land within relative `+/-3%` of the corresponding target for the hard gate.
- If a metric has stochastic variance across seeds, the implementation plan should define whether the gate uses one final run, repeated seeds, or confidence intervals before training starts.
- If a metric's definition is ambiguous in the RL-OGM paper, the local definition must be documented in the Stage 4 report before the run is interpreted.

Compatibility reporting:

- Report HOPE Normal and Complex scene success rates using the existing local evaluation style.
- Report speed and resource trends versus the accepted `hope-fast-action-mask-20k` baseline.
- These compatibility metrics support diagnosis but do not replace the OGM simulation KPI gate.

## 7. Architecture

The OGM path should be added as a first-class, default-off experiment path.

### 7.1 Proxy OGM Rasterizer

Add a focused OGM utility, expected location:

`src/env/ogm.py`

Responsibilities:

- Convert current simulator geometry into an ego-centered occupancy-grid tensor.
- Use map obstacle geometry, ego pose, and target/parking-slot geometry.
- Produce deterministic tensors under fixed scenario state.
- Keep assumptions explicit and configurable.

Initial grid:

- Size: `64 x 64`
- Resolution: approximately `0.3125` to `0.3333 m/cell`
- Span: approximately `20` to `21.3 m`

Initial channels:

- Required channel 0: occupied obstacles
- Required channel 1: target slot or goal region
- Optional channel 2: ego footprint or recent trajectory, only as a documented ablation and not as the default hard-gate input

The first strict-OGM policy should still use the target vector, because the RL-OGM paper describes OGM plus target rather than target embedded only in the grid.

### 7.2 Environment And Wrapper Integration

Extend `CarParking` with a default-off flag such as `use_ogm_observation=False`.

When disabled:

- The original HOPE observation behavior must remain unchanged.
- Existing training and evaluation entry points should continue to run without an `ogm` key.

When enabled:

- `render()` returns `observation['ogm']`.
- The existing `target` and `action_mask` keys remain present.
- Lidar may remain present internally or in observation data if required by existing action-mask code, but the OGM policy path should not include lidar as a policy token.
- `CarParkingWrapper` handles OGM tensor layout consistently, using channel-first layout before SAC consumes it.

### 7.3 Config, State Normalization, And Agent Input

Add explicit OGM config fields, default off:

- `USE_OGM=False`
- `OGM_SHAPE` or equivalent
- `ogm_shape`
- `observation_shape['ogm']` only when enabled

State normalization must include:

- `ogm: False`

This mirrors image behavior and avoids running mean/std updates over raster grids.

`SACAgent.obs2tensor()` likely supports the new key through `configs.observation_shape`, but this must be tested rather than assumed.

### 7.4 Network

Extend `MultiObsEmbedding` with an explicit OGM encoder branch.

Do not overload the existing `img` key or branch for OGM. Keeping OGM separate allows clean ablations and avoids silently mixing RGB BEV semantics with occupancy-grid semantics.

Default OGM token order:

1. `target`
2. `action_mask`
3. `ogm`
4. critic action token, only for critic paths

`n_modal` should be computed from enabled modalities, not patched by hand in a way that contaminates other experiments. Runner code should copy global config dictionaries before mutation.

### 7.5 Stage 4 Tooling

Create Stage 4 tooling in a new location such as:

`tools/stage4/`

Expected tools:

- OGM training launcher
- manifest writer
- resource monitor reuse or wrapper
- fixed OGM-style eval-set generator or loader
- checkpoint evaluator with PSR, ANGS, and PL
- TensorBoard summary exporter reuse
- 20K and final-run report generator

Do not directly edit `src/train/train_HOPE_sac.py` for OGM v1 unless the implementation plan proves there is no safe wrapper or adapter path.

## 8. Data Flow

Target OGM v1 flow:

```text
scenario geometry
  -> proxy OGM rasterizer
  -> CarParking observation['ogm']
  -> wrapper channel-first transform
  -> SAC obs2tensor
  -> MultiObsEmbedding ogm token
  -> SAC policy
  -> action-mask post-processing / HOPE action guard
  -> env.step
  -> Stage 4 metrics and reports
```

The policy should see `ogm + target + action_mask`. Lidar may still be computed for current HOPE action-mask generation, but this dependency must be named clearly in reports so the experiment is not misrepresented as a pure lidar-free policy stack.

## 9. Evaluation Data

Hard evaluation uses a new fixed OGM-style simulation set:

- `20` parallel parking cases
- `50` perpendicular parking cases

Requirements:

- Fixed seeds or persisted case metadata.
- Stable case IDs.
- Clear separation between training sampling and hard evaluation set where feasible.
- Report scenario construction parameters and difficulty assumptions.
- No real-world dataset gate unless a usable real-world OGM dataset is provided later.

The local `data/dlp.data` file is geometry-based DLP data, not a real OGM dataset. It may be used for compatibility or exploratory proxy analysis, but it must not be labeled as the RL-OGM real-world dataset.

## 10. Metrics

Required hard-gate metrics:

- PSR: parking success rate.
- ANGS: average number of gear shifts.
- PL: path length, preferably over successful attempts unless the implementation plan proves the paper used a different convention.

Required diagnostic metrics:

- Episode count.
- Environment step count.
- Reward trends.
- Actor loss, critic loss, alpha, action standard deviations.
- Scene success rates where available.
- Resource summary: process CPU, RAM, whole-GPU utilization, GPU memory.
- Throughput: episodes/hour and environment steps/second.

Gear-shift counting must be added to evaluation. The implementation plan should identify whether a gear shift is counted by sign changes in commanded motion, action mode, or vehicle kinematic state, then keep the definition fixed across all runs.

## 11. Testing Strategy

### 11.1 Unit Tests

Add tests for:

- OGM rasterizer output shape, dtype, value range, and channel count.
- Occupancy marking for simple obstacle polygons.
- Target slot marking.
- Ego-centered translation and rotation.
- Out-of-grid polygon clipping.
- Determinism under fixed input geometry.
- StateNorm handling of `ogm: False`.
- Network forward pass with `ogm + target + action_mask`.

### 11.2 Default-Off Parity

With OGM disabled:

- Original observation keys and shapes remain unchanged.
- Original action-mask outputs remain unchanged under seeded scripted cases.
- Existing Stage 3 tests still pass.
- No default training entry point silently switches to OGM.

### 11.3 Enabled-Path Smoke Tests

With OGM enabled:

- Reset and step return valid OGM observations.
- Wrapper emits channel-first tensors.
- Replay memory accepts OGM observations.
- SAC `obs2tensor()` accepts OGM observations.
- Actor and critic forward passes are finite.
- A tiny training run writes TensorBoard and checkpoint artifacts.

## 12. Training Ladder And Gates

The implementation plan should use this ladder:

1. Import and unit tests.
2. Scripted parity cases with OGM disabled.
3. OGM enabled reset/step smoke.
4. Tiny training smoke, sufficient only to prove plumbing.
5. 1K diagnostic run to check non-finite losses, logging, memory, and throughput.
6. 20K early gate against the accepted `hope-fast-action-mask-20k` local baseline.
7. 80K to 120K final OGM training run.
8. Final fixed-set evaluation against RL-OGM simulation KPI targets.

Hard reject conditions:

- Non-finite actor loss, critic loss, reward, or observation tensor values.
- Missing hard-gate checkpoint or malformed manifest.
- Broken action-mask behavior.
- Broken curriculum or scene scheduling intent.
- Severe multi-scene collapse in early 20K evaluation that cannot be explained by expected learning stage.

## 13. Reporting Requirements

Each Stage 4 run should produce:

- Manifest JSON.
- TensorBoard summary JSON and Markdown.
- Resource CSV and summary.
- Eval result JSON and Markdown.
- KPI comparison against RL-OGM targets.
- Compatibility comparison against HOPE Normal/Complex and accepted fast action-mask 20K baseline.
- Explicit note of whether real-world dataset KPI was excluded due to missing dataset.

Reports should live under:

`docs/research/`

Training logs, checkpoints, TensorBoard event files, and raw heavy artifacts must not be committed unless the user explicitly asks.

## 14. Risks

| Risk | Mitigation |
| --- | --- |
| RL-OGM paper does not fully specify grid resolution, channels, or encoding | Document proxy assumptions and make them configurable |
| Proxy OGM differs from real LiDAR/IMU-derived OGM | Exclude real-world dataset KPI until data exists; label current work as simulation/proxy OGM |
| Target representation bug may affect target semantics | Do not fix in OGM v1; record as independent future experiment |
| Internal lidar for action mask can be mistaken for policy input | Keep reports explicit: lidar is an internal action-mask helper, not an OGM policy token |
| Global config mutation can contaminate later runs | Copy config dictionaries in OGM runner before mutation |
| OGM tensors increase replay memory and sampling cost | Measure memory and throughput during 1K and 20K runs |
| Hard KPI may be sensitive to scenario mismatch | Persist fixed eval set and document construction assumptions |

## 15. Out-Of-Scope Future Work

These are valuable but outside OGM v1:

- Real-world OGM dataset ingestion.
- Real vehicle testing.
- Strict lidar-free action-mask replacement.
- Target representation bugfix and ablation.
- Full architecture search for OGM encoder designs.
- Multi-seed statistical reproduction study beyond the first final gate.

## 16. Approval State

The user approved the PRD direction section by section:

1. Goal and acceptance scope.
2. Functional scope and non-goals.
3. Architecture and data flow.
4. Testing, training ladder, and gates.
5. Deliverables, risks, and document location.

Next required step after user review of this written PRD:

Convert this PRD into an executable implementation plan with `superpowers:writing-plans`. Do not begin implementation before that plan is written and accepted.
