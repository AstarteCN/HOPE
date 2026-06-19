# Stage 3 Safe Speed Deep Research

Date: 2026-06-19

## Objective

This note studies how to make original HOPE Stage 3 training faster while preserving training quality and the author's training framework.

The goal is faster time-to-good-policy, not higher hardware utilization for its own sake. A change that fills the GPU but weakens the learned planner is a failed speedup.

This is research-only. It does not approve source changes under `src/train`, `src/env`, or `src/model`.

## Evidence Base

Local training evidence:

- Stopped original baseline: `src/log/exp/sac_20260617_233812`, stopped at `36502` TensorBoard episodes after `23.84 h`.
- Baseline 36.5K evaluation of `SAC_35999.pt`: Normal `1.000`, Complex `0.985`, Extrem `0.915`, DLP `0.955`, mean `0.96375`.
- Baseline resource profile: average process CPU `32.38%`, average whole-GPU utilization `25.76%`, peak GPU memory `3740 MB`.
- Command-only 20K smoke: `src/log/exp/sac_20260619_004316`, stopped at `20038` episodes after `12.964 h`.
- Matched 20K eval parity: command-only `SAC_19999.pt` exactly matched baseline `SAC_19999.pt` at Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`.

Local code evidence:

- `src/train/train_HOPE_sac.py` uses a single sequential environment loop with one `CarParking` environment and one `ParkingAgent`.
- `src/train/train_HOPE_sac.py` chooses among Normal, Complex, Extrem, and DLP through `SceneChoose`, then adapts scene choice after the first history window by mixing uniform sampling and worse-performing-scene sampling.
- `src/train/train_HOPE_sac.py` chooses DLP cases through `DlpCaseChoose`, with random exploration early and failure-weighted sampling later.
- `src/model/agent/parking_agent.py` switches between the RL agent and the Reeds-Shepp planner when a valid planner path is present.
- `src/env/car_parking_base.py` calls `render()` every environment step, then constructs image, lidar, target, and action-mask observations.
- `src/model/action_mask.py` precomputes parts of the mask, then uses the live lidar vector to estimate valid step sizes and post-process action probability.
- `src/model/agent/sac_agent.py` defaults to `memory_size=10240`, `batch_size=32`, `mini_epoch=1`, and performs one sampled SAC update per call.
- `src/model/replay_memory.py` uses Python deques and samples random indices into lists before converting batches for the update.

Paper evidence from `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`:

- HOPE is presented as a hybrid policy/path planner combining an RL policy with Reeds-Shepp curves.
- The paper describes four state inputs: obstacle distance/lidar, target pose, action mask, and BEV image.
- The paper treats the action mask as a mechanism that guides exploration and filters invalid actions.
- The paper describes a Reeds-Shepp switch strategy: the RS policy is activated near the target when a collision-free RS curve exists.
- The paper ranks parking scenarios by difficulty and evaluates Normal, Complex, Extreme, and real-world DLP scenarios.
- The paper reports `100,000` training episodes and `2,000` evaluation trials per scenario category.
- Appendix parameters include `Tmax=200`, RS threshold distance `10.0`, lidar dimension `120`, action-mask dimension `42`, and BEV image size `64 px`.

## Non-Negotiable HOPE Invariants

These are red lines for Stage 3 speed work. Breaking any of them means the result is no longer a faithful original-HOPE training comparison.

| Invariant | Why It Must Stay | Code Anchor |
|---|---|---|
| Action-mask semantics | The paper makes action masking part of HOPE's method. The code includes it in observation and in action selection. | `configs.py`, `car_parking_base.py`, `action_mask.py`, `sac_agent.py` |
| Scene curriculum and scene scheduling | The local code's adaptive Normal/Complex/Extrem/DLP scheduler determines the replay distribution. Flattening it would change training data. | `train_HOPE_sac.py::SceneChoose`, `DlpCaseChoose` |
| Hybrid RL + RS planner behavior | HOPE's method depends on RS assistance near the target and replay data collected from the hybrid policy. | `car_parking_base.py::find_rs_path`, `parking_agent.py` |
| Observation semantics | Lidar, target, BEV image, and action mask are the state representation used by the paper and model. | `configs.py`, `car_parking_base.py::render`, `env_wrapper.py` |
| Action and reward semantics | Steering/speed range, action rescale, terminal rewards, timeout, collision, and shaped rewards define the task. | `configs.py`, `env_wrapper.py`, `car_parking_base.py` |
| Evaluation scenes and KPI budget | Normal, Complex, Extrem, and DLP success rates are the comparison surface. | `eval_mix_scene.py`, baseline docs |

Parameter tuning and scale changes are not red-line violations by definition. They are acceptable only when treated as explicit, controlled experiments that keep the above task semantics intact and are judged by matched quality gates.

## Current Bottleneck Model

The best current explanation is environment/pipeline bound rather than pure GPU-update bound.

Macro evidence:

- The strong 36.5K baseline used only `25.76%` average whole-GPU utilization.
- CPU and GPU utilization fell as the policy improved and episodes shortened.
- The command-only 20K run improved early overhead and matched the 20K baseline exactly in external eval, but long-run episode/hour improvement was limited.

Micro-profile evidence:

| Component | Cost Signal |
|---|---:|
| `env.step_total` | `9.75 ms/call` |
| `env.render_total` | `6.19 ms/call` |
| `env.rs_probe` | `3.42 ms/call` when invoked |
| `agent.get_action` | `3.20 ms/call` |
| `env.action_mask` | `1.68 ms/call` |
| `agent.update` | `38.86 ms/update`, but called only every 10 env steps |

Interpretation:

- More GPU work can make utilization look better without increasing useful environment samples per hour.
- Environment render/observation/action-mask/RS work is the first place to measure more precisely.
- Replay sampling and SAC update still matter, especially if batch/update schedule is changed, but they are not the only wall-clock driver.

## Safe Acceleration Taxonomy

### Tier 0: Already Validated Parameter-Only Run Mode

Use the original script with:

```powershell
--visualize= --verbose=
```

This avoids the `argparse type=bool` trap where `--visualize False --verbose False` evaluate as true. The 20K matched eval showed no quality regression versus the baseline 20K checkpoint.

Status: safe as a run mode candidate, not a full 36.5K/100K equivalence proof.

### Tier 1: Measurement-Only Tooling

Before source-level optimization, add opt-in tooling outside original source paths to time:

- `env.reset`
- `env.step`
- `render`, image capture, image processing
- lidar observation
- action-mask calculation
- RS path probing
- `ParkingAgent.get_action`
- `ReplayMemory.sample`
- `SACAgent.update`
- TensorBoard logging
- checkpoint saves and final eval

This tier should not alter training outputs. It answers where time is actually spent after command-only overhead is removed.

### Tier 2: Launcher And Process Architecture

Safe engineering changes can live outside the original training files:

- Robust launcher that records run metadata and locks the TensorBoard run directory.
- Reliable child-PID resource monitor.
- Gate automation for 20K early warning, 36.5K equivalence, and 100K final comparison.
- Process/thread environment settings recorded per run.
- Optional stdout/stderr capture and external summary export.

These improve experiment control and reduce operator error without changing HOPE learning semantics.

### Tier 3: Semantics-Preserving Environment Micro-Optimizations

These require parity tests before training:

- Suppress or bypass display updates only if `img`, `lidar`, `target`, `action_mask`, reward, and status remain identical.
- Reuse or precompute lidar angle arrays and temporary buffers where outputs are exactly unchanged.
- Optimize action-mask temporary allocation or vectorization while keeping the same mask values for the same lidar input.
- Time and then optimize RS feasibility probing only if the same states produce the same `path_to_dest` decision.
- Replace generic deep copies with explicit safe copies only after aliasing tests.

Required parity gate:

- Same seeded reset cases across Normal, Complex, Extrem, and DLP.
- Same scripted action sequences.
- Compare observation tensors, action masks, reward values, terminal status, and `path_to_dest` presence.
- Any numerical tolerance must be declared before the check. Exact match is preferred for mask/reward/status.

### Tier 4: Training-Mode Experiments

These can speed time-to-good-policy but are algorithmic experiments, not guaranteed-safe engineering changes:

- Batch size sweep: `32 -> 64 -> 128`.
- Update ratio or update cadence sweep: one update per 10 env steps versus controlled alternatives.
- More than one mini epoch per update.
- Replay-buffer memory layout and batch tensorization.
- Frozen image-encoder feature caching in replay memory.
- Vectorized or multiprocess environment collection.

Rules:

- Do not combine multiple training-mode changes in one first experiment.
- Keep action mask, scene scheduler, reward, observation, and RS hybrid behavior intact.
- Preserve or explicitly model curriculum distribution if using multiple collectors.
- Use 1K only for speed/plumbing, 20K for early warning, 36.5K for local equivalence, and 100K for paper-scale comparison.

### Tier 5: High-Risk Algorithm Or Task Changes

Do not perform in Stage 3 speed work without a separate user-approved design:

- Remove or weaken action mask.
- Replace the curriculum or scene scheduler.
- Change reward shaping or terminal semantics.
- Disable RS hybrid switching.
- Remove BEV/lidar/action-mask observation fields.
- Change vehicle dynamics or collision rules.
- Introduce OGM observations or OGM-specific training code.
- Replace the training framework with a different RL library in a way that changes replay/curriculum semantics.

## Hyperparameter Position

The user clarified that scaling or tuning parameters is not automatically crossing a red line. I agree with that boundary.

However, parameter tuning splits into two categories:

Safe operational parameters:

- `--verbose=`
- `--visualize=`
- external resource-monitor interval
- final eval budget when clearly labeled
- TensorBoard summary export cadence

Controlled training experiments:

- batch size
- replay buffer size
- update frequency
- mini epochs
- learning rates
- entropy temperature behavior
- model hidden size or encoder capacity
- number of parallel collectors

The second category is allowed, but must be judged as training-methodology research. It can improve quality and speed, but it can also alter SAC dynamics, replay distribution, or exploration. Therefore it needs matched gates, not just speed measurements.

## Recommended Next Research-To-Implementation Path

### Step 1: Add A Measurement Pack

Create opt-in profiler tooling outside original source paths.

Output:

- per-component wall-clock timing
- environment steps/sec
- update calls/sec
- replay sample conversion time
- render/image/lidar/action-mask/RS split
- no changes to original training outputs

Decision gate:

- If environment/render/mask/RS dominates, optimize Tier 3 first.
- If update/replay dominates after command-only mode, test replay/update changes first.

### Step 2: Broaden Render-Mode Parity

The previous parity check found exact matches for scripted cases. Broaden it before treating `--visualize=` as default for all quality runs:

- more seeds
- all scene types
- both short and near-goal RS states
- repeated reset/action sequences

Decision gate:

- If parity remains exact, command-only mode can be the default launcher configuration.
- If not, use only `--verbose=` as the lowest-risk parameter-only change.

### Step 3: Try One Environment Micro-Optimization

Pick the highest measured semantics-preserving hot path. Good first candidates are display update suppression or reusable lidar/action-mask arrays.

Validation ladder:

1. deterministic parity suite
2. 1K speed smoke
3. 20K early-quality check
4. 36.5K local baseline equivalence

### Step 4: Run Controlled Hyperparameter Sweeps

Only after measurement clarifies whether update compute is actually limiting.

Suggested first sweep:

- baseline command-only: batch `32`
- batch `64`, same update cadence
- batch `128`, same update cadence

Do not change learning rates in the first sweep. If larger batch improves stability but slows wall time, decide based on time-to-quality rather than episodes/hour alone.

Suggested second sweep:

- keep batch fixed
- test update cadence or mini epoch
- compare by environment samples, wall time, and scene KPI

### Step 5: Consider Parallel Collection Only With Curriculum Preservation

Multiprocess collection is tempting but high risk because the current scheduler adapts from recent scene success. A safe design would need:

- one authoritative scene/case scheduler
- per-worker seed control
- identical `CarParking` semantics
- action mask calculated per worker exactly as before
- replay insertion preserving hybrid-policy samples
- monitoring of scene mix and success mix

This should not be the first implementation change.

## Quality Gates

Use the existing local baseline as the quality anchor.

20K early warning:

- Compare TensorBoard against `stage3_baseline_gate_references_20260619.json`.
- Run matched 200-episode eval when checkpoint exists.
- Treat Extrem lag as important but not automatically fatal if it matches baseline maturity.

36.5K equivalence:

- Compare against baseline `SAC_35999.pt`.
- Mean success within `0.05` absolute.
- No individual scene worse by more than `0.10`.
- Normal and Complex at least `0.95`.
- Extrem and DLP at least `0.85`.

100K paper-scale check:

- Compare candidate `SAC_99999.pt` and `SAC_best.pt` against author checkpoints.
- Use the same eval budget and command style.
- Report time-to-quality, not just total throughput.

Hard-stop conditions:

- non-finite actor/critic/alpha/action-std values
- missing expected checkpoint after save point
- severe multi-scene collapse
- reward improvement with held-out scene success collapse
- speed improvement caused by changed observation/reward/action-mask/RS/curriculum semantics

## Recommended Priority

1. Measurement pack outside original source paths.
2. Broader render/action-mask/reward/status parity suite.
3. One semantics-preserving environment hot-path optimization.
4. Controlled batch/update sweeps after profiling.
5. Parallel collection only after scheduler/replay semantics are explicitly designed.

This order keeps the red lines intact while still allowing meaningful acceleration research, including parameter scaling when it is measured and gated rather than assumed safe.
