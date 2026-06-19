# PRD: HOPE Stage 3 Safe-Speed 20K Experiment Framework

Date: 2026-06-19

## 1. Purpose

Build a safe, repeatable experiment framework for accelerating original HOPE Stage 3 training while preserving training quality and the author's core training design.

The framework must make it easy to:

- measure the current training pipeline precisely;
- prove candidate changes preserve HOPE semantics before training;
- run 20,000-episode validation tests;
- compare every candidate against the locked command-only 20K baseline;
- reject changes that are faster only because they altered action mask, curriculum, observation, reward, environment, or hybrid-planner semantics.

This PRD is based on `docs/research/2026-06-19-stage3-safe-speed-deep-research.md`.

## 2. Baseline

All future Stage 3 speed tests covered by this PRD compare against the current command-only 20K baseline.

Baseline identity:

- Run directory: `D:\Github\HOPE\src\log\exp\sac_20260619_004316`
- Stop checkpoint: `D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt`
- Stop episode snapshot: `20038`
- Environment steps: `1824904`
- Wall time: `12.964 h`
- Throughput: `1545.71` episodes/hour, `39.10` environment steps/second
- Resource profile: average process CPU `37.76%`, average whole-GPU utilization `29.42%`

Baseline TensorBoard snapshot:

| Signal | Latest | Recent Mean |
|---|---:|---:|
| `avg_reward` | `0.076828` | `0.061052` |
| `actor_loss` | `-0.471469` | `-0.543129` |
| `critic_loss` | `0.080817` | `0.103880` |
| `success_rate_Normal` | `1.000000` | `1.000000` |
| `success_rate_Complex` | `0.940000` | `0.940000` |
| `success_rate_Extrem` | `0.690000` | `0.710800` |
| `success_rate_dlp` | `0.820000` | `0.792400` |
| `step_num` | `52.000000` | `77.220000` |

Baseline 200-episode external eval:

| Scene | Success |
|---|---:|
| Normal | `0.985` |
| Complex | `0.945` |
| Extrem | `0.655` |
| DLP | `0.960` |
| Mean | `0.88625` |

## 3. Non-Negotiable Boundaries

The product must preserve these HOPE invariants:

- Action mask semantics.
- Normal, Complex, Extrem, and DLP scene scheduling.
- DLP case scheduling.
- Hybrid RL plus Reeds-Shepp switching behavior.
- Observation semantics: `img`, `lidar`, `target`, `action_mask`.
- Action scaling, reward shaping, terminal status, timeout, collision, and environment dynamics.
- Normal, Complex, Extrem, and DLP evaluation KPI surface.

The product must not:

- introduce OGM-specific code or observations;
- remove or weaken the action mask;
- flatten or replace curriculum/scene scheduling;
- change reward, terminal, vehicle, collision, or observation semantics;
- silently replace the original training framework;
- optimize for CPU/GPU saturation as an end goal.

Parameter tuning is allowed only as a controlled experiment. Batch size, replay size, update cadence, mini epochs, learning rates, model capacity, and parallel collection are not red-line violations by themselves, but they must pass the same 20K quality gates before being considered useful.

## 4. Scope

### In Scope

MVP scope:

- Measurement pack for original and candidate runs.
- Deterministic parity harness for semantic-preserving candidates.
- Experiment launcher/manifest format for 20K tests.
- TensorBoard/resource/eval summary comparison against the locked 20K baseline.
- Gate report that classifies each candidate as pass, investigate, or reject.
- Candidate-entry rules for environment micro-optimizations and training-parameter sweeps.

### Out Of Scope

- Running or implementing OGM adaptation.
- Rewriting the original training algorithm.
- Committing training artifacts or generated checkpoints.
- Replacing `src/train/train_HOPE_sac.py` as the only available training path.
- Treating 1K tests as training-quality evidence.
- Declaring equivalence to 36.5K or 100K from a 20K test.

## 5. Users And Jobs

Primary user: local researcher developing HOPE incrementally.

Jobs:

- Start a candidate experiment without accidentally modifying original HOPE semantics.
- Know whether a candidate is actually faster at 20K.
- Know whether a speed gain came with quality regression.
- Inspect which subsystem is the bottleneck before changing source code.
- Preserve a comparable paper/code baseline before OGM work begins.

## 6. Product Requirements

### R1. Experiment Manifest

Each run must produce a machine-readable manifest containing:

- run name and timestamp;
- baseline id;
- git branch and `git status -sb` snapshot;
- source-diff check for `src/train`, `src/env`, and `src/model`;
- command;
- Python executable;
- environment variables;
- run directory;
- stdout/stderr paths;
- resource CSV path;
- TensorBoard summary JSON/Markdown paths;
- checkpoint paths expected at 20K;
- eval result paths;
- candidate type and changed knobs;
- gate status.

The manifest must make it obvious whether a run belongs to:

- baseline reference;
- measurement-only diagnostic;
- parity test;
- 1K speed smoke;
- 20K validation;
- rejected/incomplete run.

### R2. Measurement Pack

The measurement pack must time these components without changing training outputs:

- `env.reset`
- `env.step`
- render total
- image capture and image processing
- lidar observation
- action-mask calculation
- Reeds-Shepp path probing
- `ParkingAgent.get_action`
- replay push
- replay sample
- SAC update
- TensorBoard logging
- checkpoint save
- final evaluation

Required outputs:

- total seconds per component;
- call count;
- average milliseconds per call;
- percentage of measured wall time;
- environment steps/second;
- updates/second;
- notes on whether the measurement was taken during warmup or post-warmup.

Measurement-only diagnostics may use fewer than 20K episodes, but must not be used as training-quality evidence.

### R3. Parity Harness

Any candidate that claims to preserve environment semantics must pass deterministic parity before training.

The harness must compare original and candidate behavior across:

- Normal, Complex, Extrem, and DLP;
- multiple seeded reset cases;
- scripted action sequences;
- short episodes and near-goal RS-probe states when possible.

Required comparisons:

- `img`
- `lidar`
- `target`
- `action_mask`
- reward value
- terminal status
- `path_to_dest` presence
- step count until terminal state for scripted rollouts

Default pass rule:

- exact match for `action_mask`, reward, status, and `path_to_dest` presence;
- exact or explicitly justified numerical tolerance for `img`, `lidar`, and `target`;
- no unexplained difference in terminal timing.

### R4. 20K Validation Workflow

Each candidate accepted into validation must run to at least the nearest 20K checkpoint using the same isolated `.venv`.

Required command properties:

- headless Windows-compatible execution;
- `SDL_VIDEODRIVER=dummy`;
- `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`;
- command-only flags unless the candidate explicitly tests those flags: `--visualize= --verbose=`;
- `--eval_episode 200` for external checkpoint evaluation.

Validation stop point:

- training reaches around 20K TensorBoard episodes;
- `SAC_19999.pt` exists or an explicitly equivalent 20K checkpoint exists;
- training, monitor, and launcher are stopped cleanly after artifacts are preserved.

### R5. 20K Comparison Report

Each candidate must produce a report under `docs/research/` containing:

- baseline id and candidate id;
- exact command;
- changed knobs;
- parity result if applicable;
- run duration;
- episodes/hour;
- environment steps/second;
- process CPU and whole-GPU trends;
- TensorBoard scalar comparison;
- 200-episode eval comparison;
- pass/investigate/reject decision;
- whether it is eligible for a later 36.5K or 100K run.

### R6. Candidate Admission Rules

Candidate classes:

1. Measurement-only tooling.
2. Operational run-mode changes.
3. Semantics-preserving environment micro-optimizations.
4. Training-parameter sweeps.
5. Parallel collection or architecture changes.

Admission rules:

- Measurement-only tooling does not require parity but must not alter training output.
- Operational run-mode changes require command/logging equivalence rationale.
- Environment micro-optimizations require parity before 1K or 20K tests.
- Training-parameter sweeps require clear changed knobs and cannot be combined on first pass.
- Parallel collection requires a separate design because it can alter replay and curriculum distribution.

First recommended candidate sequence:

1. Measurement pack.
2. Broader render-mode/action-mask/reward/status parity suite.
3. One environment hot-path optimization selected from profiler evidence.
4. Batch-size sweep only after measurement clarifies update/replay cost.

## 7. Gate Criteria

### Hard Reject

Reject a candidate regardless of speed if:

- actor, critic, alpha, or action-std contains non-finite values;
- expected 20K checkpoint is missing;
- Normal or Complex scene collapses severely;
- multiple scenes regress together;
- speed gain is caused by changed observation, reward, action mask, RS behavior, curriculum, or environment dynamics;
- parity harness fails without an approved explanation.

### Quality Pass For 20K Candidate

A candidate passes the 20K quality gate when:

- external eval mean success is at least `0.85625`, which is baseline mean `0.88625 - 0.03`;
- Normal success is at least `0.95`;
- Complex success is at least `0.90`;
- Extrem success is at least `0.58`;
- DLP success is at least `0.91`;
- TensorBoard losses remain finite;
- TensorBoard scene success does not show a new multi-scene collapse pattern;
- `step_num=200` timeout samples, if present, are not paired with reward and scene-success collapse.

Extrem is allowed a wider tolerance than Normal/Complex because the current 20K baseline itself is immature on Extrem. This tolerance is not permission to ignore an Extrem collapse.

### Speed Pass For 20K Candidate

A candidate passes the speed gate when it also achieves at least one of:

- wall-clock time to 20K improves by `10%` or more;
- environment steps/second improves by `10%` or more;
- episodes/hour improves by `10%` or more with no evidence that the gain came from poorer policy quality or shorter failed episodes.

Primary speed metric is wall-clock time to 20K. Environment steps/second is the secondary throughput metric. Episodes/hour is useful but must be interpreted with `step_num` and success rates because better or worse policies change episode length.

### Decision Labels

- `pass`: quality gate and speed gate both pass.
- `quality-pass-speed-neutral`: quality passes, speed gain is below threshold.
- `investigate`: minor quality or parity ambiguity requires another run or deeper analysis.
- `reject`: hard reject or clear quality failure.

## 8. Data Flow

Experiment flow:

1. User approves a candidate.
2. Launcher records manifest and verifies protected source diff.
3. Optional parity harness runs if candidate touches environment behavior.
4. Optional 1K smoke checks plumbing and early speed.
5. 20K validation runs with resource monitor and TensorBoard.
6. Stop at the 20K checkpoint.
7. TensorBoard summary is exported.
8. External eval runs with 200 episodes per scene.
9. Report compares candidate to baseline and assigns decision label.
10. Planning files are updated with the result.

## 9. Error Handling

The framework must handle:

- monitor process exits while training continues;
- launcher PID differs from workload child PID;
- TensorBoard run directory ambiguity;
- missing checkpoint at expected episode;
- partial runs stopped by user;
- evaluation failure after training artifacts are preserved;
- non-finite metrics;
- stale or wrong baseline selection.

Required behavior:

- preserve existing logs and checkpoints;
- restart resource monitor only when safe;
- never summarize by newest `sac_*` directory without a locked run directory;
- mark incomplete runs explicitly;
- never claim a quality result from a partial or diagnostic run.

## 10. Deliverables

Future implementation should produce:

- experiment launcher or wrapper under `tools/stage3/`;
- measurement/profiling tool under `tools/stage3/`;
- parity harness under `tools/stage3/`;
- candidate comparison/report generator under `tools/stage3/`;
- report templates under `docs/research/`;
- updated root planning files after each candidate run.

This PRD itself does not require modifying original HOPE files under:

- `src/train`
- `src/env`
- `src/model`

## 11. Acceptance Criteria For The Framework

The framework is accepted when:

- it can run a measurement-only diagnostic without changing protected source files;
- it can run parity checks for environment-semantics candidates;
- it can launch a 20K candidate run with locked metadata;
- it can stop/report a 20K run and compare it to the baseline;
- it can classify candidate status using the gate labels;
- it records enough evidence for a future 36.5K or 100K decision;
- `git diff -- src/train src/env src/model` remains empty unless a later user-approved implementation plan explicitly allows otherwise.

## 12. Open Decisions For Implementation Planning

These are intentionally deferred to the implementation plan:

- exact profiler implementation method;
- exact parity seed/action case list;
- exact experiment manifest JSON schema;
- whether candidate wrappers should monkeypatch at runtime or use opt-in copied entry points;
- whether PRD-defined reports should be generated by one script or several smaller tools.

The implementation plan must answer these before code changes begin.
