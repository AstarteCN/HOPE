# Stage 3 Post Fast Action Mask Optimization Handoff

Date: 2026-06-20

Audience: main Stage 3 agent, after the current fast action-mask 20K candidate run finishes.

## Purpose

This handoff identifies worthwhile optimization experiments after the fast action-mask 20K candidate completes. The allowed scope is broader than the earlier protected-source work: future work may substantially restructure the author's code, training runner, replay/update pipeline, and training dynamics.

The scientific boundary stays fixed. A candidate must still preserve the HOPE paper's core task and method intent:

- action mask behavior and its use in action selection and network input
- Normal/Complex/Extrem/DLP scene curriculum, scene scheduling, and DLP case selection intent
- observation semantics: BEV image, lidar, target representation, action mask
- action/reward/terminal semantics and vehicle/environment dynamics
- hybrid RL policy plus Reeds-Shepp fallback/switching behavior
- final quality judged by scene success rates, TensorBoard health, and external evaluation

This document intentionally does not update root `task_plan.md`, `findings.md`, or `progress.md`, to avoid polluting the running agent's local memory state.

## Current Run Context

Current candidate, do not disturb while running:

- Manifest: `src/log/exp/stage3_fast_action_mask_20k_20260620_085206.meta.json`
- Run dir: `src/log/exp/sac_20260620_085208`
- Expected 20K checkpoint: `src/log/exp/sac_20260620_085208/SAC_19999.pt`
- Candidate knob: default-off fast `ActionMask.get_steps` enabled through `tools/stage3/train_HOPE_sac_fast_action_mask.py`
- Baseline comparison: command-only 20K run `src/log/exp/sac_20260619_004316`, checkpoint `SAC_19999.pt`
- Resource caveat: early resource CSV rows include the venv launcher parent PID; final resource summaries should exclude or caveat those startup rows.

When it finishes, first complete the existing 20K candidate report workflow before starting any new optimization.

## Evidence Base

Paper evidence from `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`:

- HOPE combines an RL policy with a Reeds-Shepp policy through a hybrid action switch.
- The RS policy is activated near the target only when a collision-free RS curve exists.
- The action mask is a method-level mechanism: it guides exploration, filters invalid action probability, and is also a model input.
- The paper uses four state inputs: lidar/obstacle distance, target pose, action mask, and BEV image.
- The paper's training/evaluation scale is 100,000 training episodes and 2,000 evaluation trials per scenario category.
- Appendix parameters align with local code intent: `Tmax=200`, RS threshold `drs=10.0`, lidar dimension `120`, action-mask dimension `42`, BEV image `64 px`.
- Ablations report large drops without action-mask post-processing or without sufficient difficulty diversity, so removing mask use or flattening difficulty curriculum is not a safe speedup.

Local source anchors:

- `src/train/train_HOPE_sac.py` owns `SceneChoose`, `DlpCaseChoose`, the single-env training loop, replay insertion, update cadence, TensorBoard logging, best checkpointing, and final eval.
- `src/env/car_parking_base.py` builds observations every step through Pygame rendering, image processing, lidar simulation, action-mask calculation, target representation, rewards, terminal status, and RS probing.
- `src/model/agent/parking_agent.py` switches between the RL action and the RS planner action.
- `src/model/agent/sac_agent.py` uses Python replay sampling, state normalization at push time, frozen pre-trained image encoders by default, and one SAC update every 10 environment steps after warmup.
- `src/model/replay_memory.py` stores deques of Python objects and samples via Python list construction before tensor conversion.

Existing performance evidence:

- Strong local baseline: 36.5K episodes in 23.84 h, external eval Normal `1.000`, Complex `0.985`, Extrem `0.915`, DLP `0.955`, mean `0.96375`.
- Command-only 20K: 12.964 h, `1545.71` episodes/hour, `39.10` env steps/sec, matched 20K eval parity with the baseline checkpoint.
- Detailed env-step profile after command-only: raw step about `9.6 ms`; render/observation about `6.6 ms`; image draw/capture/process about `3.9 ms`; action mask about `1.78 ms`; lidar about `0.65 ms`; RS probing is state-dependent.
- Fast action-mask bounded profile: action mask about `1.694 ms -> 0.821 ms`, env step about `9.345 ms -> 8.205 ms`; this is measurement evidence only until the 20K run passes.
- SAC update is nontrivial (`~38-58 ms/update`) but called only every 10 env steps; image encoder work is repeated several times per update because actor, critic, and target networks each process image batches.

## Rating Rubric

- Potential benefit:
  - Low: likely below 5% wall-clock or mostly operational cleanliness.
  - Medium: roughly 5-20% wall-clock or improves run control enough to reduce failed experiments.
  - High: roughly 20-50% wall-clock or clearly improves time-to-quality.
  - Very high: plausible multi-x improvement, usually with higher training-distribution risk.
- Risk:
  - Low: exact-output or near-exact implementation optimization with strong parity tests.
  - Medium: code architecture change that should preserve semantics but can introduce hidden aliasing, RNG, or checkpoint issues.
  - High: changes replay distribution, policy staleness, update ratio, batch noise, curriculum timing, or model inputs.
  - Very high: risks creating a different algorithm/task even if reward and scenes are nominally unchanged.

## Candidate Matrix

| Priority | Candidate | Potential Benefit | Risk | Why Worth Testing | Main Gate |
|---|---|---:|---:|---|---|
| P0 | Finish fast action-mask 20K report | Medium | Low-medium | Already running; first exact-output source optimization candidate. | Existing 20K candidate report and eval gates. |
| P0 | Build a new opt-in Stage3 training runner | Medium indirect | Medium | Enables large restructuring without mutating author script; centralizes scene scheduler, env, agent, replay, logging, checkpointing, and gates. | Same TensorBoard/eval output schema as original runner on command-only baseline. |
| P1 | Lidar/action-mask/target buffer reuse and precompute | Medium | Low-medium | Lidar code recomputes ray arrays and edge arrays per step; action mask already proved this style of exact optimization is viable. | Exact observation/mask/reward/status parity on scripted cases. |
| P1 | RS probing optimization with exact path-decision parity | Medium | Medium | Paper and local profiles both show RS can be expensive; near-target calls matter as policies improve. | Same `path_to_dest` presence and same planner action sequence for sampled near-goal states. |
| P1 | Tensorized replay ring buffer preserving normalized observations | Medium-high | Medium | Current replay is Python deque/list/deepcopy heavy; sampling and tensor conversion happen on every update. | Same batch shape/value semantics and no quality regression at 20K. |
| P1 | Frozen image-encoder feature cache | High | Medium-high | Image encoder is loaded frozen by default; update path repeatedly re-encodes the same state images across actor/critics/targets. | Feature-level numerical parity and checkpoint compatibility plan. |
| P2 | Direct/headless BEV image pipeline replacement | High | High | Image generation is the largest env-step sub-block. Replacing Pygame rotate/crop/tostring/cv2 with an equivalent NumPy/OpenCV renderer could be substantial. | Pixel or declared-tolerance parity across scenes, rotations, obstacles, and trajectory rendering. |
| P2 | Update compute optimization without changing dynamics | Medium | Low-medium | Use `torch.inference_mode` where valid, avoid repeated dict mutation/deepcopy, use compiled/static tensor paths if stable. | Same sampled batches, finite losses, no speed-only regression. |
| P2 | Controlled batch-size/update-cadence sweeps | Medium-high | High | Can improve GPU efficiency and time-to-quality, but changes SAC gradient noise and update-to-data ratio. | Treat as training-method experiment; require 20K early gate and 36.5K equivalence. |
| P3 | Synchronous multi-env collection with authoritative scheduler | Very high | High | Best route to multi-x throughput if CPU/env bound, but scene scheduling, state norm, replay order, and policy freshness all change. | New design doc plus scene-mix/replay-distribution checks before any 20K run. |
| P3 | Async actor/learner overlap | High | High | Could overlap CPU env stepping and GPU update, but action policy becomes stale relative to learner state. | Explicit policy-staleness budget and 36.5K equivalence gate. |
| P3 | Prioritized replay or curriculum-aware replay sampling | Medium-high | High | May improve time-to-quality, especially rare hard scenes, but changes the learning algorithm. | Research experiment only; compare time-to-quality, not just throughput. |

## Recommended Sequence After Fast Action Mask

### Step 1: Close The Current Candidate

If `SAC_19999.pt` exists and TensorBoard reaches 20K:

1. Export TensorBoard summary.
2. Run external 200-episode eval per scene.
3. Summarize resource CSV, caveating or excluding launcher-PID startup rows.
4. Run `compare_stage3_20k.py` against the command-only baseline.
5. Label the result `pass`, `quality-pass-speed-neutral`, `investigate`, or `reject`.

If the candidate fails quality, do not stack more changes on top of it. First compare action distributions, mask parity, and TensorBoard failure patterns.

If it passes, future candidates can be tested in two views:

- primary report against the locked command-only 20K baseline
- secondary additive report against fast-action-mask 20K, to understand incremental gain

### Step 2: Add A New Runner Before Bigger Rewrites

The original `train_HOPE_sac.py` is too coupled for larger experiments. Create a new opt-in runner under `tools/stage3/` or a new `experiments/stage3/` package that reuses HOPE components but owns orchestration.

Minimum modules:

- `SceneScheduler`: equivalent `SceneChoose` and `DlpCaseChoose` behavior, with explicit state export.
- `EnvFactory`: command-only render settings, fast-mask toggle, future observation-pipeline toggles.
- `HybridCollector`: one environment transition loop preserving `ParkingAgent` RS switching and replay insertion.
- `Learner`: SAC update policy, update cadence, batch config, state norm, and checkpointing.
- `RunLogger`: TensorBoard scalar compatibility, manifest, resource links, and checkpoint paths.
- `GateHooks`: 1K smoke, 20K early gate, 36.5K equivalence gate, final eval.

Acceptance for the runner itself:

- Command-only run with the new runner must reproduce original scalar names and checkpoint naming.
- A short deterministic parity harness should show same scene/case scheduling decisions for a controlled RNG stream.
- Protected original script can remain as reference; the new runner is the place for larger experiments.

### Step 3: Exhaust Exact Or Near-Exact Environment Optimizations

Best next low-risk options:

1. Precompute lidar ray angles/sin/cos arrays once in `LidarSimlator`.
2. Precompute per-reset obstacle edge arrays for lidar instead of rebuilding `x1s/x2s/y1s/y2s` every step.
3. Reuse temporary arrays where safe in lidar/action-mask calculations.
4. Optimize RS `is_traj_valid` by precomputing obstacle edge arrays per reset and removing avoidable Python list/heap overhead.
5. Profile `env.reset` and map generation; consider async pre-generation only if RNG/distribution accounting is explicit.

These should be tested with exact parity before any 20K run. They are smaller than image replacement but much easier to certify.

### Step 4: Attack Replay And Frozen Image Encoding

This is probably the highest-value architecture change that still preserves the paper's inputs.

Rationale:

- `UPDATE_IMG_ENCODE=False` by default, so the loaded autoencoder encoder is frozen.
- The actor and critics load frozen image encoders; update code re-encodes state and next-state image batches repeatedly.
- Replay stores raw Python objects, then reconstructs arrays and tensors at sample time.

Candidate design:

- Store normalized lidar/target and raw action mask exactly as today.
- Store image tensors plus optionally cached frozen image features.
- Add an embedding adapter that can consume either `img` or `img_feature`.
- Keep action selection compatible: current observation still contains image semantics, but the runner may compute/carry the frozen feature once.
- Preserve old checkpoint loading by adding adapter code rather than deleting the original image branch.

Risk controls:

- Compare cached encoder features against live encoder output with a tight tolerance on sampled observations.
- First optimize replay storage without feature caching, then add feature caching as a separate candidate.
- Report whether throughput improves in env steps/sec, update/sec, and wall time to 20K.

### Step 5: Treat Training-Dynamics Changes As Research, Not Safe Engineering

Allowed but risky experiments:

- batch size `32 -> 64 -> 128`
- update cadence, for example one update per 5 or 20 env steps
- multiple updates per collection interval
- delayed actor update or target update cadence changes
- replay size changes
- prioritized replay or hard-scene replay balancing

These keep the same task and HOPE framing but change SAC dynamics. Use one variable per candidate. A 1K run only checks plumbing and obvious divergence; 20K is an early warning gate; 36.5K is the local equivalence gate before claiming quality preservation.

### Step 6: Parallel Collection Only After The Scheduler Is Designed

Multiprocess/vectorized collection has the highest upside, but it is the easiest way to silently change training distribution.

A safe design needs:

- one authoritative scene scheduler issuing episode assignments
- explicit DLP case assignment and success feedback
- per-worker `CarParking` instances with identical action-mask/RS/observation behavior
- centralized state normalization or a frozen/statistically equivalent normalization design
- replay insertion metadata: scene, case id, worker id, policy version, RS-active flag
- policy versioning so action staleness is measurable
- scene-mix and success-mix reports compared against the single-env baseline

Do not start here unless lower-risk exact optimizations plateau.

## Test Ladder For Every Candidate

Use the smallest ladder that matches risk:

1. Unit parity: deterministic inputs and edge cases.
2. Scripted environment parity: fixed resets and actions across Normal, Complex, Extrem, and DLP; compare `img`, `lidar`, `target`, `action_mask`, reward, done/status, and `path_to_dest`.
3. Bounded profile: enough transitions to measure per-call costs, not quality.
4. 1K smoke: command/monitor/checkpoint plumbing only.
5. 20K gate: TensorBoard health, resource summary, external 200-episode eval, candidate report.
6. 36.5K equivalence: compare to `SAC_35999.pt` baseline and TensorBoard reference.
7. 100K paper-scale run: only after a candidate clears earlier gates.

Hard rejects:

- non-finite actor/critic/alpha/action-std metrics
- missing expected checkpoint
- multi-scene collapse, especially Normal/Complex
- speed caused by fewer/higher-quality-ambiguous environment interactions rather than real throughput
- changed action mask, reward, terminal, observation, RS switch, or scene scheduler semantics without labeling it as a new research experiment

## Main Recommendation

After the fast action-mask 20K candidate is closed, the best next move is not parallel environments yet. Build the new opt-in runner and use it to run exact-environment optimizations first, especially lidar/RS precompute and replay tensorization. In parallel, design the frozen image-feature cache because it has the best high-upside path that still respects the paper's four-input design.

Only move to batch/update sweeps or multi-env collection after the runner can report scene mix, replay mix, state-normalization behavior, policy version, and RS/action-mask parity clearly.
