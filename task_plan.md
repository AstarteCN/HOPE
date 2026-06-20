# Task Plan: HOPE Stage 3 Training Resource Study

## Goal

Prepare Stage 3 so original HOPE retraining includes hardware-utilization research, bottleneck analysis, 40,000-episode validation runs, and TensorBoard-based quality monitoring without breaking the original training framework.

## Current Phase

Stage 4 OGM Proxy Task 1 implemented; awaiting the next Stage 4 execution task

## Phases

### Phase 1: Boundary Update

- [x] Capture the user's Stage 3 requirements.
- [x] Update `AGENTS.md` with new Stage 3 boundaries.
- [x] Add a Stage 3 design/spec document.
- [x] Verify documentation changes.
- **Status:** complete

### Phase 2: Baseline Training And Resource Profile

- [x] Convert Stage 3 requirements into an executable implementation plan.
- [x] Correct the Stage 3 budget gate to 40,000 training episodes.
- [x] Record the 300-episode run as diagnostic only.
- [x] Run original HOPE SAC training without performance modifications.
- [x] Stop at the user-approved 36,500-episode target for this baseline.
- [x] Monitor CPU, GPU, RAM, disk, wall-clock throughput, and TensorBoard scalars.
- [x] Record baseline resource utilization and training signals.
- **Status:** complete with caveat: this run stopped at 36,502 episodes by user instruction, not 40,000.

### Phase 3: Bottleneck Analysis

- [x] Determine whether the current training run fully uses local hardware.
- [x] If not, identify the dominant bottleneck: environment stepping, Pygame rendering, Shapely collision checks, action mask calculation, replay sampling, network update, logging, or evaluation.
- [x] Record evidence before proposing code changes.
- **Status:** complete

### Phase 4: Quality-Preserving Optimization Plan

- [x] Propose optimizations that preserve original HOPE training semantics.
- [x] Prefer opt-in monitoring or experiment entry points over modifying original defaults.
- [x] Define quality gates for comparing optimized runs against baseline.
- **Status:** complete

### Phase 5: Command-Only Smoke Stop And Quality Check

- [x] Run the user-approved 1000-episode command-only parameter smoke with `--verbose "" --visualize ""`.
- [x] Measure short-run training speed and resource consumption without treating it as a quality validation.
- [x] Start one continuous command-only training process so early smoke data preserves replay buffer, optimizer state, RNG state, and training counters.
- [x] Update the heartbeat automation into a one-time 20K stop hook because current speed gains are limited.
- [x] At about 20K, stop this command-only smoke run after `SAC_19999.pt` exists.
- [x] Compare TensorBoard indicators against the baseline 20K reference.
- [x] Run matched 200-episode evaluations for command-only `SAC_19999.pt` and baseline `SAC_19999.pt` if not already available.
- [x] Compare the 20K smoke checkpoint against the stopped 36.5K baseline with a clear caveat that this is not full 36.5K equivalence.
- [x] Decide whether the command-only flags caused any negative training-quality impact.
- [x] Document training failure modes and early warning indicators observed in the 20K smoke.
- **Status:** complete. The command-only flags showed no negative quality impact at the 20K smoke point; this does not prove 36.5K equivalence.

### Phase 6: Training Architecture Speed Research

- [x] If the 20K smoke shows no negative training-quality impact, begin research-only analysis of safe training-architecture speedups.
- [x] Preserve the HOPE paper's core design: action mask, curriculum and scene scheduling, observation/action/reward semantics, environment dynamics, and hybrid policy/path-planning intent.
- [x] Focus on training methodology and engineering architecture: profiling, data/compute pipeline, replay/update scheduling, evaluation/logging overhead, process structure, and opt-in experiment wrappers.
- [x] Avoid large algorithmic rewrites, OGM implementation, reward redesign, curriculum removal, action-mask removal, or source changes under `src/train`, `src/env`, or `src/model` without a separate user-approved implementation plan.
- [x] Produce a concrete research plan before any source-level optimization.
- [x] Complete a deeper paper/code/training-summary research note that separates HOPE invariants, safe operational parameters, controlled hyperparameter experiments, and red-line changes.
- [x] Execute the research plan by collecting a fresh profiler trace of the original and command-only training paths.
- [x] Turn profiler evidence into an implementation plan target for the next user-approved speed experiment.
- **Status:** env.step internal profiling complete. First safe optimization target selected: an opt-in exact-output fast path for `ActionMask.get_steps`.

### Phase 7: Safe-Speed 20K PRD

- [x] Use `superpowers:brainstorming` to convert the safe-speed deep research into a PRD scope.
- [x] Select PRD scope B: measurement pack, parity harness, 20K gated workflow, and candidate admission rules.
- [x] Lock future tests to 20K episodes and the command-only 20K baseline `sac_20260619_004316 / SAC_19999.pt`.
- [x] Write the PRD under `docs/superpowers/specs/2026-06-19-hope-stage3-safe-speed-20k-prd.md`.
- [x] User selected scope B and approved converting the PRD into an implementation plan.
- [x] Transition to `superpowers:writing-plans` for an implementation plan.
- **Status:** PRD accepted for implementation planning.

### Phase 8: Safe-Speed 20K Implementation Plan

- [x] Use `superpowers:writing-plans` to convert the PRD into an executable implementation plan.
- [x] Keep the plan outside protected HOPE source changes and centered on opt-in tooling under `tools/stage3/`.
- [x] Define concrete tasks for baseline constants, manifests, TensorBoard gate summaries, eval/resource parsing, comparison reports, parity checks, profiling, launch/stop/eval wrappers, and documentation.
- [x] Execute implementation tasks 1-9 from `docs/superpowers/plans/2026-06-19-hope-stage3-safe-speed-20k-framework.md`.
- [x] Implement opt-in framework tooling under `tools/stage3/` for baseline constants, manifests, TensorBoard gate summaries, eval/resource parsing, 20K comparison reports, parity checks, profiling, and launch/stop/eval wrappers.
- [x] Create `docs/research/stage3_20k_candidate_report_template.md` for future 20K candidate decisions.
- [x] Run Task 10 final verification for the safe-speed framework and protected-source boundaries.
- **Status:** implementation, final verification, and final code review complete. Final verification passed 28 Stage 3 tool tests, PowerShell wrapper parser checks, baseline TensorBoard summary export, parity/profile smoke checks, compare dry-run, `git diff --check`, and protected-source diff checks.

### Phase 9: Opt-In Fast Action Mask

- [x] Create `docs/superpowers/plans/2026-06-19-hope-stage3-fast-action-mask.md`.
- [x] Write failing TDD tests for default-off fast action mask parity and profiler action-mask mode.
- [x] Implement `ActionMask(fast_get_steps=False)` with default behavior unchanged.
- [x] Add `_get_steps_original()` and `_get_steps_fast()` so the optimized branch is explicit and reviewable.
- [x] Add `--action-mask-mode {original,fast}` to bounded profiler diagnostics.
- [x] Verify exact-output parity on deterministic edge lidar samples and seeded random lidar samples.
- [x] Run full Stage 3 tool tests.
- [x] Run bounded original/fast env-step profile comparison.
- [x] Document results in `docs/research/2026-06-19-stage3-fast-action-mask-opt-in.md`.
- [x] Run final hygiene checks and push to the draft PR.
- **Status:** complete. This is not a 20K quality result; fast action mask is admitted only as a future 20K gated candidate.

### Phase 10: Fast Action Mask 20K Candidate Run

- [x] Confirm current branch and planning context.
- [x] Identify that the fast action-mask branch is default-off and needs an explicit opt-in training wrapper.
- [x] Add a TDD-tested `tools/stage3/train_HOPE_sac_fast_action_mask.py` wrapper that enables `fast_get_steps=True` by default only for this candidate entry point.
- [x] Add `-FastActionMask` support to `tools/stage3/launch_stage3_20k.ps1`.
- [x] Adjust `tools/stage3/stop_stage3_at_20k.ps1` process-safety command fragment to allow the fast-action-mask training wrapper.
- [x] Run Stage 3 tool tests and PowerShell parser checks before launching.
- [x] Commit/push the opt-in launcher wrapper so the long run starts from a clean Git state.
- [x] Launch 20K fast-action-mask candidate training with command-only flags and resource monitor.
- [x] Reassign manifest workload PID from the venv launcher parent to the actual child Python training process.
- [x] Restart the resource monitor against the actual child training PID.
- [x] Set continuous monitor automation for the candidate run.
- [x] At/after 20K and `SAC_19999.pt`, stop the run, summarize TensorBoard/resource metrics, evaluate candidate checkpoint, and compare against saved 20K and 36.5K baselines.
- **Status:** milestone accepted. The fast action-mask candidate passed the 20K speed and quality gates against the command-only 20K baseline: `10.947863 h` to checkpoint, `1826.840564` episodes/hour, `46.223481` env steps/second, and matched 200-episode external eval Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`, mean `0.88625`. On 2026-06-20 the user confirmed this speedup and quality result as accepted, and `hope-fast-action-mask-20k` is now the local baseline for future work.

### Phase 11: OGM Integration PRD Brainstorming

- [x] Read the current OGM integration research report.
- [x] Record the accepted fast action-mask 20K milestone as the current Stage 3 baseline.
- [x] Clarify the first OGM product scope and reproduction target.
- [x] Propose 2-3 OGM integration PRD approaches with trade-offs.
- [x] Present the selected PRD design sections for user approval.
- [x] Write the approved PRD/spec under `docs/superpowers/specs/`.
- [x] Wait for user review of `docs/superpowers/specs/2026-06-20-hope-rl-ogm-proxy-prd.md`.
- [x] Convert the approved PRD into an executable implementation plan.
- **Status:** complete. PRD was approved and converted into `docs/superpowers/plans/2026-06-20-hope-rl-ogm-proxy-integration.md`.

### Phase 12: Stage 4 OGM Proxy Implementation

- [x] Write the implementation plan with `superpowers:writing-plans`.
- [x] Add the user-requested validation policy: first gate at 20K episodes against `hope-fast-action-mask-20k`, then every-10K monitoring with one 10K grace window before stopping stagnant or degrading runs.
- [x] Select execution mode for Task 1: Inline Execution by direct user request.
- [x] Add Stage 4 target constants and the first progress-gate baseline test.
- [ ] Implement proxy OGM rasterizer.
- [ ] Integrate default-off OGM observation into the environment and wrapper.
- [ ] Add explicit OGM network/config/state-normalization support.
- [ ] Add fixed OGM-style simulation evaluation set, OGM eval metrics, progress gates, and launch/monitor tooling.
- [ ] Run 1K, 20K, and every-10K gated validation ladder.
- **Status:** Task 1 implemented. Stage 4 now has target constants, OGM paper simulation target bands, and the accepted `hope-fast-action-mask-20k` baseline recorded under `tools/stage4/`.

## Key Questions

1. Does the original training run fully use the local RTX 4080 SUPER and CPU resources?
2. If not, which subsystem is the bottleneck?
3. Which optimizations can improve throughput without changing training quality or original HOPE semantics?
4. Which TensorBoard patterns predict early training failure?
5. What environment-step and throughput profile emerges during a 40,000-episode meaningful smoke/validation run?

## Decisions Made

| Decision | Rationale |
|----------|-----------|
| Use root-level planning files | Required by the `planning-with-files` skill so future sessions can recover context. |
| Stage 1 stops at environment/import verification | The user requested Stage 1 only; checkpoint execution belongs to Stage 2. |
| Avoid default Python 3.14 if possible | The local launcher defaults to Python 3.14.5, while this ML project is more likely to work on an older supported Python. |
| Use Python 3.13 for `.venv` | Python 3.8 is not available locally and neither Conda nor uv is installed. Python 3.13 is the oldest available local interpreter and avoids the default Python 3.14. |
| Add `.venv/` to `.gitignore` | Prevents the isolated environment directory from being accidentally tracked. |
| Start Stage 2 with `HOPE_SAC0.pt` | `AGENTS.md` explicitly recommends this checkpoint first. |
| Use `SDL_VIDEODRIVER=dummy` for evaluation | Keeps Pygame headless while preserving original code path. |
| Run short cross-checks for SAC1 and PPO | `eval_mix_scene.py` supports SAC/PPO selection by checkpoint filename, and brief runs validate the remaining author checkpoints without spending Stage 3 training time. |
| Stage 3 must start with profiling, not optimization | The user explicitly asked to research hardware utilization and bottlenecks before changes. |
| Stage 3 smoke and modification validation should cover at least 40,000 training episodes | The user clarified this is about 40% of the original author's 100,000-episode training scale; very short RL runs do not provide enough signal for reward, success-rate, or failure-mode judgment. |
| Keep environment interaction steps as a secondary metric | TensorBoard `step_num` sums remain useful for throughput and SAC-update estimates, but they are not the Stage 3 budget gate. |
| Preserve original HOPE training semantics | Performance work must not break or silently replace the original author's training framework. |
| Stage 3 execution plan saved under `docs/superpowers/plans/2026-06-17-hope-stage3-training-resource-study.md` | Future execution should follow the plan task-by-task with `superpowers:subagent-driven-development` or `superpowers:executing-plans`. |
| Do not use large batch or extra layers as the first speed lever | Current original baseline is learning well, while the prior 40K modified run mixed larger visual layers and other changes and underperformed badly; batch/layer changes alter SAC learning dynamics rather than simply accelerating wall-clock training. |
| Treat `--verbose ""` as the lowest-risk future run acceleration candidate | `argparse type=bool` makes `--verbose False` parse as `True`; an empty string parses as `False` and should disable repeated printing and `reward.png` writes without changing reward/action/environment semantics. |
| Require parity checks before changing render-mode behavior | `--visualize ""` may select `render_mode='rgb_array'`, but observation/reward/status parity should be verified before using it for quality runs. |
| Save safe-acceleration research under `docs/research/2026-06-18-stage3-safe-acceleration-research.md` | The user requested a deeper quality-preserving speed study based on current training summaries and code evidence. |
| Treat 30K as enough for preliminary resource judgment but not final baseline closure | Near-30K data already shows stable low hardware saturation and healthy learning signals, but the Stage 3 baseline still needs the requested 40,000 episodes before final reproduction-quality conclusions. |
| Stop the current run at 36,500 episodes and accept it as the local baseline | On 2026-06-18 the user explicitly superseded the 40K continuation request because the run had approached 24 hours; the run stopped at 36,502 episodes and was evaluated as the current acceleration baseline. |
| Do not optimize for resource saturation | The user clarified that the goal is faster, high-quality training, not filling CPU/GPU utilization for its own sake. |
| Use command-only overhead reduction as the first acceleration test | `--verbose ""` and `--visualize ""` preserve the original training structure and remove avoidable output/display overhead before source-level changes. |
| Limit the approved command-only acceleration test to 1000 episodes | On 2026-06-18 the user explicitly approved only a 1000-episode parameter smoke to measure speed/resource use; this must not be presented as a 35K/40K quality validation. |
| Command-only flags are a validated speed candidate for the next long run | The 1000-episode smoke finished in 37.69 minutes at 56.37 environment steps/s, versus 14.99 environment steps/s for the original baseline's first 1000 episodes. This validates the speed direction, not long-run quality. |
| Use one continuous process while collecting command-only smoke evidence | Restarting would reload weights but lose replay buffer and optimizer state, so the current run should be stopped externally around the user-approved target instead of relaunched. |
| Stop the current command-only run around 20K episodes | On 2026-06-19 the user judged the speed gain limited and superseded the 36.5K/100K continuation for this run. |
| Use `SAC_19999.pt` as the command-only smoke checkpoint | The original training script saves every 2000 episodes as `SAC_{i}.pt`; `SAC_19999.pt` is the nearest 20K checkpoint without source changes. |
| Preserve HOPE design intent during speed research | Future optimization research must keep action masking, curriculum/scene scheduling, observation/action/reward semantics, and the paper's hybrid planner framing intact. |
| Command-only flags passed the 20K smoke-quality check | Matched 200-episode eval for command-only `SAC_19999.pt` and baseline `SAC_19999.pt` was identical across Normal, Complex, Extrem, and DLP, so `--visualize= --verbose=` did not show a quality regression at 20K. |
| Treat hyperparameter scaling as allowed but gated research | Tuning batch size, update cadence, replay size, or other training parameters is not a red-line violation by itself, but it must preserve HOPE task semantics and pass matched quality gates before being called a safe speedup. |
| Prioritize measurement before source optimization | Current evidence points to environment/render/action-mask/RS/replay pipeline costs, so the next implementation should add opt-in profiling and parity harnesses before changing hot paths. |
| Use 20K as the next speed-test validation budget | The user explicitly selected future tests to use 20K episodes and compare against the current command-only 20K result as baseline. |
| PRD scope B accepted | The PRD should include measurement tooling, parity checks, a 20K gated workflow, and candidate optimization admission rules, without directly implementing optimization code. |
| Safe-speed framework synced to GitHub before further work | Commit `d4eec63` was pushed to `origin/codex/stage3-resource-study` and draft PR `https://github.com/AstarteCN/HOPE/pull/1` was created against the user's fork. |
| Fresh profiler traces should compare per-call costs, not raw total time | The 2026-06-19 original and command-only bounded diagnostics collected different transition counts, so total wall time is confounded by episode path length. Per-call costs still show `env.step` and `ParkingAgent.get_action` dominate both modes. |
| First true safe optimization point is `ActionMask.get_steps` | Detailed env-step profiling shows action mask costs about `1.78 ms/raw_step`; an equivalent allocation-free formula matched outputs exactly in a micro-probe and reduced isolated calls from about `1.48 ms` to `0.77 ms`. This preserves action-mask semantics and is safer than touching the RGB image pipeline first. |
| Keep fast action mask default-off | The approved source change must not silently alter original HOPE training. `ActionMask()` still uses the original `get_steps` behavior; fast mode requires `fast_get_steps=True` or profiler `--action-mask-mode fast`. |
| Treat fast action mask as an admitted candidate, not a validated training speedup | Bounded diagnostics show action-mask average cost dropped from `1.694 ms` to `0.821 ms`, but the change still needs a future 20K gated run against the command-only 20K baseline before it can be called safe for training. |
| Fast action-mask 20K run launched from a clean Git state | Manifest `src/log/exp/stage3_fast_action_mask_20k_20260620_085206.meta.json` records clean branch status at launch, run dir `src/log/exp/sac_20260620_085208`, command-only flags, and `fast_action_mask=true`. |
| Reassign fast candidate monitor to the actual Python child PID | The venv launcher parent PID `29844` spawned child PID `3696`, which writes TensorBoard and consumes resources. Manifest and resource monitor were updated to track PID `3696`; the first few resource CSV rows for PID `29844` should be treated as startup-only monitor bias. |
| Fast action mask passed the 20K candidate gate | Against the saved command-only 20K baseline, the fast action-mask candidate improved time to checkpoint by `15.55%`, episodes/hour by `18.19%`, and env steps/second by `18.22%`, while matched 200-episode external eval was identical. Keep the 36.5K comparison as a maturity caveat, not an equivalence claim. |
| Close Stage 3 speed research at the accepted fast action-mask 20K milestone | On 2026-06-20 the user confirmed the fast action-mask 20K speedup and training quality as accepted. Future OGM work should use `hope-fast-action-mask-20k` as the local HOPE baseline anchor unless a later long-run supersedes it. |
| Start OGM work as a PRD/design phase | The next goal is to translate paper/code research into a PRD for Stage 4 OGM integration. No OGM implementation should start until the PRD is approved and converted into a plan. |
| Exclude OGM paper Real-World dataset KPI from current acceptance | Local repository inspection found only geometry-based `data/dlp.data`, not real OGM maps or sensor-derived OGM datasets. Current PRD acceptance should focus on available simulation KPIs unless a real OGM dataset is provided later. |
| OGM first policy input scope is option A | The first PRD target uses `ogm + target + action_mask` as policy input, disables RGB BEV for the OGM policy, and keeps lidar internally only for existing HOPE action-mask generation. |
| OGM evaluation dataset scope is option C | Hard acceptance should use a new OGM-style fixed `20 parallel + 50 perpendicular` simulation evaluation set, while HOPE Normal/Complex remain compatibility and regression metrics. |
| OGM PRD route is方案 1: Proxy OGM First | Build a first-class proxy OGM observation path inside HOPE, enforce strict simulation KPI gates, and defer real-world OGM dataset acceptance until real OGM data exists. |
| OGM Proxy PRD written for review | The approved brainstorming design was saved to `docs/superpowers/specs/2026-06-20-hope-rl-ogm-proxy-prd.md`; the next step is user review, not implementation. |
| OGM Proxy implementation plan written | The approved PRD was converted into `docs/superpowers/plans/2026-06-20-hope-rl-ogm-proxy-integration.md`; execution is gated on user choosing Subagent-Driven or Inline Execution. |
| OGM validation must stop bad long runs early | Stage 4 will use a 20K first gate and every-10K monitoring. A weak early OGM result can continue if trend improves, but a stagnant/regressing gate gets only one additional 10K grace window before stopping for TensorBoard/debug-driven optimization. |

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| `TypeError: CarParkingWrapper.reset() got an unexpected keyword argument 'level'` | 1 | Root cause: wrapper accepts positional `*args` only. Retry with original training-style positional call: `w.reset(None, None, 'Normal')`. |
| `_pickle.UnpicklingError: Weights only load failed` when loading `HOPE_SAC0.pt` | 1 | Root cause: PyTorch 2.6+ defaults `torch.load(weights_only=True)`, but the trusted author checkpoint includes custom `SACConfig`. Retry command with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` rather than editing source. |
| Stage 3 resource monitor PID `25356` exited while training continued | 1 | Made the external monitor append-safe for existing CSV files and restarted it as PID `34596` against the same resource CSV. |
| Micro-profile command used the wrong `.venv` relative path from `src` | 1 | Reran with absolute interpreter path `D:\Github\HOPE\.venv\Scripts\python.exe`. |
| Render-mode parity script compared `OrderedDict` reward keys instead of values | 1 | Reran the check using `reward.values()`; Normal, Complex, Extrem, and DLP scripted cases matched exactly. |
| `git commit` for the PRD failed because Git author identity is not configured | 1 | Historical note: unstaged the PRD after the failed commit and left it as a normal working-tree file for user review at that moment; the PRD was later accepted and converted into the safe-speed framework plan. Do not set `user.name` or `user.email` without user approval. |
| `stop_stage3_at_20k.ps1` rejected the reassigned child PID because PowerShell `ConvertFrom-Json` shifted manifest timestamps through local/UTC semantics | 1 | Used a manual fallback with equivalent PID/name/command-line validation; workload and resource monitor were already inactive after natural finish, and the manifest records the fallback reason. |

## Notes

- Do not modify original HOPE source files during Stage 3 performance/resource work unless the user explicitly approves a separate source-change plan. The 2026-06-19 approved exception is the default-off `ActionMask.get_steps` fast path in `src/model/action_mask.py`.
- Do not install into global Python.
- Do not run OGM implementation work in this stage.
- 2026-06-20 OGM integration research refresh was recorded in `docs/research/2026-06-20-rl-ogm-integration-current-code-research.md`; it is research-only and does not change the current Phase 10 run status.
