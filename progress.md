# Progress Log

## Session: 2026-06-19

### GitHub Sync And Profiler Evidence Start

- **Status:** synced and profiler evidence collection started
- Actions taken:
  - Confirmed `origin` points to `https://github.com/AstarteCN/HOPE.git` and `upstream` push is disabled.
  - Kept `.codegraph/`, `.venv/`, `raw_paper/`, training logs, and generated runtime caches out of the commit.
  - Configured repo-local Git identity from the authenticated `gh` account: `AstarteCN <105480757+AstarteCN@users.noreply.github.com>`.
  - Committed the safe-speed framework and research snapshot as `d4eec63 Add Stage 3 safe speed tooling`.
  - Pushed `codex/stage3-resource-study` to `origin`.
  - Created draft PR `https://github.com/AstarteCN/HOPE/pull/1`.
  - Added explicit `--profile-mode original|command-only` support to `tools/stage3/profile_stage3_components.py`.
  - Added TDD coverage in `tools/stage3/tests/test_profile_stage3_components.py`; the new test first failed because `profile_mode_settings` did not exist, then passed after implementation.
  - Ran full Stage 3 tool tests: `30` tests passed.
  - Collected fresh bounded profiler traces:
    - Original: `docs/research/stage3_profile_original_20260619.json` and `.md`.
    - Command-only: `docs/research/stage3_profile_command_only_20260619.json` and `.md`.
  - Initial evidence: compare per-call component cost, not raw total time, because the two bounded diagnostics collected different transition counts. `env.step` and `ParkingAgent.get_action` remain the dominant components in both modes.
  - No original files under `src/train`, `src/env`, or `src/model` were modified.

### Stage 3: Env Step Internal Profile

- **Status:** complete; first safe optimization point selected
- Actions taken:
  - Read `CarParking.step`, `CarParking.render`, `CarParkingWrapper.step`, lidar, action-mask, observation-processing, vehicle, and RS-path code paths.
  - Added `--detail env-step` to `tools/stage3/profile_stage3_components.py` with nested timers for render, image, lidar, action mask, simulation, status, reward, RS probing, and wrapper overhead.
  - Added TDD coverage in `tools/stage3/tests/test_profile_stage3_components.py`; the new detail test failed before `profile_detail_components` existed and passed after implementation.
  - Ran full Stage 3 tool tests: `32` tests passed.
  - Collected command-only and original env-step detail traces under `docs/research/stage3_env_step_detail_*_20260619.*`.
  - Ran a no-source-change micro-probe for an allocation-free `ActionMask.get_steps` equivalent formula. The formula matched outputs exactly on checked samples and improved isolated calls from about `1.481 ms` to `0.768 ms`.
  - Ran a runtime monkeypatch estimate inside the bounded diagnostic: `env.render.action_mask` dropped to about `0.83 ms/call`, and raw-step average dropped to about `8.48 ms`.
  - Wrote `docs/research/2026-06-19-stage3-env-step-internal-profile.md`.
  - Selected the first safe optimization target: an opt-in exact-output fast path for `ActionMask.get_steps`.
  - No original files under `src/train`, `src/env`, or `src/model` were modified.

### Stage 3: Safe-Speed 20K PRD

- **Status:** PRD accepted and converted to implementation plan
- Actions taken:
  - Used `superpowers:brainstorming` after the user requested a PRD based on `docs/research/2026-06-19-stage3-safe-speed-deep-research.md`.
  - Re-read the safe-speed deep research note, the command-only 20K smoke result, and planning files.
  - Presented three PRD scope options and recommended scope B.
  - User approved scope B.
  - Wrote `docs/superpowers/specs/2026-06-19-hope-stage3-safe-speed-20k-prd.md`.
  - The PRD defines measurement tooling, parity harness, 20K validation workflow, comparison report, candidate admission rules, quality/speed gates, and acceptance criteria.
  - Updated `task_plan.md` and `findings.md` with Phase 7, baseline identity, and PRD gate decisions.
  - Attempted to commit only the PRD file as required by the brainstorming flow, but Git rejected the commit because `user.name` and `user.email` are not configured in this repository/global Git config.
  - Historical note: unstaged the PRD after the failed commit attempt and left it as an ordinary working-tree file for user review at that moment; it was later accepted and converted into the implementation plan.
  - Did not implement tooling or modify original `src/train`, `src/env`, or `src/model` files.

### Stage 3: Safe-Speed 20K Implementation Plan

- **Status:** implementation, final verification, and final code review complete
- Actions taken:
  - Used `superpowers:writing-plans` after the user requested converting the PRD into an executable implementation plan.
  - Wrote `docs/superpowers/plans/2026-06-19-hope-stage3-safe-speed-20k-framework.md`.
  - The plan defines implementation tasks for baseline constants, manifests, TensorBoard gate windows, eval/resource parsers, 20K comparison reports, parity checks, component profiling, launch/stop/eval wrappers, report templates, and final verification.
  - Implementation proceeded through Tasks 1-9 of the safe-speed 20K framework plan: baseline constants, manifest helper, TensorBoard gate summary, eval/resource parsers, comparison report, parity harness, component profiler, 20K launch/stop/eval wrappers, and Task 9 report/planning records.
  - Framework tooling now exists under `tools/stage3/` and keeps original `src/train`, `src/env`, and `src/model` protected.
  - Task 9 created `docs/research/stage3_20k_candidate_report_template.md` and updated planning records.
  - Task 10 final verification passed:
    - `python -m unittest discover -s tools\stage3\tests -p "test_*.py" -v`: 28 tests passed.
    - PowerShell parser checks passed for `launch_stage3_20k.ps1`, `stop_stage3_at_20k.ps1`, and `eval_stage3_checkpoint.ps1`.
    - Baseline TensorBoard summary export against `src/log/exp/sac_20260619_004316` succeeded with `episode_count=20040`, `training_budget_met=true`, and `env_step_count=1825016`.
    - Parity smoke passed for Normal, Complex, Extrem, and DLP.
    - Profiling smoke passed and wrote measurement-only reports.
    - Compare dry-run passed with decision `pass` and no hard reject reasons.
    - `git diff --check` passed for tools/docs/planning changes.
    - `git diff -- src/train src/env src/model` remained empty.
    - Final code review passed after the 20K stop-wrapper PID-safety fixes and comparison-gate severity fixes.
  - No original HOPE files under `src/train`, `src/env`, or `src/model` were intentionally modified.

### Stage 3: Safe Speed Deep Research

- **Status:** research-only complete; later converted into the accepted PRD and verified framework implementation
- Actions taken:
  - Re-read the current Stage 3 planning state, findings, baseline resource profile, command-only smoke notes, and existing training-architecture speed research note.
  - Inspected local HOPE paper PDF from `raw_paper/2405.20579_HOPE_A_Reinforcement_Learning-based_Hybrid_Policy_Path_Planner_for_Diverse_Parking_Scenarios.pdf`.
  - Inspected original HOPE training, environment, action-mask, planner-wrapper, SAC, replay-memory, lidar, wrapper, config, and evaluation code paths.
  - Confirmed core invariants for safe speed work: action mask, adaptive scene/DLP case scheduling, hybrid RL plus Reeds-Shepp behavior, observation/action/reward semantics, environment dynamics, and Normal/Complex/Extrem/DLP evaluation KPIs.
  - Confirmed that parameter scaling is allowed as controlled research, but not automatically a safe speedup: batch size, replay size, update cadence, mini epochs, learning rates, model size, and parallel collection all require matched quality gates.
  - Wrote `docs/research/2026-06-19-stage3-safe-speed-deep-research.md` with the evidence base, red lines, bottleneck model, safe acceleration taxonomy, hyperparameter position, recommended path, and quality gates.
  - Updated `task_plan.md` and `findings.md` with the deep-research conclusions.
  - Did not modify original `src/train`, `src/env`, or `src/model` files.

## Session: 2026-06-17

### Stage 3: Command-Only 0-100K Gated Validation

- **Status:** in progress
- **Started:** 2026-06-19
- Actions taken:
  - User requested a command-only 0-100K run with early 20K evaluation, 36.5K equivalence comparison against the stopped local baseline, and final 100K comparison against author checkpoints.
  - Decided to run one continuous process rather than split/restart at 36.5K, because restart would lose replay buffer and optimizer state.
  - Exported baseline gate references to `docs/research/stage3_baseline_gate_references_20260619.json` and `.md`.
  - Created `docs/research/2026-06-19-stage3-command-only-100k-gated-run.md`.
  - Updated `task_plan.md` and `findings.md` with gate criteria and checkpoint mapping.
  - Started command-only 0-100K training at `2026-06-19T00:43:15+08:00`.
  - Locked run directory: `D:\Github\HOPE\src\log\exp\sac_20260619_004316`.
  - Recorded metadata at `D:\Github\HOPE\src\log\exp\stage3_command_only_100k_20260619_004315.meta.json`.
  - Confirmed workload PID `11136`, launcher PID `9300`, and resource monitor PID `14536`.
  - Confirmed TensorBoard is available at `http://127.0.0.1:6006`.
  - Created heartbeat automation `hope-command-only-100k-gated-monitor` to monitor progress and execute the 20K/36.5K/100K gates.
  - Heartbeat at 2026-06-19 01:14 confirmed launcher/workload/monitor processes are still running: 861 episodes, 110,691 environment steps, estimated 10,045 SAC updates after warmup, 376 resource samples, average normalized CPU `44.12%`, average whole-GPU utilization `34.09%`, peak GPU memory `3190 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 01:44 confirmed launcher/workload/monitor processes are still running: 1,713 episodes, 209,123 environment steps, estimated 19,888 SAC updates after warmup, 732 resource samples, average normalized CPU `45.31%`, average whole-GPU utilization `34.06%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 02:14 confirmed launcher/workload/monitor processes are still running: 2,584 episodes, 306,303 environment steps, estimated 29,606 SAC updates after warmup, 1,090 resource samples, average normalized CPU `45.62%`, average whole-GPU utilization `33.78%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 02:44 confirmed launcher/workload/monitor processes are still running: 3,437 episodes, 400,109 environment steps, estimated 38,986 SAC updates after warmup, 1,446 resource samples, average normalized CPU `45.62%`, average whole-GPU utilization `33.82%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 03:14 confirmed launcher/workload/monitor processes are still running: 4,317 episodes, 492,380 environment steps, estimated 48,214 SAC updates after warmup, 1,802 resource samples, average normalized CPU `45.61%`, average whole-GPU utilization `34.08%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 03:44 confirmed launcher/workload/monitor processes are still running: 5,181 episodes, 582,813 environment steps, estimated 57,257 SAC updates after warmup, 2,160 resource samples, average normalized CPU `45.52%`, average whole-GPU utilization `34.32%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 04:14 confirmed launcher/workload/monitor processes are still running: 6,013 episodes, 667,924 environment steps, estimated 65,768 SAC updates after warmup, 2,515 resource samples, average normalized CPU `45.35%`, average whole-GPU utilization `34.31%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 04:44 confirmed launcher/workload/monitor processes are still running: 6,890 episodes, 752,809 environment steps, estimated 74,256 SAC updates after warmup, 2,870 resource samples, average normalized CPU `45.20%`, average whole-GPU utilization `34.36%`, peak GPU memory `3211 MB`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 05:14 confirmed launcher/workload/monitor processes are still running: 7,756 episodes, 836,258 environment steps, estimated 82,601 SAC updates after warmup, 3,241 resource samples, average monitor CPU `44.96%`, average whole-GPU utilization `34.22%`, peak GPU memory `3211 MB`, latest saved training checkpoint `SAC_5999.pt`, and no protected `src/train`, `src/env`, or `src/model` diff.
  - Heartbeat at 2026-06-19 05:44 confirmed launcher/workload/monitor processes are still running without monitor restart: 8,564 episodes, 913,717 environment steps, estimated 90,347 SAC updates after warmup, 3,590 resource samples, average monitor CPU `44.71%`, average whole-GPU utilization `34.10%`, peak GPU memory `3211 MB`, latest saved training checkpoint `SAC_7999.pt`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite, `avg_reward` trend delta was positive at `0.0401`, latest success rates were Normal `0.91`, Complex `0.86`, Extrem `0.47`, and DLP `0.78`; the latest `step_num=200` timeout warning is being watched but is not a gate failure before 20K.
  - Heartbeat at 2026-06-19 06:14 confirmed launcher/workload/monitor processes are still running without monitor restart: 9,352 episodes, 987,453 environment steps, estimated 97,721 SAC updates after warmup, 3,944 resource samples, average monitor CPU `44.34%`, average whole-GPU utilization `33.98%`, peak GPU memory `3211 MB`, latest saved training checkpoint `SAC_7999.pt`, `SAC_best.pt` updated at 05:48, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite, `avg_reward` trend delta was positive at `0.0283`, latest success rates were Normal `0.99`, Complex `0.77`, Extrem `0.51`, and DLP `0.73`; the previous latest-step timeout warning was not present in this snapshot.
  - Heartbeat at 2026-06-19 06:44 confirmed launcher/workload/monitor processes are still running without monitor restart: 10,144 episodes, 1,061,329 environment steps, estimated 105,108 SAC updates after warmup, 4,302 resource samples, average monitor CPU `44.02%`, average whole-GPU utilization `33.83%`, peak GPU memory `3211 MB`, latest saved training checkpoint `SAC_9999.pt`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite, `avg_reward` trend delta was positive at `0.0309`, latest success rates were Normal `0.91`, Complex `0.86`, Extrem `0.48`, and DLP `0.73`; warnings were limited to being below the 20K minimum.
  - Heartbeat at 2026-06-19 07:14 confirmed launcher/workload/monitor processes are still running without monitor restart: 10,903 episodes, 1,133,044 environment steps, estimated 112,280 SAC updates after warmup, 4,658 resource samples, average monitor CPU `43.70%`, average whole-GPU utilization `33.92%`, peak GPU memory `3211 MB`, latest saved training checkpoint `SAC_9999.pt`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite, `avg_reward` trend delta was positive at `0.0445`, latest success rates were Normal `0.90`, Complex `0.82`, Extrem `0.53`, and DLP `0.79`; warnings were limited to being below the 20K minimum.
  - User judged the command-only long-run speed gain limited and requested a one-time hook to stop this smoke run around 20K episodes instead of continuing to 36.5K or 100K.
  - Updated heartbeat automation `hope-command-only-100k-gated-monitor` into `HOPE command-only 20K smoke stop hook`, running every 15 minutes. The hook should stop after `SAC_19999.pt` exists, evaluate/report the 20K smoke, start safe speed-architecture research if quality is not negatively impacted, and delete itself after completion.
  - Updated `AGENTS.md`, `task_plan.md`, `findings.md`, and `docs/research/2026-06-19-stage3-command-only-100k-gated-run.md` with the 20K stop scope and training-architecture speed research boundaries.
  - Created `docs/research/2026-06-19-stage3-training-architecture-speed-research.md` as the pending research-only entry point.
  - 20K stop-hook heartbeat at 2026-06-19 07:54 confirmed the run is still below the stop target: 11,926 episodes, 1,219,983 environment steps, estimated 120,974 SAC updates after warmup, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 5,126 samples, average monitor CPU `43.11%`, average whole-GPU utilization `33.59%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite; latest success rates were Normal `0.93`, Complex `0.90`, Extrem `0.64`, and DLP `0.82`.
  - 20K stop-hook heartbeat at 2026-06-19 08:09 confirmed the run is still below the stop target: 12,312 episodes, 1,251,439 environment steps, estimated 124,119 SAC updates after warmup, latest saved training checkpoint `SAC_11999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 5,305 samples, average monitor CPU `42.86%`, average whole-GPU utilization `33.39%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite; latest success rates were Normal `0.92`, Complex `0.89`, Extrem `0.55`, and DLP `0.84`.
  - 20K stop-hook heartbeat at 2026-06-19 08:24 confirmed the run is still below the stop target: 12,721 episodes, 1,283,232 environment steps, estimated 127,299 SAC updates after warmup, latest saved training checkpoint `SAC_11999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 5,483 samples, average monitor CPU `42.65%`, average whole-GPU utilization `33.24%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard losses and rewards remained finite; latest success rates were Normal `0.99`, Complex `0.85`, Extrem `0.67`, and DLP `0.84`. Latest `step_num=200` timeout warning recurred and should continue to be watched before the 20K stop.
  - 20K stop-hook heartbeat at 2026-06-19 08:54 confirmed the run is still below the stop target: 13,485 episodes, 1,345,429 environment steps, estimated 133,518 SAC updates after warmup, latest saved training checkpoint `SAC_11999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 5,838 samples, average monitor CPU `42.20%`, average whole-GPU utilization `32.83%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.87`, Complex `0.85`, Extrem `0.54`, and DLP `0.82`. Latest `step_num=27` means the 08:24 timeout sample did not persist at the most recent point, but Extrem remains volatile and should be checked again at the 20K stop.
  - 20K stop-hook heartbeat at 2026-06-19 09:09 confirmed the run is still below the stop target: 13,829 episodes, 1,374,583 environment steps, estimated 136,434 SAC updates after warmup, latest saved training checkpoint `SAC_11999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,014 samples, average monitor CPU `41.97%`, average whole-GPU utilization `32.71%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.97`, Complex `0.84`, Extrem `0.67`, and DLP `0.84`. Latest `step_num=62`; Extrem recovered from the previous low snapshot but remains a key 20K quality-check signal.
  - 20K stop-hook heartbeat at 2026-06-19 09:24 confirmed the run is still below the stop target: 14,193 episodes, 1,403,436 environment steps, estimated 139,319 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,190 samples, average monitor CPU `41.74%`, average whole-GPU utilization `32.52%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.99`, Complex `0.87`, Extrem `0.62`, and DLP `0.82`. Latest `step_num=12`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 09:39 confirmed the run is still below the stop target: 14,597 episodes, 1,432,446 environment steps, estimated 142,220 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,369 samples, average monitor CPU `41.51%`, average whole-GPU utilization `32.29%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.93`, Complex `0.91`, Extrem `0.69`, and DLP `0.87`. Latest `step_num=200` timeout warning recurred, but scene success rates did not indicate catastrophic collapse; continue monitoring until `SAC_19999.pt`.
  - 20K stop-hook heartbeat at 2026-06-19 09:54 confirmed the run is still below the stop target: 14,961 episodes, 1,459,318 environment steps, estimated 144,907 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,547 samples, average monitor CPU `41.24%`, average whole-GPU utilization `32.02%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.90`, Complex `0.88`, Extrem `0.68`, and DLP `0.87`. Latest `step_num=121`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 10:09 confirmed the run is still below the stop target: 15,314 episodes, 1,487,666 environment steps, estimated 147,742 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,726 samples, average monitor CPU `41.02%`, average whole-GPU utilization `31.83%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.92`, Complex `0.93`, Extrem `0.58`, and DLP `0.81`. Latest `step_num=9`; Extrem dipped again and remains the main scene-specific signal to inspect at the 20K stop.
  - 20K stop-hook heartbeat at 2026-06-19 10:24 confirmed the run is still below the stop target: 15,652 episodes, 1,513,887 environment steps, estimated 150,364 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 6,904 samples, average monitor CPU `40.76%`, average whole-GPU utilization `31.63%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.89`, Complex `0.90`, Extrem `0.67`, and DLP `0.77`. Latest `step_num=200` timeout warning recurred; keep watching timeout frequency and DLP/Extrem before the 20K stop.
  - 20K stop-hook heartbeat at 2026-06-19 10:39 confirmed the run is still below the stop target: 15,999 episodes, 1,540,060 environment steps, estimated 152,982 SAC updates after warmup, latest saved training checkpoint `SAC_13999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,083 samples, average monitor CPU `40.51%`, average whole-GPU utilization `31.41%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.95`, Complex `0.91`, Extrem `0.59`, and DLP `0.87`. Latest `step_num=200` timeout warning recurred; Extrem remains the weakest scene-specific 20K quality signal.
  - 20K stop-hook heartbeat at 2026-06-19 10:54 confirmed the run is still below the stop target: 16,313 episodes, 1,565,128 environment steps, estimated 155,488 SAC updates after warmup, latest saved training checkpoint `SAC_15999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,260 samples, average monitor CPU `40.26%`, average whole-GPU utilization `31.25%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.98`, Complex `0.88`, Extrem `0.62`, and DLP `0.80`. Latest `step_num=200` timeout warning recurred; continue monitoring until the `SAC_19999.pt` stop checkpoint exists.
  - 20K stop-hook heartbeat at 2026-06-19 11:09 confirmed the run is still below the stop target: 16,644 episodes, 1,590,860 environment steps, estimated 158,062 SAC updates after warmup, latest saved training checkpoint `SAC_15999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,439 samples, average monitor CPU `40.03%`, average whole-GPU utilization `31.10%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.93`, Complex `0.91`, Extrem `0.63`, and DLP `0.83`. Latest `step_num=98`; timeout did not persist at this snapshot.
  - 20K stop-hook heartbeat at 2026-06-19 11:24 confirmed the run is still below the stop target: 17,054 episodes, 1,616,019 environment steps, estimated 160,577 SAC updates after warmup, latest saved training checkpoint `SAC_15999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,617 samples, average monitor CPU `39.79%`, average whole-GPU utilization `30.96%`, peak GPU memory `3317 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard rewards and losses remained finite; latest success rates were Normal `0.97`, Complex `0.98`, Extrem `0.79`, and DLP `0.86`. Latest `step_num=85`; timeout did not persist, and `SAC_best.pt` updated at 2026-06-19 11:22.
  - 20K stop-hook heartbeat at 2026-06-19 11:39 confirmed the run is still below the stop target: 17,399 episodes, 1,642,342 environment steps, estimated 163,210 SAC updates after warmup, latest saved training checkpoint `SAC_15999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,817 samples, average monitor CPU `39.52%`, average whole-GPU utilization `30.83%`, peak GPU memory `3450 MB`, and no protected `src/train`, `src/env`, or `src/model` diff. TensorBoard reward and loss summaries remained finite; latest success rates were Normal `0.93`, Complex `0.94`, Extrem `0.69`, and DLP `0.82`. Latest `step_num=12`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 11:54 confirmed the run is still below the stop target: 17,726 episodes, 1,664,906 environment steps, estimated 165,466 SAC updates after warmup, latest saved training checkpoint `SAC_15999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 7,976 samples, average monitor CPU `39.32%`, average whole-GPU utilization `30.64%`, recent-window CPU/GPU averages `29.09%`/`23.76%`, and peak GPU memory `3450 MB`. TensorBoard reward and loss summaries remained finite; latest success rates were Normal `0.98`, Complex `0.92`, Extrem `0.72`, and DLP `0.85`. Latest `step_num=54`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 12:09 confirmed the run is still below the stop target: 18,059 episodes, 1,688,445 environment steps, estimated 167,820 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 8,155 samples, average monitor CPU `39.08%`, average whole-GPU utilization `30.47%`, recent-window CPU/GPU averages `28.48%`/`22.91%`, and peak GPU memory `3471 MB`. TensorBoard reward and loss summaries remained finite; latest success rates were Normal `0.97`, Complex `0.92`, Extrem `0.77`, and DLP `0.82`. Latest `step_num=27`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 12:24 confirmed the run is still below the stop target: 18,406 episodes, 1,712,092 environment steps, estimated 170,185 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 8,333 samples, average monitor CPU `38.86%`, average whole-GPU utilization `30.30%`, recent-window CPU/GPU averages `28.64%`/`22.47%`, and peak GPU memory `3471 MB`. TensorBoard reward and loss summaries remained finite; latest success rates were Normal `0.97`, Complex `0.91`, Extrem `0.74`, and DLP `0.87`. Latest `step_num=15`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 12:39 confirmed the run is still below the stop target: 18,730 episodes, 1,734,520 environment steps, estimated 172,428 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 8,510 samples, average monitor CPU `38.63%`, average whole-GPU utilization `30.14%`, recent-window CPU/GPU averages `27.91%`/`23.01%`, and peak GPU memory `3471 MB`. TensorBoard reward and loss summaries remained finite; latest success rates were Normal `1.00`, Complex `0.90`, Extrem `0.70`, and DLP `0.91`. Latest `step_num=96`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 12:56 confirmed the run is still below the stop target: 19,112 episodes, 1,762,101 environment steps, estimated 175,186 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 8,712 samples, average monitor CPU `38.41%`, average whole-GPU utilization `29.98%`, recent-window CPU/GPU averages `29.04%`/`22.91%`, peak whole-GPU utilization `100%`, and current TensorBoard success rates were Normal `0.98`, Complex `0.96`, Extrem `0.73`, and DLP `0.89`. Latest `step_num=47`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 13:10 confirmed the run is still below the stop target: 19,381 episodes, 1,781,253 environment steps, estimated 177,101 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 8,870 samples, average monitor CPU `38.20%`, average whole-GPU utilization `29.83%`, recent-window CPU/GPU averages `26.58%`/`23.07%`, and current TensorBoard success rates were Normal `0.99`, Complex `0.94`, Extrem `0.74`, and DLP `0.80`. Latest `step_num=92`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 13:25 confirmed the run is still below the stop target: 19,739 episodes, 1,803,577 environment steps, estimated 179,333 SAC updates after warmup, latest saved training checkpoint `SAC_17999.pt`, `SAC_19999.pt` not yet present, and launcher/workload/monitor processes still running without monitor restart. Resource CSV had 9,049 samples, average monitor CPU `37.99%`, average whole-GPU utilization `29.63%`, recent-window CPU/GPU averages `27.34%`/`19.41%`, and current TensorBoard success rates were Normal `1.00`, Complex `0.97`, Extrem `0.76`, and DLP `0.80`. Latest `step_num=36`; no stop action was taken.
  - 20K stop-hook heartbeat at 2026-06-19 13:40 confirmed the stop condition was met: 20,038 episodes, 1,824,904 environment steps, latest saved checkpoint `SAC_19999.pt` at `D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt`, and TensorBoard budget met. The workload PID `11136`, launcher PID `9300`, and resource monitor PID `14536` were stopped externally after checkpoint preservation.
  - Updated the run metadata JSON to mark the user-superseded 20K smoke stop, set the 20K gate to `stopped_for_smoke_eval`, and mark the 36.5K/100K gates as superseded for this run.
  - Ran matched 200-episode evaluations for command-only `SAC_19999.pt` and baseline `D:\Github\HOPE\src\log\exp\sac_20260617_233812\SAC_19999.pt`.
  - Command-only eval directory: `D:\Github\HOPE\src\log\eval\20260619_134153`.
  - Baseline 20K eval directory: `D:\Github\HOPE\src\log\eval\20260619_135349`.
  - Matched eval results were identical: Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`, mean `0.88625`. This supports the judgment that `--visualize= --verbose=` did not negatively affect training quality at the 20K smoke point.
  - Compared against the stopped 36.5K baseline with the required caveat that 20K is not a full equivalence proof: Normal `-0.015`, Complex `-0.040`, Extrem `-0.260`, DLP `+0.005`, mean `-0.07750`.
  - Updated `docs/research/2026-06-19-stage3-command-only-100k-gated-run.md` with final stop, TensorBoard, resource, eval, caveat, quality judgment, and early-warning notes.
  - Updated `docs/research/2026-06-19-stage3-training-architecture-speed-research.md` from pending to research-only active and added the safe research plan.
  - Updated `task_plan.md` and `findings.md` to close Phase 5 and start Phase 6 research-only work.
- Files created/modified:
  - `docs/research/2026-06-19-stage3-command-only-100k-gated-run.md` created.
  - `docs/research/2026-06-19-stage3-training-architecture-speed-research.md` updated.
  - `docs/research/stage3_baseline_gate_references_20260619.json` created.
  - `docs/research/stage3_baseline_gate_references_20260619.md` created.
  - `src/log/exp/stage3_command_only_100k_20260619_004315.meta.json` updated.
  - `task_plan.md` updated.
  - `findings.md` updated.
  - `progress.md` updated.

### Stage 3: 1000-Episode Command-Only Parameter Smoke

- **Status:** complete
- **Started:** 2026-06-18
- Actions taken:
  - The user approved a small parameter-only acceleration test using the original training script and a 1000-episode budget.
  - Scope is limited to training speed and resource-consumption observation; this run is not a 35K/40K training-quality validation.
  - Planned command-only change: use empty-string bool arguments for `--verbose "" --visualize ""` so the original argparse `type=bool` path resolves them to `False`.
  - Ran `train_HOPE_sac.py` unchanged with `--train_episode 1000 --eval_episode 1 --visualize= --verbose=`.
  - Detected the Windows `.venv` launcher PID trap and switched resource monitoring to the real child Python workload PID.
  - Completed 1,000 TensorBoard training episodes and 127,470 environment steps.
  - Measured 2,261.16 seconds from launch to the last training scalar: `1592.10` episodes/hour and `56.37` environment steps/second.
  - Measured workload resource use: 442 samples, average normalized CPU `44.74%`, average whole-GPU utilization `36.75%`, peak whole-GPU memory `3519 MB`.
  - Compared against the original baseline's first 1,000 episodes: command-only was `3.46x` faster by episodes/hour and `3.76x` faster by environment steps/second.
  - Recorded that this validates the speed direction only; it does not validate 35K/40K training quality.
- Files created/modified:
  - `docs/research/2026-06-19-stage3-command-only-1000-smoke.md` created.
  - `docs/research/2026-06-18-stage3-safe-acceleration-research.md` updated.
  - `task_plan.md` updated to mark the short smoke complete while leaving full quality validation pending.
  - `findings.md` updated with command-only smoke results.
  - `progress.md` updated with this session entry.

### Stage 3: 36.5K Baseline Closure And Safe Acceleration Research

- **Status:** complete
- **Started:** 2026-06-18
- Actions taken:
  - Stopped the original SAC baseline at `36502` episodes after the user requested a 36,500-episode baseline stop.
  - Stopped the training child process, launcher process, and external resource monitor.
  - Deleted the obsolete heartbeat automation `hope-40k-baseline-monitor`.
  - Exported final TensorBoard summary files at `src/log/exp/stage3_baseline_36500_stop.tb_summary.json` and `.md`.
  - Computed final resource profile: 16,963 samples over 23.84 hours, average normalized process CPU `32.38%`, average whole-GPU utilization `25.76%`, and peak whole-GPU memory `3740 MB`.
  - Ran original evaluation on `src/log/exp/sac_20260617_233812/SAC_35999.pt` with `eval_episode 200`.
  - Recorded final evaluation rates: Normal `1.00`, Complex `0.985`, Extrem `0.915`, DLP `0.955`.
  - Ran a no-source-change 500-step micro-profile; main costs were `env.step_total`, `env.render_total`, `env.rs_probe`, `agent.get_action`, `env.action_mask`, and `agent.update`.
  - Ran a render-mode parity check for `render_mode=None` versus `render_mode='rgb_array'` on Normal, Complex, Extrem, and DLP scripted cases; observations, rewards, and statuses matched exactly in the tested cases.
  - Wrote `docs/research/2026-06-17-stage3-baseline-resource-profile.md`.
  - Updated `docs/research/2026-06-18-stage3-safe-acceleration-research.md`, `docs/research/2026-06-17-stage3-40k-baseline-live-run.md`, `task_plan.md`, and `findings.md`.
- Files created/modified:
  - `docs/research/2026-06-17-stage3-baseline-resource-profile.md` created.
  - `docs/research/2026-06-18-stage3-safe-acceleration-research.md` updated.
  - `docs/research/2026-06-17-stage3-40k-baseline-live-run.md` updated.
  - `task_plan.md` updated.
  - `findings.md` updated.
  - `progress.md` updated.

### Stage 3: Safe Training Acceleration Research

- **Status:** complete
- **Started:** 2026-06-18
- Actions taken:
  - Re-read `task_plan.md`, `findings.md`, `progress.md`, the live Stage 3 baseline note, and the active TensorBoard/resource summaries.
  - Exported a fresh live TensorBoard summary for `D:\Github\HOPE\src\log\exp\sac_20260617_233812`: 28,525 episodes, `training_budget_met=false`, `env_step_count=2326377`, finite actor/critic losses, and improving reward/success trends.
  - Recomputed resource CSV statistics with the real monitor column names: 12,012 samples, average normalized process CPU about `37.14%`, peak `49.09%`, average whole-GPU utilization about `28.0%`, peak `97%`, and peak whole-GPU memory about `3740 MB`.
  - Inspected original SAC training, SAC update, replay memory, network, wrapper, environment, lidar, action mask, and image-processing code paths.
  - Compared the current original baseline against the old 2026-04 run summaries under `src/log/exp/sac_20260418_165251`.
  - Recorded new findings about why large batch/layer changes are risky and why safe acceleration should target external overhead and measured environment bottlenecks first.
  - Answered the user's near-30K sufficiency question: 30K is enough for preliminary resource/bottleneck judgment, but not enough to close the requested 40K baseline or final reproduction-quality conclusion.
  - Wrote `docs/research/2026-06-18-stage3-safe-acceleration-research.md`.
- Files created/modified:
  - `docs/research/2026-06-18-stage3-safe-acceleration-research.md` created.
  - `task_plan.md` updated with safe-acceleration decisions.
  - `findings.md` updated with current run, old run, and code-path evidence.
  - `progress.md` updated with this session.

### Stage 3: Budget Correction And Diagnostic Record

- **Status:** in progress
- **Started:** 2026-06-17
- Actions taken:
  - Corrected Stage 3 budget language from environment-step wording to 40,000 training episodes.
  - Rewrote the Stage 3 execution plan to use `episode_count` / `training_budget_met` and to treat `env_step_count` as an observational metric.
  - Recorded the earlier 300-episode run as a diagnostic dry run only.
  - Preserved the boundary that original HOPE source files under `src/` are not modified for Stage 3 performance/resource work without explicit approval of a separate source-change plan.
  - Started the true original SAC baseline with `--train_episode 40000 --eval_episode 200`.
  - Locked the active TensorBoard run directory to `D:\Github\HOPE\src\log\exp\sac_20260617_233812` in runtime metadata.
  - Captured an early live snapshot: 103 episodes, `training_budget_met=false`, `env_step_count=12193`, and 35 resource samples.
  - Captured a later immediate check: 198 episodes, `training_budget_met=false`, `env_step_count=21551`, and 74 resource samples.
  - Created a 30-minute heartbeat monitor for the ongoing 40,000-episode baseline run.
  - Heartbeat check at 2026-06-18 00:13 confirmed training and monitoring processes were still running: 938 episodes, `training_budget_met=false`, `env_step_count=119667`, 422 resource samples, average CPU `43.84`, and average whole-GPU `31.06`.
  - Heartbeat check at 2026-06-18 00:43 confirmed training and monitoring processes were still running: 1812 episodes, `training_budget_met=false`, `env_step_count=219445`, 778 resource samples, average CPU `45.05`, and average whole-GPU `31.01`.
  - Heartbeat check at 2026-06-18 01:13 confirmed training and monitoring processes were still running: 2681 episodes, `training_budget_met=false`, `env_step_count=317285`, 1136 resource samples, average CPU `45.37`, average whole-GPU `30.88`, and a `step_num=200` early-warning sample to keep watching.
  - Heartbeat check at 2026-06-18 01:43 confirmed training and monitoring processes were still running: 3566 episodes, `training_budget_met=false`, `env_step_count=412868`, 1492 resource samples, average CPU `45.46`, average whole-GPU `31.08`, and no repeated latest-step timeout warning.
  - Heartbeat check at 2026-06-18 02:13 confirmed training and monitoring processes were still running: 4438 episodes, `training_budget_met=false`, `env_step_count=506112`, 1848 resource samples, average CPU `45.45`, average whole-GPU `31.06`, and no repeated latest-step timeout warning.
  - Heartbeat check at 2026-06-18 02:43 confirmed training and monitoring processes were still running: 5373 episodes, `training_budget_met=false`, `env_step_count=603081`, 2223 resource samples, average CPU `45.39`, average whole-GPU `31.13`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 03:13 confirmed training and monitoring processes were still running: 6174 episodes, `training_budget_met=false`, `env_step_count=684683`, 2561 resource samples, average CPU `45.26`, average whole-GPU `31.09`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 03:43 confirmed training and monitoring processes were still running: 7083 episodes, `training_budget_met=false`, `env_step_count=771519`, 2917 resource samples, average CPU `45.13`, average whole-GPU `31.12`, and the latest `step_num=200` timeout warning recurred.
  - Heartbeat check at 2026-06-18 04:13 confirmed training and monitoring processes were still running: 7947 episodes, `training_budget_met=false`, `env_step_count=855907`, 3274 resource samples, average CPU `44.97`, average whole-GPU `31.26`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 04:43 confirmed training and monitoring processes were still running: 8826 episodes, `training_budget_met=false`, `env_step_count=938316`, 3630 resource samples, average CPU `44.81`, average whole-GPU `31.27`, and the latest `step_num=200` timeout warning recurred.
  - Heartbeat check at 2026-06-18 05:13 confirmed training and monitoring processes were still running: 9687 episodes, `training_budget_met=false`, `env_step_count=1018732`, 3986 resource samples, average CPU `44.62`, average whole-GPU `31.25`, and the latest `step_num=200` timeout warning recurred.
  - Recovered the Stage 3 resource monitor after PID `25356` exited while training continued: made `tools/stage3/monitor_stage3_resources.ps1` append-safe for existing CSV files, restarted monitoring as PID `34596`, updated run metadata, and verified the CSV reached 3991 samples.
  - Heartbeat check at 2026-06-18 05:43 confirmed training and the restarted monitor were still running: 10532 episodes, `training_budget_met=false`, `env_step_count=1098069`, 4323 resource samples, average CPU `44.47`, average whole-GPU `31.25`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 06:13 confirmed training and the restarted monitor were still running: 11382 episodes, `training_budget_met=false`, `env_step_count=1175578`, 4679 resource samples, average CPU `44.26`, average whole-GPU `31.19`, and the latest `step_num=200` timeout warning recurred.
  - Heartbeat check at 2026-06-18 06:43 confirmed training and the restarted monitor were still running: 12361 episodes, `training_budget_met=false`, `env_step_count=1255349`, 5056 resource samples, average CPU `44.01`, average whole-GPU `31.11`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 07:13 confirmed training and the restarted monitor were still running: 13217 episodes, `training_budget_met=false`, `env_step_count=1324321`, 5392 resource samples, average CPU `43.78`, average whole-GPU `30.96`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 07:43 confirmed training and the restarted monitor were still running: 14112 episodes, `training_budget_met=false`, `env_step_count=1396127`, 5748 resource samples, average CPU `43.52`, average whole-GPU `30.98`, and the latest `step_num=200` timeout warning recurred.
  - Heartbeat check at 2026-06-18 08:13 confirmed training and the restarted monitor were still running: 15053 episodes, `training_budget_met=false`, `env_step_count=1465661`, 6105 resource samples, average CPU `43.24`, average whole-GPU `30.86`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 08:43 confirmed training and the restarted monitor were still running: 15903 episodes, `training_budget_met=false`, `env_step_count=1533805`, 6462 resource samples, average CPU `42.97`, average whole-GPU `30.84`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 09:13 confirmed training and the restarted monitor were still running: 16772 episodes, `training_budget_met=false`, `env_step_count=1599400`, 6818 resource samples, average CPU `42.66`, average whole-GPU `30.67`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 09:43 confirmed training and the restarted monitor were still running: 17701 episodes, `training_budget_met=false`, `env_step_count=1663404`, 7175 resource samples, average CPU `42.35`, average whole-GPU `30.58`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 10:13 confirmed training and the restarted monitor were still running: 18587 episodes, `training_budget_met=false`, `env_step_count=1724598`, 7532 resource samples, average CPU `42.01`, average whole-GPU `30.34`, and no repeated latest-step timeout warning at this snapshot.
  - Recovered the Stage 3 resource monitor again after PID `34596` exited while training continued: restarted monitoring as PID `19512`, updated run metadata, and verified the CSV reached 7537 samples.
  - Heartbeat check at 2026-06-18 10:43 confirmed the launcher, training process, and restarted monitor were still running: 19469 episodes, `training_budget_met=false`, `env_step_count=1785915`, 7888 resource samples, average CPU `41.66`, average whole-GPU `30.04`, latest scene success rates Normal `1.00`, Complex `0.95`, Extrem `0.79`, DLP `0.81`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 11:13 confirmed the launcher, training process, and restarted monitor were still running: 20260 episodes, `training_budget_met=false`, `env_step_count=1838002`, 8228 resource samples, average CPU `41.27`, average whole-GPU `29.86`, latest scene success rates Normal `1.00`, Complex `0.91`, Extrem `0.75`, DLP `0.84`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 11:43 confirmed the launcher, training process, and restarted monitor were still running: 21126 episodes, `training_budget_met=false`, `env_step_count=1893703`, 8585 resource samples, average CPU `40.91`, average whole-GPU `29.65`, latest scene success rates Normal `1.00`, Complex `0.94`, Extrem `0.75`, DLP `0.87`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 12:13 confirmed the launcher, training process, and restarted monitor were still running: 22035 episodes, `training_budget_met=false`, `env_step_count=1948185`, 8943 resource samples, average CPU `40.55`, average whole-GPU `29.38`, latest scene success rates Normal `1.00`, Complex `0.96`, Extrem `0.75`, DLP `0.86`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 12:43 confirmed the launcher, training process, and restarted monitor were still running: 22851 episodes, `training_budget_met=false`, `env_step_count=2000424`, 9299 resource samples, average CPU `40.19`, average whole-GPU `29.17`, latest scene success rates Normal `0.99`, Complex `0.97`, Extrem `0.81`, DLP `0.87`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 13:13 confirmed the launcher, training process, and restarted monitor were still running: 23634 episodes, `training_budget_met=false`, `env_step_count=2047873`, 9655 resource samples, average CPU `39.77`, average whole-GPU `28.97`, latest scene success rates Normal `0.99`, Complex `0.94`, Extrem `0.79`, DLP `0.82`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 13:43 confirmed the launcher, training process, and restarted monitor were still running: 24478 episodes, `training_budget_met=false`, `env_step_count=2094869`, 10013 resource samples, average CPU `39.37`, average whole-GPU `28.67`, latest scene success rates Normal `0.99`, Complex `0.98`, Extrem `0.83`, DLP `0.81`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 14:13 confirmed the launcher, training process, and restarted monitor were still running: 25219 episodes, `training_budget_met=false`, `env_step_count=2139060`, 10369 resource samples, average CPU `38.96`, average whole-GPU `28.43`, latest scene success rates Normal `1.00`, Complex `0.98`, Extrem `0.81`, DLP `0.89`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 14:43 confirmed the launcher, training process, and restarted monitor were still running: 26042 episodes, `training_budget_met=false`, `env_step_count=2184733`, 10726 resource samples, average CPU `38.59`, average whole-GPU `28.18`, latest scene success rates Normal `0.98`, Complex `0.98`, Extrem `0.84`, DLP `0.87`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 15:13 confirmed the launcher, training process, and restarted monitor were still running: 26803 episodes, `training_budget_met=false`, `env_step_count=2226640`, 11081 resource samples, average CPU `38.19`, average whole-GPU `27.99`, latest scene success rates Normal `0.99`, Complex `0.97`, Extrem `0.92`, DLP `0.81`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 15:43 confirmed the launcher, training process, and restarted monitor were still running: 27466 episodes, `training_budget_met=false`, `env_step_count=2266633`, 11438 resource samples, average CPU `37.79`, average whole-GPU `28.02`, latest scene success rates Normal `0.97`, Complex `0.99`, Extrem `0.85`, DLP `0.89`, and the latest `step_num=200` timeout warning recurred.
  - Heartbeat check at 2026-06-18 16:13 confirmed the launcher, training process, and restarted monitor were still running: 28154 episodes, `training_budget_met=false`, `env_step_count=2304733`, 11795 resource samples, average CPU `37.38`, average whole-GPU `28.07`, latest scene success rates Normal `0.99`, Complex `0.98`, Extrem `0.92`, DLP `0.80`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 16:43 confirmed the launcher, training process, and restarted monitor were still running: 28767 episodes, `training_budget_met=false`, `env_step_count=2340737`, 12152 resource samples, average CPU `36.97`, average whole-GPU `27.97`, latest scene success rates Normal `0.99`, Complex `0.99`, Extrem `0.91`, DLP `0.76`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 17:13 confirmed the launcher, training process, and restarted monitor were still running: 29368 episodes, `training_budget_met=false`, `env_step_count=2374104`, 12508 resource samples, average CPU `36.55`, average whole-GPU `28.01`, latest scene success rates Normal `1.00`, Complex `0.96`, Extrem `0.89`, DLP `0.81`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 17:43 confirmed the launcher, training process, and restarted monitor were still running: 29941 episodes, `training_budget_met=false`, `env_step_count=2408546`, 12868 resource samples, average CPU `36.15`, average whole-GPU `27.81`, latest scene success rates Normal `0.97`, Complex `0.97`, Extrem `0.83`, DLP `0.79`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 18:13 confirmed the launcher, training process, and restarted monitor were still running and crossed 30,000 episodes: 30520 episodes, `training_budget_met=false`, `env_step_count=2445989`, 13236 resource samples, average CPU `35.80`, average whole-GPU `27.63`, latest scene success rates Normal `1.00`, Complex `0.98`, Extrem `0.84`, DLP `0.75`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 18:43 confirmed the launcher, training process, and restarted monitor were still running: 31070 episodes, `training_budget_met=false`, `env_step_count=2477760`, 13576 resource samples, average CPU `35.46`, average whole-GPU `27.38`, latest scene success rates Normal `0.99`, Complex `0.99`, Extrem `0.92`, DLP `0.79`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 19:13 confirmed the launcher, training process, and restarted monitor were still running: 31654 episodes, `training_budget_met=false`, `env_step_count=2509566`, 13933 resource samples, average CPU `35.09`, average whole-GPU `27.05`, latest scene success rates Normal `0.99`, Complex `1.00`, Extrem `0.91`, DLP `0.85`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 19:43 confirmed the launcher, training process, and restarted monitor were still running: 32249 episodes, `training_budget_met=false`, `env_step_count=2541512`, 14290 resource samples, average CPU `34.75`, average whole-GPU `26.75`, latest scene success rates Normal `1.00`, Complex `0.97`, Extrem `0.90`, DLP `0.89`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 20:13 confirmed the launcher, training process, and restarted monitor were still running: 32750 episodes, `training_budget_met=false`, `env_step_count=2570732`, 14647 resource samples, average CPU `34.39`, average whole-GPU `26.49`, latest scene success rates Normal `0.98`, Complex `0.98`, Extrem `0.89`, DLP `0.80`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 20:43 confirmed the launcher, training process, and restarted monitor were still running: 33291 episodes, `training_budget_met=false`, `env_step_count=2599153`, 15002 resource samples, average CPU `34.05`, average whole-GPU `26.20`, latest scene success rates Normal `0.99`, Complex `1.00`, Extrem `0.89`, DLP `0.84`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 21:13 confirmed the launcher, training process, and restarted monitor were still running: 33826 episodes, `training_budget_met=false`, `env_step_count=2628426`, 15360 resource samples, average CPU `33.72`, average whole-GPU `25.96`, latest scene success rates Normal `1.00`, Complex `0.99`, Extrem `0.86`, DLP `0.83`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 21:43 confirmed the launcher, training process, and restarted monitor were still running: 34489 episodes, `training_budget_met=false`, `env_step_count=2657445`, 15715 resource samples, average CPU `33.41`, average whole-GPU `25.78`, latest scene success rates Normal `0.99`, Complex `0.98`, Extrem `0.93`, DLP `0.89`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 22:13 confirmed the launcher, training process, and restarted monitor were still running: 35111 episodes, `training_budget_met=false`, `env_step_count=2687415`, 16077 resource samples, average CPU `33.12`, average whole-GPU `25.80`, latest scene success rates Normal `0.99`, Complex `0.99`, Extrem `0.90`, DLP `0.85`, and no repeated latest-step timeout warning at this snapshot.
  - Heartbeat check at 2026-06-18 22:43 confirmed the launcher, training process, and restarted monitor were still running: 35595 episodes, `training_budget_met=false`, `env_step_count=2714233`, 16435 resource samples, average CPU `32.81`, average whole-GPU `25.82`, latest scene success rates Normal `0.99`, Complex `0.98`, Extrem `0.90`, DLP `0.83`, and the latest `step_num=200` timeout warning recurred while loss/reward trends remained healthy.
  - Heartbeat check at 2026-06-18 23:13 confirmed the launcher, training process, and restarted monitor were still running: 36206 episodes, `training_budget_met=false`, `env_step_count=2741287`, 16787 resource samples, average CPU `32.53`, average whole-GPU `25.83`, latest scene success rates Normal `0.99`, Complex `1.00`, Extrem `0.94`, DLP `0.92`, and no repeated latest-step timeout warning at this snapshot.
- Files created/modified:
  - `AGENTS.md` updated.
  - `task_plan.md` updated.
  - `findings.md` updated.
  - `progress.md` updated.
  - `docs/superpowers/specs/2026-06-17-hope-stage3-training-resource-design.md` updated.
  - `docs/superpowers/plans/2026-06-17-hope-stage3-training-resource-study.md` rewritten.
  - `docs/research/2026-06-17-stage3-diagnostic-dry-run.md` created.
  - `docs/research/2026-06-17-stage3-40k-baseline-live-run.md` created.

### Stage 3: Executable Planning

- **Status:** complete
- **Started:** 2026-06-17
- Actions taken:
  - Read `superpowers:writing-plans` skill instructions.
  - Re-read Stage 3 boundaries in `AGENTS.md`, `task_plan.md`, `findings.md`, and the Stage 3 design note.
  - Inspected original HOPE SAC training, environment, evaluation, replay memory, and agent code paths.
  - Created a task-by-task Stage 3 execution plan with 40,000-episode verification, resource monitoring, TensorBoard summary export, baseline reporting, and bottleneck gate.
  - Corrected the Stage 3 budget definition after user clarification: the budget gate is 40,000 training episodes, not summed environment interaction steps.
  - Verified the new plan has no reserved incomplete-marker matches.
  - Verified `git diff -- src` remains empty.
- Files created/modified:
  - `docs/superpowers/plans/2026-06-17-hope-stage3-training-resource-study.md` created.
  - `task_plan.md` updated with the executable-plan checkpoint.
  - `findings.md` updated with the plan decision.
  - `progress.md` updated with this session.

### Stage 3: Boundary And Resource-Study Requirements

- **Status:** complete
- **Started:** 2026-06-17
- Actions taken:
  - Read the requested `planning-with-files`, `using-superpowers`, and `brainstorming` skill files.
  - Ran planning session catchup and confirmed the prior Stage 2 summary was already reflected in files.
  - Re-read `AGENTS.md`, `task_plan.md`, `findings.md`, and `progress.md`.
  - Captured the user's added Stage 3 requirements: resource utilization, bottleneck research, quality-preserving acceleration, original-framework preservation, 40,000-episode smoke/validation budget, and TensorBoard failure-mode monitoring.
  - Updated `AGENTS.md`, `task_plan.md`, and `findings.md` with Stage 3 boundaries.
  - Added a Stage 3 design note for the resource-utilization study.
  - Verified no training source files under `src/` were changed.
- Files created/modified:
  - `AGENTS.md` updated.
  - `task_plan.md` updated for Stage 3.
  - `findings.md` updated with Stage 3 requirements.
  - `progress.md` updated with this session.
  - `docs/superpowers/specs/2026-06-17-hope-stage3-training-resource-design.md` created.

### Stage 2: Checkpoint Validation

- **Status:** complete
- **Started:** 2026-06-17
- Actions taken:
  - Re-read planning files and `AGENTS.md`.
  - Listed author-provided checkpoints under `src/model/ckpt/`.
  - Reframed `task_plan.md` for Stage 2 checkpoint validation.
  - Ran first `HOPE_SAC0.pt` evaluation attempt; checkpoint load failed due PyTorch `weights_only=True` default.
  - Reran `HOPE_SAC0.pt` with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1`; evaluation completed.
  - Read result files for the SAC0 validation run.
  - Ran quick `HOPE_SAC1.pt` validation with `eval_episode 3`; evaluation completed.
  - Ran quick `HOPE_PPO.pt` validation with `eval_episode 3`; evaluation completed.
  - Read result files for SAC1 and PPO validation runs.
  - Wrote `docs/research/2026-06-17-stage2-checkpoint-validation.md`.
  - Verified `git diff -- src` is empty after Stage 2.
- Files created/modified:
  - `task_plan.md` updated for Stage 2.
  - `findings.md` updated with Stage 2 requirements and checkpoint inventory.
  - `progress.md` updated with Stage 2 session log.
  - `docs/research/2026-06-17-stage2-checkpoint-validation.md` created.

### Phase 1: Planning Files And Baseline Discovery

- **Status:** complete
- **Started:** 2026-06-17
- Actions taken:
  - Read `planning-with-files` skill instructions.
  - Ran session catchup for the current repository; no unsynced context was reported.
  - Read the planning file templates.
  - Created root-level `task_plan.md`, `findings.md`, and `progress.md`.
  - Inspected Python launcher versions, default Python version, Conda availability, GPU/CUDA status, `.gitignore`, and `requirements.txt`.
  - Checked for `uv`; it was not installed.
  - Added `.venv/` to `.gitignore`.
  - Created `.venv` using Python 3.13.
  - Upgraded base packaging tools inside `.venv`.
  - Installed `requirements.txt` into `.venv`.
  - Checked PyTorch official installation documentation for Windows/Python/CUDA compatibility.
  - Installed `torch 2.11.0+cu128` into `.venv` from the official PyTorch CUDA 12.8 wheel index.
  - Ran dependency consistency, key import, CUDA, and environment reset checks.
  - Verified `.venv` is ignored by Git and `src/` has no diff.
  - Wrote `docs/research/2026-06-17-stage1-environment-setup.md`.
  - Ran final verification commands for dependency consistency, CUDA, HOPE reset, docs, ignore rules, reserved token scan, and Git status.
- Files created/modified:
  - `task_plan.md` created.
  - `findings.md` created.
  - `progress.md` created.
  - `.gitignore` modified to ignore `.venv/`.
  - `docs/research/2026-06-17-stage1-environment-setup.md` created.

## Test Results

| Test | Input | Expected | Actual | Status |
|------|-------|----------|--------|--------|
| Planning session catchup | `python "$env:USERPROFILE\.codex\skills\planning-with-files\scripts\session-catchup.py" (Get-Location)` | No blocking unsynced context | No output | Pass |
| Runtime discovery | `py -0p`, `python --version`, `Get-Command conda`, `nvidia-smi` | Identify available project environment options | Python 3.13/3.14 only via launcher; no Conda; RTX 4080-class GPU visible | Pass |
| Create environment | `py -3.13 -m venv .venv` | Project-local virtual environment is created | Command exited 0 | Pass |
| Upgrade packaging tools | `.venv\Scripts\python.exe -m pip install --upgrade pip setuptools wheel` | Tools install inside `.venv` | Installed pip 26.1.2, setuptools 82.0.1, wheel 0.47.0 | Pass |
| Install requirements | `.venv\Scripts\python.exe -m pip install -r requirements.txt` | All listed project packages install into `.venv` | Command exited 0; gym built wheel successfully | Pass |
| Install PyTorch | `.venv\Scripts\python.exe -m pip install torch --index-url https://download.pytorch.org/whl/cu128` | Torch installs into `.venv` | Installed `torch 2.11.0+cu128`; `setuptools` adjusted to 70.2.0 | Pass |
| Dependency consistency | `.venv\Scripts\python.exe -m pip check` | No broken requirements | `No broken requirements found.` | Pass |
| Key imports | Import `numpy`, `shapely`, `pygame`, `gym`, `cv2`, `scipy`, `torch` | Imports succeed from `.venv` | Imports succeeded; versions recorded in setup note | Pass |
| PyTorch CUDA | Print `torch.cuda.is_available()` and device | CUDA available on local GPU | `True`; device `NVIDIA GeForce RTX 4080 SUPER` | Pass |
| HOPE environment reset | `w.reset(None, None, 'Normal')` from `D:\Github\HOPE\src` | Observation dict returned | Returned `img`, `lidar`, `target`, `action_mask` with expected shapes | Pass |
| Git ignore check | `git check-ignore -v .venv\pyvenv.cfg` | `.venv` ignored | `.gitignore:12:.venv/` | Pass |
| Source diff check | `git diff -- src` | No source code changes | No output | Pass |
| Reserved token scan | Reserved incomplete-marker scan across planning docs | No incomplete-marker matches | No matches | Pass |
| SAC0 checkpoint smoke evaluation | `eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 10 --visualize False` with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Evaluation completes | Completed; success rate `1.0` for extreme, dlp, complex, normalize result files | Pass |
| SAC1 checkpoint sanity evaluation | `eval_mix_scene.py .\model\ckpt\HOPE_SAC1.pt --eval_episode 3 --visualize False` with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Evaluation completes | Completed; success rate `1.0` for four evaluation segments | Pass |
| PPO checkpoint sanity evaluation | `eval_mix_scene.py .\model\ckpt\HOPE_PPO.pt --eval_episode 3 --visualize False` with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` | Evaluation completes | Completed; success rate `1.0` for four evaluation segments | Pass |

## Error Log

| Timestamp | Error | Attempt | Resolution |
|-----------|-------|---------|------------|
| 2026-06-17 | `_pickle.UnpicklingError: Weights only load failed` while loading `HOPE_SAC0.pt` | 1 | PyTorch 2.6+ changed `torch.load` default. Retry with `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` for this trusted repository checkpoint. |
| 2026-06-17 | `TypeError: CarParkingWrapper.reset() got an unexpected keyword argument 'level'` | 1 | Command issue, not source issue. `CarParkingWrapper.reset()` passes positional `*args`; retry with `w.reset(None, None, 'Normal')`. |
| 2026-06-18 | `tensorboard_stage3_summary.py` was called once without required `--json` and `--markdown` outputs | 1 | Reran the same summary against the locked run directory with explicit JSON and Markdown output paths. |
| 2026-06-18 | Stage 3 resource monitor PID `25356` exited while the training PID was still active | 1 | Made the monitor script append-safe for existing CSV files, restarted monitoring as PID `34596`, updated metadata, and verified new samples appended. |
| 2026-06-19 | Heartbeat monitoring command used a malformed `.venv` path and then passed unsupported `--run-dir` to `tensorboard_stage3_summary.py` | 1 | Reran with absolute `D:\Github\HOPE\.venv\Scripts\python.exe` and the script's positional `log_dir` argument; training and resource monitor were unaffected. |
| 2026-06-19 | While exploring TensorBoard JSON, attempted to inspect a non-existent `scalar_summaries` property | 1 | Read the actual schema and used `scalars.<tag>` entries for reward, loss, success-rate, and step summaries. |
| 2026-06-19 | A PowerShell comparison helper used invalid dynamic property syntax with inline `if`, so the eval comparison rows were empty on the first try | 1 | Re-ran the helper with an explicit scene-to-property map; final comparison output was complete. |

### Phase 9: Opt-In Fast Action Mask

- **Status:** complete and pushed to `origin/codex/stage3-resource-study`.
- **Started:** 2026-06-19
- Actions taken:
  - Read current planning context and confirmed the working tree was clean on `codex/stage3-resource-study`.
  - Created `docs/superpowers/plans/2026-06-19-hope-stage3-fast-action-mask.md`.
  - Wrote failing TDD tests in `tools/stage3/tests/test_fast_action_mask.py` and `tools/stage3/tests/test_profile_stage3_components.py`.
  - Verified red tests:
    - `ActionMask.__init__()` rejected missing `fast_get_steps`.
    - profiler tests failed because `profile_action_mask_settings` was not implemented.
  - Implemented default-off `fast_get_steps` dispatch in `src/model/action_mask.py`.
  - Kept the original implementation in `_get_steps_original()` and added `_get_steps_fast()` as a separate explicit branch.
  - Added profiler `--action-mask-mode {original,fast}` for bounded diagnostics.
  - Ran full Stage 3 tool tests: 37 tests passed.
  - Ran bounded fast action-mask profile and same-budget original action-mask profile.
  - Wrote `docs/research/2026-06-19-stage3-fast-action-mask-opt-in.md`.
  - Committed `40292ab Add opt-in fast action mask` and pushed it to the draft PR branch.
  - Committed a follow-up planning-state sync after the push so root planning files no longer report the phase as pending.
- Key measurement:
  - Original action mask profile: `1.694 ms` action-mask avg, `9.345 ms` env-step avg.
  - Fast action mask profile: `0.821 ms` action-mask avg, `8.205 ms` env-step avg.
- Caveat:
  - This is only bounded measurement evidence. It does not replace the future 20K gated candidate run against the command-only 20K baseline.

## 5-Question Reboot Check

| Question | Answer |
|----------|--------|
| Where am I? | Stage 3 safe-speed 20K framework implementation and Task 10 final verification are complete. |
| Where am I going? | Final review and user handoff for the verified safe-speed 20K framework. |
| What's the goal? | Preserve the original HOPE baseline while using the safe-speed 20K framework to evaluate future speed candidates with quality gates. |
| What have I learned? | The verified framework can summarize TensorBoard, parse eval/resource artifacts, run parity/profile smoke checks, compare 20K candidates, and protect original HOPE source paths. |
| What have I done? | Implemented and verified Tasks 1-10 from the safe-speed 20K framework plan. |

### Phase 10: Fast Action Mask 20K Candidate Run

- **Status:** complete; passed 20K speed and quality gates against the saved command-only 20K baseline.
- **Started:** 2026-06-20
- Actions taken:
  - Recovered planning context and confirmed branch `codex/stage3-resource-study` was clean before new changes.
  - Confirmed existing `launch_stage3_20k.ps1` did not enable the default-off fast action-mask branch.
  - Added a failing TDD test for a training wrapper that enables `ActionMask()` fast mode by default while respecting explicit `fast_get_steps=False`.
  - Implemented `tools/stage3/train_HOPE_sac_fast_action_mask.py` as an external opt-in wrapper around the original `src/train/train_HOPE_sac.py`.
  - Added `-FastActionMask` to `tools/stage3/launch_stage3_20k.ps1`.
  - Updated `tools/stage3/stop_stage3_at_20k.ps1` safety fragment from `train_HOPE_sac.py` to `train_HOPE_sac` so it can stop either original or fast wrapper training processes.
  - Verified wrapper test passed.
  - Verified PowerShell parser checks for launch and stop scripts passed.
  - Verified `git diff -- src\train src\env` produced no output.
  - Ran full Stage 3 tool tests: 38 tests passed.
  - Committed and pushed `03105b9 Add fast action mask training launcher`.
  - Launched candidate with `tools/stage3/launch_stage3_20k.ps1 -RunName stage3_fast_action_mask_20k -CandidateType fast_action_mask_20k_validation -TrainEpisode 20000 -EvalEpisode 200 -FastActionMask`.
  - Locked manifest: `src/log/exp/stage3_fast_action_mask_20k_20260620_085206.meta.json`.
  - Locked run dir: `src/log/exp/sac_20260620_085208`.
  - Initial workload PID in manifest was venv launcher parent `29844`; actual TensorBoard/training child PID is `3696`.
  - Stopped the original resource monitor on parent PID and restarted it on child PID `3696` as monitor PID `19300`.
  - Updated manifest `workload_pid=3696`, `workload_parent_pid=29844`, `python_executable=C:\Users\zhang\AppData\Local\Programs\Python\Python313\python.exe`, and monitor PID `19300`.
  - Confirmed resource CSV is now appending rows for PID `3696`; first startup rows for PID `29844` are monitor-bias rows and should be excluded or caveated in final resource interpretation.
  - TensorBoard initial live summary reached 52 episodes and 6,222 env steps; `hard_reject_has_nonfinite=False`.
  - Opened TensorBoard in the in-app browser at `http://127.0.0.1:6006/?darkMode=true#timeseries&runSelectionState=eyJzYWNfMjAyNjA2MjBfMDg1MjA4Ijp0cnVlfQ%3D%3D`.
  - Created heartbeat monitor automation `hope-fast-action-mask-20k-monitor` at 15-minute intervals.
  - Monitored the run until TensorBoard reached 20,000 episodes and `src/log/exp/sac_20260620_085208/SAC_19999.pt` existed.
  - Attempted `tools/stage3/stop_stage3_at_20k.ps1`; the stop script rejected the reassigned child PID because PowerShell `ConvertFrom-Json` shifted manifest timestamps through local/UTC semantics in the process-window safety check.
  - Used a manual fallback with equivalent PID/name/command-line validation. The workload PID `3696` and resource monitor PID `19300` were already inactive after natural finish, with logs and checkpoints preserved.
  - Recomputed speed metrics with Python raw JSON timestamp parsing to avoid the same UTC/local conversion problem.
  - Ran matched 200-episode external evaluation for `SAC_19999.pt`; result: Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`, mean `0.88625`.
  - Compared against the command-only 20K baseline using `tools/stage3/compare_stage3_20k.py`; decision `pass`.
  - Final speed gate metrics to 20K checkpoint: `10.947863 h`, `1826.840564` episodes/hour, `46.223481` env steps/second; improvements over baseline were `15.55%`, `18.19%`, and `18.22%`.
  - Resource summary used 8,060 samples: average process CPU `39.67%`, average whole-GPU utilization `32.22%`, average GPU memory used `3130.39 MB`, peak GPU memory `3643 MB`.
  - Wrote comparison artifacts:
    - `docs/research/stage3_fast_action_mask_20k_candidate_20260620.comparison.json`
    - `docs/research/2026-06-20-stage3-fast-action-mask-20k-candidate-report.md`
  - Compared to the stopped 36.5K baseline only as a maturity caveat: the 20K candidate is lower on Extrem and mean success, but it exactly matches the saved 20K baseline, so this is not treated as a fast action-mask regression.
  - Verified this run did not require modifying original `src/train` or `src/env` files; the fast action-mask behavior remains opt-in through the wrapper and default-off in normal code paths.

### OGM Integration Research Refresh

- **Status:** in progress.
- **Started:** 2026-06-20 11:35:54 +08:00
- Actions taken:
  - Read current AGENTS.md instructions from the user message, root planning context, recent Stage 3 findings, and the previous RL-OGM research and implementation plan.
  - Confirmed this pass is research-only: no OGM source implementation should begin while Stage 3 baseline/safe-speed validation remains active, unless the user explicitly approves Stage 4.
  - Confirmed current repository state has a running fast action-mask 20K candidate and one untracked handoff document under `docs/research/`.
  - Identified existing OGM notes as useful but stale relative to the current code state because Stage 3 tooling, fast action-mask opt-in code, and 20K gate workflow now exist.
  - Extracted text from both local PDFs with `pypdf`; `pdfinfo.exe` was not available in the bundled bin path, but text extraction and `pdfplumber` table extraction worked.
  - Re-read current HOPE source attachment points: `CarParking`, `CarParkingWrapper`, `LidarSimlator`, `ActionMask`, `MultiObsEmbedding`, `SACAgent`, `StateNorm`, `ReplayMemory`, `train_HOPE_sac.py`, `eval_utils.py`, and Stage 3 tooling.
  - Wrote `docs/research/2026-06-20-rl-ogm-integration-current-code-research.md`.
  - Appended current OGM integration findings to `findings.md`.
  - Added a root `task_plan.md` note that the OGM refresh is research-only and does not change the active Phase 10 run status.
