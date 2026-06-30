# Task Plan: HOPE Trajectory Style Evaluation Toolkit

## Goal

Turn the current human-like parking trajectory idea into a production-ready, default-off offline toolkit for Stage 4 HOPE+OGM traces.

The P0 product should evaluate whether successful parking trajectories look like plausible human parking behavior, generate scene-conditioned human reference trajectory families, and produce explainable reports for future model/planner experiments.

## Current Phase

P0 implementation and real-trace smoke verification are complete for the trajectory-style evaluation toolkit.

Standalone PRD:

- `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/production_requirement.md`

Implementation plan:

- `docs/superpowers/plans/2026-06-30-trajectory-style-evaluation-toolkit.md`

Research packet:

- `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/research_report.md`
- `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/source_inventory.md`
- `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/attachments/user_annotations/`

Historical planning archive:

- `docs/planning_archive/2026-06-30-pre-trajectory-style-prd/`

## Active Scope

P0 must support:

- Perpendicular parking:
  - enough opposite space: one-shot reverse sweep with late straighten;
  - limited opposite space: staged reverse or planned-cusp reverse.
- Parallel parking:
  - standard slot: reverse S-curve;
  - tight slot: reverse S-curve with planned forward correction.
- Offline trace loading from existing Stage 4 trace JSON.
- Deterministic scene classification with reasons.
- Scene-conditioned reference trajectory family generation.
- Style metrics and diagnosis labels.
- JSON and Markdown reports.

Deferred:

- Angled parking route families.
- RS candidate reranking.
- Reward shaping.
- Auxiliary trajectory heads.
- MPC/OBCA path polishing.
- Imitation-learning training.
- Any checkpoint, policy, reward, or planner behavior change.

## Hard Boundaries

- Do not modify training behavior for this task.
- Do not modify checkpoints or generated training logs.
- Do not change `ParkingAgent` RL/RS switching behavior.
- Do not treat a single red-line annotation as the only ground truth.
- Do not use style score as a replacement for existing PSR/ANGS/PL metrics.
- Keep the first implementation default-off and read-only with respect to existing traces.

## Phases

### Phase 1: Research And PRD

- [x] Save user screenshots and related research in `raw_paper/`.
- [x] Survey related papers and open-source tools.
- [x] Decide that no complete drop-in evaluator/generator exists.
- [x] Define P0 scope as perpendicular + parallel offline style evaluation.
- [x] Write standalone production requirement.
- **Status:** complete, awaiting user review.

### Phase 2: Implementation Plan

- [x] After PRD approval, convert the PRD into an implementation plan.
- [x] Keep the implementation plan focused on offline tools and tests.
- [x] Decide exact file/module names under `tools/stage4/`.
- [x] Define first smoke trace inputs and synthetic golden cases.
- **Status:** complete, awaiting user choice of Subagent-Driven or Inline Execution.

### Phase 3: P0 Implementation

- [x] Implement trace loader.
- [x] Implement scene classifier.
- [x] Implement perpendicular and parallel reference family generators.
- [x] Implement style evaluator metrics.
- [x] Implement JSON/Markdown report writer.
- [x] Add unit, golden, and trace-smoke tests.
- **Status:** complete.

### Phase 4: P0 Review And Next Experiment Decision

- [x] Run the evaluator on representative Stage 4 120K traces.
- [x] Review top style-mismatch cases.
- [ ] Decide whether the next behavior-changing experiment should be RS reranking, reward shaping, auxiliary trajectory head, or another route.
- **Status:** evidence collected; next behavior-changing experiment still requires a separate user decision.

## Acceptance Summary For P0

P0 is accepted when:

- the CLI runs from Windows PowerShell using the project Python environment;
- existing Stage 4 selected trace JSONs can be parsed;
- at least 10 perpendicular and 10 parallel supported cases can produce reports when data is available;
- every supported case has a reference family and style score;
- unsupported cases include explicit reasons;
- identical-path, shifted-path, and extra-cusp golden tests pass;
- reports list top style mismatches and aggregate summaries by route family;
- no checkpoint, training-log, reward, policy, or planner behavior is changed.

## Errors Encountered

| Error | Attempt | Resolution |
|-------|---------|------------|
| Report writer could write case summaries outside `case_summaries/` when `case_uid` contained path separators. | Task 6 code-quality review and red/green CLI test. | Sanitize case summary filenames, resolve-check output paths, and reject sanitized filename collisions. |
| `--case-filter` could silently produce a successful empty report. | Task 6 CLI hardening. | Fail closed with exit code 2 when a supplied filter matches zero traces. |
| `--reference-annotations` and `--visualization-dir` were accepted but unused. | Task 6 CLI hardening. | Return an explicit P0 "not implemented" error instead of silently ignoring the flags. |

## Notes

- The previous long Stage 3/Stage 4 training planning files were archived before compaction in `docs/planning_archive/2026-06-30-pre-trajectory-style-prd/`.
- The root planning files now intentionally describe only the current trajectory-style evaluation work.
- Final focused verification passed on 2026-06-30: 48 trajectory-style tests, py_compile, and a real trace smoke on `docs/research/stage4_ogm_120k_selected_traces_get_action.json`.
- Smoke output is under `raw_paper/trajectory_style_eval_runs/2026-06-30_p0_smoke/`: 4 cases, 4 supported, 0 unsupported, 4 case summaries.
