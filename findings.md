# Findings: HOPE Trajectory Style Evaluation Toolkit

## Current Task

The active task is to productize a P0 offline evaluator for human-like parking trajectory shape in Stage 4 HOPE+OGM.

Standalone PRD:

- `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/production_requirement.md`

Implementation plan:

- `docs/superpowers/plans/2026-06-30-trajectory-style-evaluation-toolkit.md`

Historical planning archive:

- `docs/planning_archive/2026-06-30-pre-trajectory-style-prd/`

## Research Conclusions

- Current HOPE+OGM checkpoints can succeed in selected DLP parking cases while producing trajectories that do not match common human parking style.
- The visible problem is not only success/failure. It is route shape: slot-mouth entry, clearance bias, planned cusp placement, curvature shape, and RL/RS handoff behavior.
- No complete open-source tool was found that directly provides both:
  - a HOPE-trace-aware offline trajectory style evaluator;
  - a scene-conditioned human reference trajectory family generator.
- ParkingE2E is useful for trajectory metrics such as L2, Hausdorff, and Fourier descriptor differences, but it is not a drop-in HOPE style evaluator.
- Human-like parking papers support route-family and behavior-feature approaches instead of optimizing only shortest feasible paths.
- Generic trajectory-distance libraries and parking planners are useful components, but the HOPE-specific trace schema, scene classification, and diagnosis layer need to be built locally.

## P0 Scope Decisions

P0 includes:

- Perpendicular parking:
  - `perpendicular_enough_space`;
  - `perpendicular_limited_space`.
- Parallel parking:
  - `parallel_standard`;
  - `parallel_tight`.
- Trace loading, scene classification, reference family generation, style evaluation, and reporting.

P0 excludes:

- angled parking;
- RS reranking;
- reward shaping;
- auxiliary trajectory heads;
- MPC/OBCA post-optimization;
- imitation-learning training;
- any behavior-changing modification to training, reward, checkpoint, or planner execution.

## Required Modules

- `TraceLoader`: normalize existing Stage 4 trace JSON into a stable trajectory structure.
- `SceneClassifier`: classify perpendicular/parallel scene subtype with explicit reasons.
- `ReferenceFamilyGenerator`: generate route families, corridors, phase labels, expected cusp/gear ranges, slot-mouth windows, and clearance preferences.
- `StyleEvaluator`: compute shape, behavior, clearance, and segment-level metrics.
- `ReportWriter`: write JSON and Markdown reports with batch summaries and unsupported-case reasons.

## Implementation Findings

- P0 implementation landed under `tools/stage4/` as a default-off offline toolkit.
- The schema layer now enforces finite numeric values, strict JSON-safe serialization, known style labels, known scene classes, string diagnosis/action-source sequences, and bool planner-route flags.
- The loader accepts single-trace and collection trace JSON, validates duplicate/non-empty `case_uid`, handles older action payload shapes, computes target-polygon centroids correctly, and rejects non-finite summary/action-selection metrics at load time.
- Shape metrics compare resampled trajectory geometry, so identical paths with different waypoint density do not produce false Hausdorff mismatch.
- Manual scene-label overrides are validated immediately; invalid scene classes fail early instead of propagating into later report generation.
- Unsupported evaluator reports preserve the classifier reason and supported empty-reference reports preserve their original scene class.
- The report writer keeps all cases in `cases`, keeps unsupported cases in `unsupported_cases`, ranks top mismatches from supported cases only, writes per-case summaries, and protects `case_summaries/` from unsafe `case_uid` path components.
- CLI hardening: zero-match `--case-filter` fails closed; `--reference-annotations` and `--visualization-dir` return explicit P0-not-implemented errors instead of acting as silent no-ops.

## Implementation Plan Decisions

- The implementation plan splits P0 into seven TDD slices: schema, metrics, loader, reference generation, evaluator, report/CLI, and integrated smoke verification.
- New code should live under `tools/stage4/`; tests should live under `tools/stage4/tests/`.
- The CLI entry point should be `tools/stage4/evaluate_trajectory_style.py`.
- The first real-trace smoke target is `docs/research/stage4_ogm_120k_selected_traces_get_action.json`, with smoke output under ignored `raw_paper/trajectory_style_eval_runs/`.

## Smoke Result

The 2026-06-30 P0 smoke run used:

```powershell
.\.venv\Scripts\python.exe tools/stage4/evaluate_trajectory_style.py --trace-json docs/research/stage4_ogm_120k_selected_traces_get_action.json --output-dir raw_paper/trajectory_style_eval_runs/2026-06-30_p0_smoke --slot-types perpendicular,parallel --write-markdown --write-json --allow-unsupported
```

Result:

- 4 cases loaded from the selected Stage 4 trace file.
- 4 supported cases; 0 unsupported cases.
- Case summaries written for `parallel_008`, `parallel_013`, `parallel_014`, and `parallel_016`.
- All four selected cases are currently labeled `style_mismatch`; common diagnoses are `route_family_mismatch` and `unplanned_extra_cusp`, with `parallel_014` also showing `excessive_chatter`.
- This is evaluation evidence only. It does not approve a behavior-changing experiment.

## Metrics To Preserve In Implementation

Shape:

- L2 distance;
- Hausdorff distance;
- Fourier descriptor difference;
- curvature mean/max/variation;
- slot-mouth entry pose error.

Behavior:

- gear-shift count versus expected range;
- cusp count and location;
- steering sign changes;
- low-speed direction chatter;
- planned-cusp compliance.

Clearance:

- minimum obstacle clearance;
- obstacle-side clearance bias;
- corridor violation when available.

Segment:

- full trajectory score;
- RL-segment mismatch;
- RS-segment mismatch;
- mismatch phase labels.

## Diagnosis Labels

Implementation should keep diagnosis explainable. Useful labels include:

- `route_family_mismatch`
- `slot_mouth_entry_too_shallow`
- `slot_mouth_entry_too_late`
- `excessive_chatter`
- `unplanned_extra_cusp`
- `object_side_clearance_too_close`
- `curvature_shape_mismatch`
- `rl_segment_mismatch`
- `rs_segment_mismatch`

## Key Design Boundary

Human style should be represented as a route family with a corridor and phase labels, not as a single mandatory red-line polyline.

Planned cusps in limited-space perpendicular or tight parallel parking are acceptable human-like maneuvers. Random low-speed direction chatter and extra unplanned cusps are not.

## Useful Local Inputs

- `docs/research/stage4_ogm_120k_selected_traces_get_action.json`
- `docs/research/stage4_ogm_120k_selected_traces_choose_action.json`
- `docs/research/stage4_ogm_fixed_eval_case_geometry_20260627.json`
- User annotation screenshots under `raw_paper/2026-06-30-human-like-parking-trajectory-shape-research/attachments/user_annotations/`

## Prior Stage 4 Boundary

Current HOPE Stage 4 already uses a hybrid `RL + Reeds-Shepp + action mask` contract. This task should measure and explain style mismatch first. Behavior changes should be a later, separately approved experiment.
