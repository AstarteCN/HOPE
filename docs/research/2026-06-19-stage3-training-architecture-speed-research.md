# Stage 3 Training Architecture Speed Research

## Status

Research-only active after the command-only 20K smoke stop and quality check.

This note defines the research boundary for the next speed-improvement phase. It is research-only until the user approves an implementation plan.

Deep follow-up note:

- `docs/research/2026-06-19-stage3-safe-speed-deep-research.md`

The follow-up note combines local paper evidence, code evidence, baseline/command-only training summaries, safe acceleration boundaries, and the recommended next research-to-implementation ladder.

## 20K Smoke Gate Outcome

The command-only 20K smoke stopped at `20038` TensorBoard episodes after `SAC_19999.pt` was written.

Matched 200-episode eval showed exact parity between the command-only checkpoint and the original baseline 20K checkpoint:

| Scene | Command-only 20K | Baseline 20K | Delta |
|-------|------------------|--------------|-------|
| Normal | `0.985` | `0.985` | `0.000` |
| Complex | `0.945` | `0.945` | `0.000` |
| Extrem | `0.655` | `0.655` | `0.000` |
| DLP | `0.960` | `0.960` | `0.000` |
| Mean | `0.88625` | `0.88625` | `0.00000` |

Conclusion: `--visualize= --verbose=` did not show a negative quality impact at the 20K smoke point. This is enough to start architecture-speed research, but it is not a 36.5K or 100K equivalence proof.

## Trigger

Start this research after the command-only smoke run is stopped around 20K episodes and reviewed:

- TensorBoard compared against the 20K baseline reference.
- Command-only `SAC_19999.pt` evaluated with the same budget as the baseline comparison.
- Command-only 20K indicators compared against the stopped 36.5K baseline with a clear caveat that 20K is not full equivalence.
- The command-only flags show no clear negative impact on training quality.

## Non-Negotiable Boundaries

Preserve the original HOPE paper's design intent:

- Keep action-mask behavior intact.
- Keep curriculum and scene scheduling intact unless a later user-approved research plan explicitly studies a controlled curriculum variant.
- Keep observation, action, reward, environment dynamics, and success/failure semantics intact.
- Keep the hybrid reinforcement-learning policy plus path-planning framing intact.
- Do not introduce OGM-specific design or source code during Stage 3.
- Do not modify original `src/train`, `src/env`, or `src/model` files without a separate user-approved implementation plan.

The goal is faster and high-quality training, not higher hardware utilization for its own sake.

## Research Direction

Focus on training methodology and engineering architecture:

- More precise profiling around environment stepping, render/image generation, lidar/action-mask calculation, replay sampling, and SAC update cadence.
- Opt-in experiment entry points or wrappers that leave original scripts and defaults available.
- Evaluation and TensorBoard/logging overhead isolation.
- Data movement and replay-buffer efficiency.
- Update scheduling experiments that preserve SAC semantics or are explicitly marked as algorithmic experiments.
- Multiprocess or batched data-collection feasibility, only if it can preserve curriculum, action mask, and replay semantics.
- Checkpoint/evaluation automation so long runs can be stopped, compared, and resumed intentionally.

## Safe Research Plan

Start with measurement and opt-in wrappers before source changes:

1. Build a no-source-change profiler harness for the original training loop that records time in action selection, environment stepping, render/image generation, lidar/action-mask work, Reeds-Shepp probing, replay sampling, SAC update, TensorBoard logging, and checkpoint/eval overhead.
2. Repeat the profiler with command-only flags to isolate which overhead was removed and which bottleneck remains.
3. Create an opt-in experiment launcher outside `src/train`, `src/env`, and `src/model` that can run the original training entry point with controlled environment variables, process affinity/thread settings, monitoring, and checkpoint gates.
4. Study evaluation/logging frequency and external monitoring overhead separately from training semantics. Any change here must be reversible and must not alter collected observations, actions, rewards, masks, curriculum, or replay samples.
5. Only after measurement, design candidate speed experiments such as replay/data movement improvements, controlled update scheduling tests, or environment-side cache probes. Each candidate needs parity tests and an explicit quality gate before long training.

Do not start with larger model, larger batch, reward changes, action-mask changes, curriculum changes, or OGM observation changes. Those are algorithmic or distribution changes, not safe speed architecture work.

## Quality Gates

Any future speed optimization must be compared against the local HOPE baseline using:

- TensorBoard reward/loss/alpha/action-std stability.
- Scene success rates for Normal, Complex, Extrem, and DLP.
- `step_num` trend and timeout recurrence.
- Matched checkpoint evaluation with the same `eval_episode` budget.
- Throughput in both episodes/hour and environment steps/second.

Do not claim training-quality equivalence from a short run. Short runs may validate plumbing, speed direction, or obvious failure only.

## First Research Questions

1. Which part of the original training loop dominates wall time after command-only logging/display overhead is removed?
2. Can resource use improve through process structure or data pipeline changes without changing the HOPE algorithm?
3. Can evaluation/logging be isolated so training throughput improves without changing collected experience?
4. Are environment-step costs mostly render/image/lidar/action-mask/RS-probe bound, and which of those can be cached or batched safely?
5. What minimum validation budget is needed for each proposed speed change before it is allowed into a longer run?
