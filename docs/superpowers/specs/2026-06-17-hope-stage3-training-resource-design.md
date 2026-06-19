# HOPE Stage 3 Training Resource Study Design

Date: 2026-06-17

## Goal

Stage 3 will retrain the original HOPE agent locally while studying whether the current training code uses this machine's hardware effectively. It must preserve the original HOPE training framework and postpone OGM adaptation.

## Scope

In scope:

- Original HOPE SAC retraining.
- Resource utilization measurement.
- Training bottleneck analysis.
- Quality-preserving throughput improvements.
- TensorBoard monitoring and early warning summaries.

Out of scope:

- OGM implementation.
- Reward redesign.
- Environment dynamics changes.
- Model architecture changes.
- Replacing the original training framework with an incompatible trainer.

## Recommended Approach

Use a staged, evidence-first approach:

1. Baseline run: execute the original training path and collect resource and TensorBoard data.
2. Bottleneck analysis: identify whether CPU, GPU, environment simulation, rendering, collision checks, action masking, replay sampling, PyTorch update work, logging, or evaluation dominates runtime.
3. Quality-preserving optimization: propose one change at a time, keep it opt-in or easily reversible, and compare against baseline.

This approach is preferred over direct optimization because reinforcement learning can look healthy in a short run while hiding instability, poor exploration, or scene-specific failure.

## Alternative Approaches Considered

### Direct Performance Tuning First

This would immediately change code to increase GPU use or parallelize environment work. It is risky because it can silently alter training behavior, reward timing, replay distribution, or scene sampling before a baseline exists.

### Full Paper-Scale Training First

This would run a long unmodified reproduction attempt before profiling. It is faithful to the original method but may waste time if local hardware is mostly idle due to a known bottleneck.

### Recommended Hybrid

Run a meaningful baseline first, but make it long enough to produce early RL signals and resource data. Use at least 40,000 training episodes for smoke and optimization validation.

## Training Budget Rule

For Stage 3, the 40K budget means 40,000 training episodes, approximately 40% of the original HOPE paper/repository training scale of 100,000 episodes.

Smoke and modification-validation runs should cover at least 40,000 training episodes. Shorter commands are allowed only for syntax, monitoring, import, or crash diagnostics. They must not be used to judge training quality.

TensorBoard `step_num` should still be summed and recorded as the measured environment interaction step count. That count is useful for throughput, SAC-update estimates, and failure-mode analysis, but it is not the Stage 3 budget gate.

## Resource Measurements

Each meaningful run should record:

- GPU utilization.
- GPU memory usage.
- CPU utilization.
- RAM usage.
- Disk activity if logging or checkpointing appears expensive.
- Wall-clock runtime.
- Approximate episodes per second and environment steps per second.
- TensorBoard log directory.

Unless a PID-level GPU monitor is explicitly added, GPU utilization should be interpreted as whole-device `nvidia-smi` state. It is useful for coarse trends but not proof that a specific Python process owns all sampled GPU activity.

Suggested Windows tools include:

- `nvidia-smi` for GPU utilization and memory.
- `Get-Process` for Python CPU and memory.
- TensorBoard for training curves.
- A small future monitoring script if manual observation is not enough.

## Training Quality Signals

Monitor these TensorBoard scalars:

- `total_reward`
- `avg_reward`
- `actor_loss`
- `critic_loss`
- `action_std0`
- `action_std1`
- `alpha`
- `success_rate_Normal`
- `success_rate_Complex`
- `success_rate_Extrem`
- `success_rate_dlp`
- `step_num`

## Early Warning Failure Modes

Summaries should watch for:

- Reward plateau or degradation after enough training episodes.
- Success rate improving only in easy scenes.
- `step_num` staying near timeout.
- Actor or critic loss becoming NaN, exploding, or oscillating without reward improvement.
- `alpha` collapsing or exploding.
- `action_std0` or `action_std1` collapsing too early, indicating weak exploration.
- DLP success diverging from synthetic-scene success.
- GPU utilization staying low while CPU is high.
- GPU memory available but PyTorch update work not keeping the GPU busy.

## Change Boundaries

Any optimization must:

- Preserve original reward shaping.
- Preserve environment dynamics.
- Preserve action mask semantics.
- Preserve RS fallback behavior.
- Preserve scene selection semantics unless explicitly studied.
- Be measured against the baseline.
- Be validated with at least 40,000 training episodes.
- Be documented before Stage 4 OGM work begins.

## Documentation Outputs

Future Stage 3 work should produce:

- A baseline resource profile note under `docs/research/`.
- A TensorBoard signal summary.
- A bottleneck analysis note.
- If changes are made, an optimization comparison note showing baseline versus modified throughput and training signals.
