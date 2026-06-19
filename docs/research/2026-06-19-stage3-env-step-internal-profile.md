# Stage 3 Env Step Internal Profile

## Scope

This note records the first detailed `env.step` internal timing pass after the Stage 3 safe-speed framework was synced to GitHub.

Boundary:

- No original HOPE training, environment, or model source files were modified.
- The profiler was enhanced under `tools/stage3/` only.
- The traces are bounded diagnostics and are not training-quality evidence.

## Method

The profiler now supports:

```powershell
.\.venv\Scripts\python.exe tools\stage3\profile_stage3_components.py `
  --profile-mode command-only `
  --detail env-step `
  --episodes 16 `
  --max-steps 160 `
  --updates 20 `
  --json docs\research\stage3_env_step_detail_command_only_20260619.json `
  --markdown docs\research\stage3_env_step_detail_command_only_20260619.md
```

The same command was repeated with `--profile-mode original`.

The `env-step` detail mode instruments nested calls, so component percentages are not exclusive. Use per-call timing and normalized milliseconds per raw step for decisions.

## Key Results

| Mode | Raw step avg | Render total | Image draw/capture/process | Action mask | Lidar | RS find path | Vehicle step |
|------|-------------:|-------------:|---------------------------:|------------:|------:|-------------:|-------------:|
| command-only | `9.635 ms` | `6.621 ms` | `3.890 ms` | `1.779 ms` | `0.651 ms` | `1.323 ms` | `0.876 ms` |
| original | `9.287 ms` | `6.692 ms` | `3.912 ms` | `1.780 ms` | `0.642 ms` | `0.694 ms` | `0.983 ms` |

The two modes collected different trajectories and transition counts, so total wall-clock duration is not directly comparable. The per-step profile is stable enough to identify the major internal costs:

1. Render/observation generation dominates `env.step` at about `6.6-6.7 ms/raw_step`.
2. The RGB image pipeline is the largest render sub-block at about `3.9 ms/raw_step`.
3. `ActionMask.get_steps` is the largest isolated deterministic sub-block at about `1.78 ms/raw_step`.
4. Lidar is smaller at about `0.64-0.65 ms/raw_step`.
5. `pygame.display.update()` / `clock.tick()` residual is only about `0.28-0.33 ms/raw_step`, so skipping display-update/frame-cap code is not a high-leverage first source optimization.
6. RS path probing can matter when invoked, but invocation frequency depends heavily on trajectory maturity and near-goal states.

## Action Mask Micro-Probe

A no-source-change micro-probe tested an equivalent vectorized formula for `ActionMask.get_steps`.

Current implementation allocates and fills `step_save` every call:

```python
step_save = np.zeros_like(self.dist_star)
step_save[self.dist_star <= dist_obs] = 1
step_save[self.dist_star > dist_obs] = 0
max_step = np.argmin(step_save, axis=-1)
max_step[np.sum(step_save, axis=-1) == self.n_iter] = self.n_iter
```

Equivalent candidate:

```python
step_valid = self.dist_star <= dist_obs
max_step = np.argmin(step_valid, axis=-1)
max_step[np.all(step_valid, axis=-1)] = self.n_iter
```

Micro-probe result:

- `200` random lidar samples checked with exact `np.allclose(..., atol=0, rtol=0)`.
- Max absolute difference: `0.0`.
- `2000` calls: original `1.481 ms/call`, candidate `0.768 ms/call`.
- Isolated speedup: about `1.93x`.

Runtime monkeypatch estimate inside the bounded env diagnostic:

- `env.render.action_mask`: about `1.78 ms` to `0.83 ms`.
- `env.raw_step.total`: about `9.64 ms` to `8.48 ms`.
- Estimated raw-step improvement: roughly `12%` in that bounded run.

This is not a candidate acceptance result. It is evidence for selecting the first optimization target.

## Decision: First Safe Optimization Point

The first real safe optimization candidate should be an opt-in fast path for `ActionMask.get_steps` that removes the per-step `step_save` allocation and assignment while preserving exact output semantics.

Why this should be first:

- It is a measured `env.step` hot path.
- It is deterministic and local: input is the lidar vector, output is the same action-mask vector.
- The candidate formula is simple and output-equivalent in a micro-probe.
- It preserves the HOPE paper's action-mask design intent; it only changes implementation efficiency.
- It is safer than starting with the larger RGB image pipeline, where preserving exact pixel semantics is harder.
- It has more expected payoff than bypassing display update or frame cap, which the trace shows is a small residual cost.

## Required Validation Before Any 20K Run

Before this optimization is used in training:

1. Add unit tests comparing original and fast action-mask outputs across random, edge-case, and env-collected lidar samples.
2. Add a parity harness that runs scripted environment transitions with original and fast action masks and compares `img`, `lidar`, `target`, `action_mask`, reward, done/status, and RS path availability.
3. Keep the optimization opt-in until it passes parity.
4. Run a bounded speed smoke to verify action-mask and raw-step timing improvement.
5. Run a 20K gated candidate only after the smoke passes, comparing against the locked command-only 20K baseline.

## Not Selected First

- RGB image pipeline: bigger than action mask, but higher semantic risk because exact observation pixels must remain compatible with the original policy input.
- Pygame display update / frame cap bypass: lower risk, but only about `0.3 ms/raw_step`; not enough leverage for the first real optimization.
- RS path probing: important but state-dependent and tied to hybrid planner behavior; optimize only after more targeted RS-heavy profiling.
- SAC update/replay tensor conversion: worth later investigation, but outside `env.step` and not the bottleneck requested in this step.
