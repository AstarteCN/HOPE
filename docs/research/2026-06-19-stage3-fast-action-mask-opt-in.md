# Stage 3 Opt-In Fast Action Mask

Date: 2026-06-19

## Scope

This change implements the first approved safe optimization target from the Stage 3 env-step profile: an opt-in exact-output fast path for `ActionMask.get_steps`.

The default HOPE behavior remains unchanged:

- `ActionMask()` keeps `fast_get_steps=False`.
- `CarParking` and SAC/PPO agents still construct `ActionMask()` without fast mode.
- The optimized branch runs only when `fast_get_steps=True` is explicitly set by tests or bounded diagnostics.

## Safety Boundary

This change preserves the HOPE paper and code design intent:

- Action mask semantics are preserved.
- Curriculum and scene scheduling are not touched.
- Observation, action, reward, vehicle dynamics, and hybrid planner behavior are not changed.
- No files under `src/train` or `src/env` are modified.
- The only protected-source change is the default-off branch in `src/model/action_mask.py`.

## Implementation

The original implementation materialized a full `step_save` array, wrote `1` for valid steps, wrote `0` for invalid steps, then used `argmin` and `sum` to find the first invalid step.

The fast implementation computes the same boolean validity directly:

```python
step_valid = self.dist_star <= dist_obs
max_step = np.argmin(step_valid, axis=-1)
max_step[np.all(step_valid, axis=-1)] = self.n_iter
```

This avoids one allocation and two full-array assignment passes while keeping the same downstream `post_process` behavior.

## Parity Evidence

Test command:

```powershell
.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -p "test_*.py" -v
```

Result: PASS, 37 tests.

New parity tests:

- `tools/stage3/tests/test_fast_action_mask.py`
- Edge lidar samples: `-5.0`, `0.0`, `0.01`, `1.0`, `9.99`, `10.0`, `25.0`
- Seeded random lidar samples: 32 samples from range `[-2.0, 14.0]`
- Assertion: `ActionMask(fast_get_steps=True).get_steps(lidar)` exactly equals `ActionMask().get_steps(lidar)`.

## Bounded Profile Evidence

Both runs used:

```powershell
$env:SDL_VIDEODRIVER = "dummy"
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = "1"
.\.venv\Scripts\python.exe tools\stage3\profile_stage3_components.py --profile-mode command-only --detail env-step --episodes 16 --max-steps 160 --updates 20
```

| Action mask mode | Transitions | Total seconds | Action mask avg ms | Env step avg ms | Raw step avg ms | Render avg ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `original` | 1654 | 25.448 | 1.694 | 9.345 | 9.391 | 6.302 |
| `fast` | 1551 | 22.188 | 0.821 | 8.205 | 8.279 | 5.596 |

Artifacts:

- `docs/research/stage3_env_step_detail_action_mask_original_check_20260619.json`
- `docs/research/stage3_env_step_detail_action_mask_original_check_20260619.md`
- `docs/research/stage3_env_step_detail_fast_action_mask_20260619.json`
- `docs/research/stage3_env_step_detail_fast_action_mask_20260619.md`

## Interpretation

The fast branch reduces isolated action-mask cost by about 51.5% in this bounded diagnostic:

- Original: `1.694 ms/call`
- Fast: `0.821 ms/call`

The observed `env.step` average improves by about 12.2%:

- Original: `9.345 ms/call`
- Fast: `8.205 ms/call`

Because episode paths and transition counts differ slightly, this is a bounded measurement-only result, not a training-quality result. It is enough to admit the fast action-mask branch as a candidate for a future 20K gated run, where quality must be compared against the command-only 20K baseline.

## Next Gate

Before using this in a long run:

1. Launch an explicit candidate run with fast action mask enabled through an opt-in wrapper or entry point.
2. Train to 20K episodes.
3. Compare against the command-only 20K baseline `src/log/exp/sac_20260619_004316 / SAC_19999.pt`.
4. Require matched eval, TensorBoard health, resource summary, and speed gates before calling it a safe speedup.
