# Stage 3 Command-Only 1000-Episode Smoke

## Scope

This was a user-approved parameter-only speed/resource smoke. It is not a 35K or 40K training-quality validation.

The original HOPE training, environment, and model source files were not modified. The run only changed command-line parameters so that the existing `argparse type=bool` options receive empty strings and evaluate to `False`.

## Command

Run started on 2026-06-18 and completed on 2026-06-19 local time.

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
D:\Github\HOPE\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 1000 --eval_episode 1 --visualize= --verbose=
```

`--eval_episode 1` was used only to avoid spending time on final evaluation if the process completed naturally. The speed result below uses the last TensorBoard training scalar, not the final eval output.

## Artifacts

- Run dir: `D:\Github\HOPE\src\log\exp\sac_20260618_235234`
- Metadata: `D:\Github\HOPE\src\log\exp\stage3_command_only_1000_20260618_235232.meta.json`
- TensorBoard summary: `D:\Github\HOPE\src\log\exp\stage3_command_only_1000_20260618_235232.tb_summary.json`
- Metrics: `D:\Github\HOPE\src\log\exp\stage3_command_only_1000_20260618_235232.metrics.json`
- Workload resource CSV: `D:\Github\HOPE\src\log\exp\stage3_command_only_1000_20260618_235232.resources.workload.csv`

The first resource monitor briefly sampled the Windows venv launcher PID. The analysis uses `resources.workload.csv`, which tracks the real child Python training process.

## Result

| Metric | Value |
|--------|-------|
| Episodes | 1,000 |
| Environment steps | 127,470 |
| Average `step_num` | 127.47 |
| Start to last training scalar | 2,261.16 s / 37.69 min |
| Episodes/hour | 1,592.10 |
| Environment steps/second | 56.37 |
| Resource samples | 442 |
| Average normalized CPU | 44.74% |
| P95 normalized CPU | 48.56% |
| Peak normalized CPU | 48.91% |
| Average whole-GPU utilization | 36.75% |
| P95 whole-GPU utilization | 51.00% |
| Peak whole-GPU utilization | 78.00% |
| Peak whole-GPU memory | 3,519 MB |

TensorBoard scalars were finite. `total_reward` trend improved by about `+2.04`; `critic_loss` trend decreased by about `-0.197`; `alpha` moved from `0.0100` to `0.00943`.

The final short-run warning was `success_rate_Extrem remains below 0.2`. This is expected for a 1K early-training smoke and should not be treated as a failure by itself.

## Fair Baseline Comparison

The fair comparison is the original baseline's first 1,000 episodes, not the 36.5K full-run average, because early random-policy episodes are much longer.

| Metric | Original first 1K | Command-only 1K | Ratio |
|--------|-------------------|-----------------|-------|
| Episodes/hour | 459.58 | 1,592.10 | 3.46x |
| Environment steps/second | 14.99 | 56.37 | 3.76x |
| Average normalized CPU | 45.41% | 44.74% | similar |
| Average whole-GPU utilization | 31.13% | 36.75% | +5.62 pp |
| Peak GPU memory | 3,303 MB | 3,519 MB | +216 MB |

Compared against the full stopped 36.5K baseline, this smoke is about `1.04x` by episodes/hour and `1.76x` by environment steps/second. That comparison is less meaningful because the 36.5K run includes later learned-policy episodes with much shorter average `step_num`.

## Interpretation

The parameter-only change is a strong candidate for future runs because it preserves the original training script and removes non-learning overhead. The main benefit appears in early training throughput, where the original run was slowed heavily by effective verbose/visual behavior caused by `bool("False") == True`.

This does not prove long-run training-quality equivalence. To claim that, the command-only configuration still needs a matched long validation with TensorBoard and final evaluation comparison.

## Next Decision

Use the command-only flags for the next long validation candidate:

```powershell
--visualize= --verbose=
```

or equivalently pass empty string values from PowerShell. Continue to avoid source-level optimization until a measured hot path and parity test justify it.
