# Stage 3 20K Comparison Report

- Decision: `pass`
- Baseline id: `stage3_command_only_20k_20260619`
- Candidate type: `fast_action_mask_20k_validation`
- Run dir: `D:\Github\HOPE\src\log\exp\sac_20260620_085208`
- Parity status: `pass`
- Checkpoint missing: `False`

## Run Summary

The fast action-mask candidate reached the 20K gate and produced `SAC_19999.pt`. The candidate passes both gates against the saved command-only 20K baseline: wall-clock time to the 20K checkpoint improved by `15.55%`, episode throughput improved by `18.19%`, environment-step throughput improved by `18.22%`, and matched external 200-episode scene evaluation was identical to the command-only 20K baseline.

Speed metrics use the actual child training process start time and the `SAC_19999.pt` checkpoint mtime. The later natural process exit included post-training evaluation overhead and is recorded separately in the manifest, but it is not the speed gate denominator.

## 36.5K Baseline Caveat

This is a 20K smoke-quality gate, not a 36.5K equivalence proof. Against the stopped 36.5K reference (`SAC_35999.pt`) of Normal `1.000`, Complex `0.985`, Extrem `0.915`, DLP `0.955`, mean `0.96375`, the 20K candidate is lower on maturity-sensitive scenes: Normal `-0.015`, Complex `-0.040`, Extrem `-0.260`, DLP `+0.005`, mean `-0.0775`. Because the candidate exactly matches the saved 20K baseline external eval, this is interpreted as normal 20K maturity gap rather than a fast action-mask quality regression.

## Stop Handling

`tools/stage3/stop_stage3_at_20k.ps1` was attempted after the checkpoint existed, but its process-window safety check rejected the reassigned child PID because PowerShell `ConvertFrom-Json` converted manifest timestamps through local/UTC semantics. A manual fallback used equivalent PID/name/command-line validation; by that point the workload and resource monitor had already exited naturally after preserving logs and checkpoints. The manifest records the fallback reason and process stop results.

## Changed Knobs

```json
{
  "eval_episode": 200,
  "command_only_flags": {
    "visualize": "",
    "verbose": ""
  },
  "train_episode": 20000,
  "fast_action_mask": true
}
```

## Speed

- Speed pass: `True`
- Candidate metrics: `{"wall_time_hours": 10.947863, "episodes_per_hour": 1826.840564, "env_steps_per_second": 46.223481}`
- Wall-time improvement ratio: `0.1555181271212589`
- Episodes/hour improvement ratio: `0.1818779486449593`
- Env steps/second improvement ratio: `0.18218621483375955`

## Quality

- Quality pass: `True`
- Mean success: `0.88625`
- Mean pass: `True`
- Hard reject reasons: `[]`

## Scene Eval

- Normal: `0.985`
- Complex: `0.945`
- Extrem: `0.655`
- DLP: `0.96`
- Mean: `0.88625`

## Resource Summary

- Sample count: `8060`
- Avg process CPU percent: `39.66741811414392`
- Avg whole GPU util percent: `32.22158808933003`
- Avg GPU memory used MB: `3130.3939205955335`
- Max GPU memory used MB: `3643.0`

## TensorBoard

- Has nonfinite: `True`
- Hard-reject has nonfinite: `False`
- Nonfinite scalar tags: `["success_rate_Complex", "success_rate_Extrem", "success_rate_dlp"]`
- Hard-reject nonfinite scalar tags: `[]`
- Multi-scene collapse: `False`
- Low scene success tags: `[]`
- Core scene collapse reasons: `[]`

## Command

```powershell
D:\Github\HOPE\.venv\Scripts\python.exe D:\Github\HOPE\tools\stage3\train_HOPE_sac_fast_action_mask.py --train_episode 20000 --eval_episode 200 --visualize= --verbose=
```
