# Stage 3 Diagnostic Dry Run

Date: 2026-06-17

## Purpose

This run was a diagnostic check of the Stage 3 command, TensorBoard, and resource-monitoring pipeline. It is not a Stage 3 baseline and must not be used to judge training quality.

The user clarified that the 40K Stage 3 requirement refers to approximately 40,000 training episodes, about 40% of the original author's 100,000-episode training scale. The diagnostic run below used only 300 training episodes.

## Command

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
..\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 300 --eval_episode 200 --visualize False --verbose True
```

## Outputs

- Stamp: `20260617_230339`
- TensorBoard run directory: `D:\Github\HOPE\src\log\exp\sac_20260617_230340`
- Stdout log: `D:\Github\HOPE\src\log\exp\stage3_baseline_20260617_230339.stdout.log`
- Stderr log: `D:\Github\HOPE\src\log\exp\stage3_baseline_20260617_230339.stderr.log`
- Initial resource CSV: `D:\Github\HOPE\src\log\exp\stage3_resource_20260617_230339.csv`
- Child-process diagnostic resource CSV: `D:\Github\HOPE\src\log\exp\stage3_resource_20260617_230339.child29396.csv`
- Fixed child-process diagnostic resource CSV: `D:\Github\HOPE\src\log\exp\stage3_resource_20260617_230339.child29396.fixed.csv`

## Observations

- Training reached 300 episodes.
- Summed TensorBoard `step_num` values were about 35,689 environment steps.
- A partial DLP result file showed success rate `0.44`, with step num `60.54545454545455 +-(47.79784825199416)`.
- Final evaluation was interrupted after the user clarified the required scale.
- The run produced Gym/NumPy warnings, a trusted-checkpoint `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD` warning, a Shapely pickle compatibility warning, and a PyTorch slow tensor-construction warning.

## Monitoring Correction

The first monitor followed the launcher PID, which showed little useful CPU signal. The actual workload ran in a child Python process. Future long runs should detect and monitor the child process when present.

`tools/stage3/monitor_stage3_resources.ps1` was also corrected so repeated `nvidia-smi` sampling does not lose GPU fields after the first row.

## Interpretation

- This run proves the command and monitoring pipeline can start.
- It does not satisfy the 40,000-episode Stage 3 budget.
- It does not provide reliable reward, success-rate, or failure-mode evidence.
- The next meaningful Stage 3 baseline must use `--train_episode 40000`.

## Source Modification Status

`git diff -- src` produced no source diff after the diagnostic run.
