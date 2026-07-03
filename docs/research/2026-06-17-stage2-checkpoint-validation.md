# Stage 2 Checkpoint Validation

Date: 2026-06-17

Scope: run original HOPE evaluation code with author-provided policy checkpoints inside the isolated `.venv` environment.

This stage validates checkpoint loading and runtime execution. The episode counts are intentionally small and are not paper reproduction metrics.

## Environment

- Python executable: `D:\Github\HOPE\.venv\Scripts\python.exe`
- Python version: `3.13.12`
- PyTorch: `2.11.0+cu128`
- CUDA visible to PyTorch: yes
- GPU: `NVIDIA GeForce RTX 4080 SUPER`
- Working directory for evaluation commands: `D:\Github\HOPE\src`

## Important Compatibility Note

The first `HOPE_SAC0.pt` run failed during checkpoint loading because PyTorch 2.6+ defaults `torch.load()` to `weights_only=True`. The author checkpoints include trusted custom config objects such as `model.agent.sac_agent.SACConfig`.

To preserve the original source code path, checkpoint runs were executed with:

```powershell
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
```

PyTorch printed a warning confirming that this forced `weights_only=False`. This is acceptable here because the checkpoints are local files from the trusted repository source.

## Command Pattern

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\evaluation\eval_mix_scene.py <checkpoint> --eval_episode <n> --visualize False
```

`SDL_VIDEODRIVER=dummy` keeps Pygame headless while preserving the original evaluation path.

## Checkpoint Inventory

Policy checkpoints:

- `src/model/ckpt/HOPE_SAC0.pt`
- `src/model/ckpt/HOPE_SAC1.pt`
- `src/model/ckpt/HOPE_PPO.pt`

Non-policy checkpoint:

- `src/model/ckpt/autoencoder.pt`

`autoencoder.pt` was not evaluated as a policy checkpoint.

## HOPE_SAC0 Validation

Command:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 10 --visualize False
```

Result directory:

```text
src/log/eval/20260617_221106/
```

Summary:

| Segment | Success Rate | Step Num |
|---------|--------------|----------|
| extreme | 1.0 | 21.7 +-(5.496362433464518) |
| dlp | 1.0 | 20.1 +-(5.769748694700662) |
| complex | 1.0 | 36.6 +-(48.006666203768) |
| normalize | 1.0 | 13.5 +-(3.383784863137726) |

The command completed successfully and printed `load pre-trained model!`.

## HOPE_SAC1 Sanity Check

Command:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC1.pt --eval_episode 3 --visualize False
```

Result directory:

```text
src/log/eval/20260617_221207/
```

Summary:

| Segment | Success Rate | Step Num |
|---------|--------------|----------|
| extreme | 1.0 | 28.666666666666668 +-(13.523641850067197) |
| dlp | 1.0 | 20.666666666666668 +-(1.247219128924647) |
| complex | 1.0 | 10.666666666666666 +-(0.9428090415820634) |
| normalize | 1.0 | 24.0 +-(10.98483803552272) |

The command completed successfully and printed `load pre-trained model!`.

## HOPE_PPO Sanity Check

Command:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_PPO.pt --eval_episode 3 --visualize False
```

Result directory:

```text
src/log/eval/20260617_221220/
```

Summary:

| Segment | Success Rate | Step Num |
|---------|--------------|----------|
| extreme | 1.0 | 63.333333333333336 +-(59.26962872238098) |
| dlp | 1.0 | 18.666666666666668 +-(4.642796092394707) |
| complex | 1.0 | 44.666666666666664 +-(24.526629518862872) |
| normalize | 1.0 | 15.0 +-(1.4142135623730951) |

The command completed successfully and printed `load pre-trained model!`.

## Warnings Observed

Warnings that did not block execution:

- `gym 0.26.2` prints an unmaintained/Gymnasium migration warning under NumPy 2.x.
- `gym.spaces.Box` prints a float precision warning.
- `parking_map_dlp.py` prints a Shapely warning when unpickling shapely `<2.0` geometry from `data/dlp.data`.
- PyTorch prints a warning when `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` forces `weights_only=False`.

No source changes were made to address these warnings during Stage 2.

## Source Tree Status

After the checkpoint validations:

```text
git diff -- src
```

had no output, meaning no source files under `src/` were changed.

Evaluation logs were generated under `src/log/eval/`, which is ignored by `.gitignore`.

## Stage 2 Conclusion

Stage 2 is complete:

- `HOPE_SAC0.pt` loaded and completed the recommended smoke evaluation.
- `HOPE_SAC1.pt` loaded and completed a quick sanity evaluation.
- `HOPE_PPO.pt` loaded and completed a quick sanity evaluation.
- The original evaluation script works in the isolated project environment.
- PyTorch 2.11 requires the checkpoint compatibility environment variable unless source loading code is updated in a future maintenance pass.

## Next Stage

Stage 3 can start original HOPE retraining in the isolated environment. Start with a short SAC smoke training run before any long run:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' .\train\train_HOPE_sac.py --train_episode 100 --eval_episode 20 --visualize False --verbose True
```
