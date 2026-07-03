# Stage 4 OGM Proxy 1K Diagnostic

Date: 2026-06-20

## Run

- Manifest: `D:\Github\HOPE\src\log\exp\stage4_ogm_proxy_1k_20260620_234404.meta.json`.
- Run directory: `D:\Github\HOPE\src\log\exp\sac_ogm_stage4_ogm_proxy_1k_20260620_234404`.
- Periodic checkpoint: none expected at 1K; `SAC_999.pt` is not expected because the training loop saves every 2000 episodes.
- Best checkpoint: `D:\Github\HOPE\src\log\exp\sac_ogm_stage4_ogm_proxy_1k_20260620_234404\SAC_best.pt`.
- Resource CSV: `D:\Github\HOPE\src\log\exp\stage4_ogm_proxy_1k_20260620_234404.resources.csv`.

## TensorBoard

- Training budget met: `true`.
- Episode count: `1000`.
- Environment step count: `131691`.
- Estimated SAC updates after warmup: `12145`.
- Hard-reject nonfinite: `false`.
- Missing scalars: none.
- Avg reward latest: `0.026393`; `mean_first500=0.014867`, `mean_last100=0.024356`, trend delta `+0.006286`.
- Total reward latest: `5.517873`; `mean_first500=1.953888`, `mean_last100=2.660554`, trend delta `+1.306628`.
- Step-num latest: `10`; `mean_first500=131.41`, `mean_last100=123.47`.
- Actor loss latest: `-0.496791`; `mean_last100=-0.388306`; nonfinite `false`.
- Critic loss latest: `0.047034`; `mean_last100=0.057117`; nonfinite `false`.
- Alpha latest: `0.009414`; `mean_last100=0.009444`; nonfinite `false`.

## Resource Summary

- Total resource CSV samples: `794`.
- Resource wall window: `1.112331 h`.
- Approximate throughput over resource window: `899.013 episodes/hour`, `32.887 env steps/second`.
- Trusted actual-PID sample count after retarget: `309`.
- Trusted avg process CPU: `5.946%`.
- Trusted avg whole-GPU utilization: `35.013%`.
- Trusted avg GPU memory used: `3033.353 MB`.
- Trusted peak GPU memory used: `3377 MB`.
- Trusted avg working set: `2509.379 MB`.
- Trusted avg private memory: `4441.116 MB`.

## Operational Notes

- The launcher initially monitored the `.venv\Scripts\python.exe` shim PID (`2116`) instead of the actual training child PID (`16288`). The resource monitor was manually retargeted during this run, so early process CPU/RAM rows in the CSV should be treated as startup monitor bias. The launcher was fixed in `404a679` so future Stage 4 runs record and monitor the actual child Python PID.
- A direct runner smoke from `D:\Github\HOPE` first failed on `../data/dlp.data`; this exposed the original HOPE current-working-directory dependency. The Stage 4 runner now normalizes CWD to `D:\Github\HOPE\src` before environment construction, and the direct 20-episode smoke passed afterward.
- Runtime warnings were limited to existing Gym/Gymnasium deprecation notice, Gym Box precision warning, short-run NumPy empty-slice warnings, Shapely pickle compatibility warning, and a PyTorch replay-batch tensor-construction performance warning.

## Decision

Proceed to the 20K gate. The 1K diagnostic completed without process failure, TensorBoard budget was met, all required scalars were present and finite, and OGM observations remained usable through training and evaluation.
