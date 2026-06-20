# Stage 4 OGM Proxy Smoke Report

Date: 2026-06-20

## Commands

- Unit tests: `.\.venv\Scripts\python.exe -m unittest discover -s tools\stage4\tests -v`
- Stage 3 regression tests: `.\.venv\Scripts\python.exe -m unittest discover -s tools\stage3\tests -v`
- Tiny OGM training smoke: `D:\Github\HOPE\.venv\Scripts\python.exe D:\Github\HOPE\tools\stage4\train_HOPE_sac_ogm.py --train_episode 20 --eval_episode 1 --visualize= --verbose=`

## Result

- Stage 4 unit tests: passed, `Ran 48 tests in 2.112s`, `OK`.
- Stage 3 regression tests: passed, `Ran 38 tests in 0.714s`, `OK`.
- OGM smoke run directory: `D:\Github\HOPE\src\log\exp\sac_ogm_20260620_233953`.
- TensorBoard event file: `events.out.tfevents.1781969993.pengzhou_travel.14752.0`, size `10288` bytes.
- Checkpoints: none expected and none found for this 20-episode smoke.
- Tiny smoke status: exit code `0` after the runner working-directory fix.

## Debug Note

The first direct smoke attempt from `D:\Github\HOPE` failed before training because original HOPE DLP map loading uses `../data/dlp.data` relative to the current working directory. The Stage 4 launcher already starts from `D:\Github\HOPE\src`, but the direct Task 10 command did not. This was fixed with a TDD change in `tools/stage4/train_HOPE_sac_ogm.py`: the runner now normalizes its process working directory to `SRC_ROOT` before environment construction. The targeted test first failed with `ImportError: cannot import name 'ensure_src_working_directory'`, then passed after implementation.

## Notes

- Existing warnings observed: legacy Gym/Gymnasium notice, Gym `Box bound precision lowered by casting to float64`, NumPy empty-slice warnings during very short evaluation windows, and Shapely pickle compatibility warning.
- OGM policy inputs printed by the runner: `target + action_mask + ogm`.
- The raw environment observation space still contains lidar because lidar remains enabled internally for action-mask generation; the Stage 4 SAC config excludes lidar from actor/critic policy inputs.

## Boundaries

This smoke validates OGM plumbing only. It is not a 20K quality gate and not an OGM-paper reproduction result.
