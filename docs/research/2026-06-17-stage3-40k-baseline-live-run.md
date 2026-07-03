# Stage 3 40K Baseline Live Run

Date: 2026-06-17

## Status

In progress. This is the true Stage 3 original HOPE SAC baseline run, but it is not complete until it reaches at least 40,000 training episodes and final reporting is written.

## Command

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
..\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 40000 --eval_episode 200 --visualize False --verbose True
```

## Runtime Metadata

- Stamp: `20260617_233810`
- Launcher PID: `17768`
- Child training PID: `22224`
- Monitored PID: `22224`
- Current monitor PID: `19512` (initial `25356`, restarted as `34596`, then `19512`)
- TensorBoard URL: `http://127.0.0.1:6006`
- TensorBoard run directory: `D:\Github\HOPE\src\log\exp\sac_20260617_233812`
- Metadata file: `D:\Github\HOPE\src\log\exp\stage3_baseline_40k_20260617_233810.meta.json`
- Stdout log: `D:\Github\HOPE\src\log\exp\stage3_baseline_40k_20260617_233810.stdout.log`
- Stderr log: `D:\Github\HOPE\src\log\exp\stage3_baseline_40k_20260617_233810.stderr.log`
- Resource CSV: `D:\Github\HOPE\src\log\exp\stage3_resource_40k_20260617_233810.csv`

## Early Snapshot

At an early live check:

- Episode count: `103`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `12193`
- Estimated SAC updates after warmup: `195`
- Resource samples: `35`
- Average process CPU percent: `13.04`
- Average whole-GPU percent: `32.4`
- Peak whole-GPU percent: `45`
- Peak whole-GPU memory used MB: `3054`

GPU utilization values are whole-device `nvidia-smi` samples. They are useful for coarse resource trends but are not PID-level GPU attribution.

## Latest Immediate Check

At the latest immediate check before handing off to heartbeat monitoring:

- Episode count: `198`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `21551`
- Estimated SAC updates after warmup: `1131`
- Resource samples: `74`
- Average process CPU percent: `29.65`
- Average whole-GPU percent: `31.5`
- Peak whole-GPU percent: `58`
- Peak whole-GPU memory used MB: `3086`

## Heartbeat Check: 2026-06-18 00:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `938`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `119667`
- Estimated SAC updates after warmup: `10942`
- Resource samples: `422`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T00:13:48.6299056+08:00`
- Average process CPU percent: `43.84`
- Peak process CPU percent: `48.86`
- Average whole-GPU percent: `31.06`
- Peak whole-GPU percent: `69`
- Peak whole-GPU memory used MB: `3256`
- TensorBoard warning: `episode_count 938 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss`, `critic_loss`, `alpha`, `action_std0`, or `action_std1`. `total_reward` and `avg_reward` had positive trend deltas in the first-to-last 50-episode windows. The stdout tail still prints `nan nan` for the script's rolling loss console line, which appears to come from empty/early rolling lists; TensorBoard loss scalars were finite at this check.

## Heartbeat Check: 2026-06-18 00:43

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `1812`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `219445`
- Estimated SAC updates after warmup: `20920`
- Resource samples: `778`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T00:43:47.4693180+08:00`
- Average process CPU percent: `45.05`
- Peak process CPU percent: `48.92`
- Average whole-GPU percent: `31.01`
- Peak whole-GPU percent: `69`
- Peak whole-GPU memory used MB: `3278`
- TensorBoard warning: `episode_count 1812 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` continued to show positive trend deltas. Latest scene success rates were Normal `0.84`, Complex `0.79`, Extrem `0.27`, and DLP `0.65`. The run remains far below the 40,000-episode budget, so these are progress indicators only.

## Heartbeat Check: 2026-06-18 01:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `2681`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `317285`
- Estimated SAC updates after warmup: `30704`
- Resource samples: `1136`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T01:13:55.8461145+08:00`
- Average process CPU percent: `45.37`
- Peak process CPU percent: `48.99`
- Average whole-GPU percent: `30.88`
- Peak whole-GPU percent: `69`
- Peak whole-GPU memory used MB: `3278`
- TensorBoard warnings: `episode_count 2681 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.82`, Complex `0.75`, Extrem `0.38`, and DLP `0.64`. The last recorded episode hit `step_num=200`; treat this as an early-warning sample to keep watching, not as a standalone failure conclusion.

## Heartbeat Check: 2026-06-18 01:43

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `3566`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `412868`
- Estimated SAC updates after warmup: `40262`
- Resource samples: `1492`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T01:43:54.3881338+08:00`
- Average process CPU percent: `45.46`
- Peak process CPU percent: `48.99`
- Average whole-GPU percent: `31.08`
- Peak whole-GPU percent: `69`
- Peak whole-GPU memory used MB: `3278`
- TensorBoard warning: `episode_count 3566 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.85`, Complex `0.77`, Extrem `0.47`, and DLP `0.67`. The previous heartbeat's last-episode timeout warning did not recur in the latest `step_num` value (`123`).

## Heartbeat Check: 2026-06-18 02:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `4438`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `506112`
- Estimated SAC updates after warmup: `49587`
- Resource samples: `1848`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T02:13:52.3604838+08:00`
- Average process CPU percent: `45.45`
- Peak process CPU percent: `48.99`
- Average whole-GPU percent: `31.06`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 4438 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.84`, Complex `0.79`, Extrem `0.35`, and DLP `0.66`. The latest `step_num` value was `22`, so the previous timeout sample has not persisted in the most recent episode.

## Heartbeat Check: 2026-06-18 02:43

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `5373`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `603081`
- Estimated SAC updates after warmup: `59284`
- Resource samples: `2223`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T02:45:27.5163015+08:00`
- Average process CPU percent: `45.39`
- Peak process CPU percent: `48.99`
- Average whole-GPU percent: `31.13`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 5373 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.89`, Complex `0.83`, Extrem `0.36`, and DLP `0.64`. The latest `step_num` value was `47`, so the previously seen timeout sample did not persist at this snapshot; keep watching for recurring near-`200` episodes as an early warning pattern rather than a standalone failure conclusion.

## Heartbeat Check: 2026-06-18 03:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `6174`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `684683`
- Estimated SAC updates after warmup: `67444`
- Resource samples: `2561`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T03:13:54.6389168+08:00`
- Average process CPU percent: `45.26`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.09`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 6174 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.92`, Complex `0.76`, Extrem `0.43`, and DLP `0.66`. The latest `step_num` value was `29`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 03:43

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `7083`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `771519`
- Estimated SAC updates after warmup: `76127`
- Resource samples: `2917`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T03:43:53.4953272+08:00`
- Average process CPU percent: `45.13`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.12`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warnings: `episode_count 7083 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.92`, Complex `0.85`, Extrem `0.47`, and DLP `0.74`. The latest `step_num` value was `200`; this timeout sample has recurred after earlier non-timeout snapshots, so keep watching it as a possible early-warning pattern but not yet a standalone failure conclusion.

## Heartbeat Check: 2026-06-18 04:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `7947`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `855907`
- Estimated SAC updates after warmup: `84566`
- Resource samples: `3274`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T04:13:56.9609650+08:00`
- Average process CPU percent: `44.97`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.26`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 7947 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.93`, Complex `0.78`, Extrem `0.57`, and DLP `0.62`. The latest `step_num` value was `90`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 04:43

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `8826`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `938316`
- Estimated SAC updates after warmup: `92807`
- Resource samples: `3630`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T04:43:55.1943964+08:00`
- Average process CPU percent: `44.81`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.27`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warnings: `episode_count 8826 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.94`, Complex `0.85`, Extrem `0.46`, and DLP `0.70`. The latest `step_num` value was `200`; the timeout sample has recurred and should continue to be monitored as an early-warning pattern, but current loss and reward signals do not indicate a run failure.

## Heartbeat Check: 2026-06-18 05:13

The heartbeat monitor confirmed the training process and resource monitor were still running.

- Episode count: `9687`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1018732`
- Estimated SAC updates after warmup: `100849`
- Resource samples: `3986`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T05:13:53.5601559+08:00`
- Average process CPU percent: `44.62`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.25`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warnings: `episode_count 9687 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.97`, Complex `0.84`, Extrem `0.51`, and DLP `0.73`. The latest `step_num` value was `200`; this recurring timeout sample remains worth watching, but current loss and reward signals still do not indicate a run failure.

Resource-monitor recovery: after the 05:13 snapshot, monitor PID `25356` was no longer running while the training process was still active. The monitor script was updated so an existing CSV is appended to instead of overwritten, then the resource monitor was restarted as PID `34596` against the same CSV. A follow-up sample confirmed the CSV continued appending, reaching `3991` samples with last timestamp `2026-06-18T05:16:01.2160843+08:00`.

## Heartbeat Check: 2026-06-18 05:43

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `10532`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1098069`
- Estimated SAC updates after warmup: `108782`
- Resource samples: `4323`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T05:43:58.2648155+08:00`
- Average process CPU percent: `44.47`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.25`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 10532 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.90`, Complex `0.84`, Extrem `0.38`, and DLP `0.81`. The latest `step_num` value was `23`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 06:13

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `11382`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1175578`
- Estimated SAC updates after warmup: `116533`
- Resource samples: `4679`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T06:13:56.6163207+08:00`
- Average process CPU percent: `44.26`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.19`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warnings: `episode_count 11382 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.95`, Complex `0.89`, Extrem `0.57`, and DLP `0.73`. The latest `step_num` value was `200`; this recurring timeout sample remains an early-warning signal to watch, while the loss and reward trends remain healthy.

## Heartbeat Check: 2026-06-18 06:43

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `12361`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1255349`
- Estimated SAC updates after warmup: `124510`
- Resource samples: `5056`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T06:45:40.7281906+08:00`
- Average process CPU percent: `44.01`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `31.11`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 12361 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.94`, Complex `0.91`, Extrem `0.59`, and DLP `0.83`. The latest `step_num` value was `38`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 07:13

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `13217`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1324321`
- Estimated SAC updates after warmup: `131408`
- Resource samples: `5392`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T07:13:58.7986494+08:00`
- Average process CPU percent: `43.78`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.96`
- Peak whole-GPU percent: `95`
- Peak whole-GPU memory used MB: `3303`
- TensorBoard warning: `episode_count 13217 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.92`, Complex `0.88`, Extrem `0.57`, and DLP `0.79`. The latest `step_num` value was `18`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 07:43

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `14112`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1396127`
- Estimated SAC updates after warmup: `138588`
- Resource samples: `5748`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T07:43:56.9913114+08:00`
- Average process CPU percent: `43.52`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.98`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warnings: `episode_count 14112 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.86`, Extrem `0.68`, and DLP `0.85`. The latest `step_num` value was `200`; this timeout sample has recurred, but current loss, reward, and scene success signals still do not indicate a failed run.

## Heartbeat Check: 2026-06-18 08:13

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `15053`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1465661`
- Estimated SAC updates after warmup: `145542`
- Resource samples: `6105`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T08:13:59.9722065+08:00`
- Average process CPU percent: `43.24`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.86`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 15053 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.93`, Complex `0.88`, Extrem `0.70`, and DLP `0.91`. The latest `step_num` value was `17`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 08:43

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `15903`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1533805`
- Estimated SAC updates after warmup: `152356`
- Resource samples: `6462`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T08:44:03.1561589+08:00`
- Average process CPU percent: `42.97`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.84`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 15903 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.94`, Complex `0.92`, Extrem `0.59`, and DLP `0.87`. The latest `step_num` value was `8`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 09:13

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `16772`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1599400`
- Estimated SAC updates after warmup: `158916`
- Resource samples: `6818`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T09:14:00.9151392+08:00`
- Average process CPU percent: `42.66`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.67`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 16772 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.93`, Complex `0.93`, Extrem `0.74`, and DLP `0.81`. The latest `step_num` value was `66`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 09:43

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `17701`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1663404`
- Estimated SAC updates after warmup: `165316`
- Resource samples: `7175`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T09:44:03.9969935+08:00`
- Average process CPU percent: `42.35`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.58`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 17701 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.98`, Complex `0.92`, Extrem `0.70`, and DLP `0.85`. The latest `step_num` value was `124`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 10:13

The heartbeat monitor confirmed the training process and restarted resource monitor were still running.

- Episode count: `18587`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1724598`
- Estimated SAC updates after warmup: `171435`
- Resource samples: `7532`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T10:14:06.3843850+08:00`
- Average process CPU percent: `42.01`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.34`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 18587 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.92`, Extrem `0.70`, and DLP `0.89`. The latest `step_num` value was `19`, so the timeout warning did not recur at this snapshot.

Resource-monitor recovery: after the 10:13 snapshot, monitor PID `34596` was no longer running while the training process was still active. The resource monitor was restarted as PID `19512` against the same CSV, and the metadata file was updated. A follow-up sample confirmed the CSV continued appending, reaching `7537` samples with last timestamp `2026-06-18T10:15:43.5617160+08:00`.

## Heartbeat Check: 2026-06-18 10:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `19469`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1785915`
- Estimated SAC updates after warmup: `177567`
- Resource samples: `7888`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T10:45:15.9958318+08:00`
- Average process CPU percent: `41.66`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `30.04`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 19469 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.95`, Extrem `0.79`, and DLP `0.81`. The latest `step_num` value was `29`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 11:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `20260`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1838002`
- Estimated SAC updates after warmup: `182776`
- Resource samples: `8228`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T11:13:52.8966834+08:00`
- Average process CPU percent: `41.27`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `29.86`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 20260 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.91`, Extrem `0.75`, and DLP `0.84`. The latest `step_num` value was `80`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 11:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `21126`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1893703`
- Estimated SAC updates after warmup: `188346`
- Resource samples: `8585`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T11:43:55.1580477+08:00`
- Average process CPU percent: `40.91`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `29.65`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 21126 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.94`, Extrem `0.75`, and DLP `0.87`. The latest `step_num` value was `11`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 12:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `22035`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `1948185`
- Estimated SAC updates after warmup: `193794`
- Resource samples: `8943`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T12:14:02.8662434+08:00`
- Average process CPU percent: `40.55`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `29.38`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 22035 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.96`, Extrem `0.75`, and DLP `0.86`. The latest `step_num` value was `11`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 12:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `22851`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2000424`
- Estimated SAC updates after warmup: `199018`
- Resource samples: `9299`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T12:44:00.4505246+08:00`
- Average process CPU percent: `40.19`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `29.17`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3433`
- TensorBoard warning: `episode_count 22851 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.97`, Extrem `0.81`, and DLP `0.87`. The latest `step_num` value was `99`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 13:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `23634`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2047873`
- Estimated SAC updates after warmup: `203763`
- Resource samples: `9655`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T13:13:58.3516469+08:00`
- Average process CPU percent: `39.77`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.97`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 23634 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.94`, Extrem `0.79`, and DLP `0.82`. The latest `step_num` value was `24`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 13:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `24478`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2094869`
- Estimated SAC updates after warmup: `208462`
- Resource samples: `10013`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T13:44:05.9997459+08:00`
- Average process CPU percent: `39.37`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.67`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 24478 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.98`, Extrem `0.83`, and DLP `0.81`. The latest `step_num` value was `136`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 14:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `25219`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2139060`
- Estimated SAC updates after warmup: `212882`
- Resource samples: `10369`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T14:14:03.6540185+08:00`
- Average process CPU percent: `38.96`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.43`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 25219 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.98`, Extrem `0.81`, and DLP `0.89`. The latest `step_num` value was `13`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 14:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `26042`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2184733`
- Estimated SAC updates after warmup: `217449`
- Resource samples: `10726`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T14:44:06.0308595+08:00`
- Average process CPU percent: `38.59`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.18`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 26042 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.98`, Complex `0.98`, Extrem `0.84`, and DLP `0.87`. The latest `step_num` value was `18`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 15:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `26803`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2226640`
- Estimated SAC updates after warmup: `221640`
- Resource samples: `11081`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T15:14:00.6839017+08:00`
- Average process CPU percent: `38.19`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.99`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 26803 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.97`, Extrem `0.92`, and DLP `0.81`. The latest `step_num` value was `11`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 15:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `27466`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2266633`
- Estimated SAC updates after warmup: `225639`
- Resource samples: `11438`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T15:44:02.8447862+08:00`
- Average process CPU percent: `37.79`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.02`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warnings: `episode_count 27466 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows despite a negative latest `total_reward` sample. Latest scene success rates were Normal `0.97`, Complex `0.99`, Extrem `0.85`, and DLP `0.89`. The latest `step_num` value was `200`, so the timeout warning recurred and should remain on the early-warning watch list.

## Heartbeat Check: 2026-06-18 16:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `28154`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2304733`
- Estimated SAC updates after warmup: `229449`
- Resource samples: `11795`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T16:14:05.2801925+08:00`
- Average process CPU percent: `37.38`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.07`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3681`
- TensorBoard warning: `episode_count 28154 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.98`, Extrem `0.92`, and DLP `0.80`. The latest `step_num` value was `23`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 16:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `28767`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2340737`
- Estimated SAC updates after warmup: `233049`
- Resource samples: `12152`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T16:44:08.1223338+08:00`
- Average process CPU percent: `36.97`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.97`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 28767 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.99`, Extrem `0.91`, and DLP `0.76`. The latest `step_num` value was `12`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 17:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `29368`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2374104`
- Estimated SAC updates after warmup: `236386`
- Resource samples: `12508`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T17:14:05.2857524+08:00`
- Average process CPU percent: `36.55`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `28.01`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 29368 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.96`, Extrem `0.89`, and DLP `0.81`. The latest `step_num` value was `48`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 17:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `29941`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2408546`
- Estimated SAC updates after warmup: `239830`
- Resource samples: `12868`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T17:44:22.7511618+08:00`
- Average process CPU percent: `36.15`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.81`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 29941 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.97`, Complex `0.97`, Extrem `0.83`, and DLP `0.79`. The latest `step_num` value was `67`, so the timeout warning did not recur at this snapshot. This snapshot is close enough to 30,000 episodes to reinforce the preliminary resource-utilization conclusion, but it still does not close the requested 40,000-episode baseline.

## Heartbeat Check: 2026-06-18 18:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running. This snapshot crossed the 30,000-episode mark, which is useful for preliminary resource and bottleneck judgment, but the requested 40,000-episode baseline remains incomplete.

- Episode count: `30520`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2445989`
- Estimated SAC updates after warmup: `243574`
- Resource samples: `13236`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T18:15:20.8101727+08:00`
- Average process CPU percent: `35.80`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.63`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 30520 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.98`, Extrem `0.84`, and DLP `0.75`. The latest `step_num` value was `54`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 18:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `31070`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2477760`
- Estimated SAC updates after warmup: `246752`
- Resource samples: `13576`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T18:43:57.6699441+08:00`
- Average process CPU percent: `35.46`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.38`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 31070 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.99`, Extrem `0.92`, and DLP `0.79`. The latest `step_num` value was `50`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 19:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `31654`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2509566`
- Estimated SAC updates after warmup: `249932`
- Resource samples: `13933`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T19:14:00.2182082+08:00`
- Average process CPU percent: `35.09`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `27.05`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 31654 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `1.00`, Extrem `0.91`, and DLP `0.85`. The latest `step_num` value was `24`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 19:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `32249`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2541512`
- Estimated SAC updates after warmup: `253127`
- Resource samples: `14290`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T19:44:02.6636924+08:00`
- Average process CPU percent: `34.75`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `26.75`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 32249 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.97`, Extrem `0.90`, and DLP `0.89`. The latest `step_num` value was `23`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 20:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `32750`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2570732`
- Estimated SAC updates after warmup: `256049`
- Resource samples: `14647`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T20:14:05.0976337+08:00`
- Average process CPU percent: `34.39`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `26.49`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 32750 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.98`, Complex `0.98`, Extrem `0.89`, and DLP `0.80`. The latest `step_num` value was `11`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 20:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `33291`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2599153`
- Estimated SAC updates after warmup: `258891`
- Resource samples: `15002`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T20:43:58.2437940+08:00`
- Average process CPU percent: `34.05`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `26.20`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 33291 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `1.00`, Extrem `0.89`, and DLP `0.84`. The latest `step_num` value was `29`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 21:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `33826`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2628426`
- Estimated SAC updates after warmup: `261818`
- Resource samples: `15360`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T21:14:05.5318505+08:00`
- Average process CPU percent: `33.72`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `25.96`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 33826 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `1.00`, Complex `0.99`, Extrem `0.86`, and DLP `0.83`. The latest `step_num` value was `23`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 21:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `34489`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2657445`
- Estimated SAC updates after warmup: `264720`
- Resource samples: `15715`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T21:43:58.0305610+08:00`
- Average process CPU percent: `33.41`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `25.78`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 34489 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.98`, Extrem `0.93`, and DLP `0.89`. The latest `step_num` value was `39`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 22:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `35111`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2687415`
- Estimated SAC updates after warmup: `267717`
- Resource samples: `16077`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T22:14:25.2413190+08:00`
- Average process CPU percent: `33.12`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `25.80`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 35111 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.99`, Extrem `0.90`, and DLP `0.85`. The latest `step_num` value was `8`, so the timeout warning did not recur at this snapshot.

## Heartbeat Check: 2026-06-18 22:43

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `35595`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2714233`
- Estimated SAC updates after warmup: `270399`
- Resource samples: `16435`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T22:44:32.4946716+08:00`
- Average process CPU percent: `32.81`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `25.82`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warnings: `episode_count 35595 is below required minimum 40000`; `final step_num is near the 200-step timeout`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `0.98`, Extrem `0.90`, and DLP `0.83`. The latest `step_num` value was `200`, so the timeout warning recurred at this snapshot and should remain on the early-warning watch list, but it is not by itself a training-failure signal because the reward and success-rate trends remain healthy.

## Heartbeat Check: 2026-06-18 23:13

The heartbeat monitor confirmed the launcher process, training process, and restarted resource monitor were still running.

- Episode count: `36206`
- Required minimum episodes: `40000`
- Training budget met: `false`
- Environment step count: `2741287`
- Estimated SAC updates after warmup: `273104`
- Resource samples: `16787`
- Resource sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T23:14:09.5637495+08:00`
- Average process CPU percent: `32.53`
- Peak process CPU percent: `49.09`
- Average whole-GPU percent: `25.83`
- Peak whole-GPU percent: `97`
- Peak whole-GPU memory used MB: `3740`
- TensorBoard warning: `episode_count 36206 is below required minimum 40000`

TensorBoard scalar checks at this heartbeat showed no NaN/Inf in `actor_loss` or `critic_loss`. `total_reward` and `avg_reward` stayed positive in the first-to-last trend windows. Latest scene success rates were Normal `0.99`, Complex `1.00`, Extrem `0.94`, and DLP `0.92`. The latest `step_num` value was `15`, so the timeout warning did not recur at this snapshot.

## User-Directed Stop: 2026-06-18 23:29

The user requested stopping the run around 36,500 episodes and using it as the current baseline for safe acceleration research.

- Final stopped episode count: `36502`
- Environment step count: `2753956`
- Estimated SAC updates after warmup: `274371`
- Stop target met: `36500`
- Training child PID `22224`, launcher PID `17768`, and monitor PID `19512` were stopped.
- The heartbeat automation `hope-40k-baseline-monitor` was deleted because the 40K monitor objective was superseded.
- Final baseline profile: `docs/research/2026-06-17-stage3-baseline-resource-profile.md`

The run should not be described as a completed 40,000-episode baseline. It is the user-approved 36,500-episode baseline for the next acceleration research step.

## Notes

- `git diff -- src` produced no output before and during the early run checks.
- TensorBoard summary should use the locked run directory from metadata, not the newest `sac_*` directory.
- This live snapshot is for monitoring only and must not be interpreted as baseline training quality.
