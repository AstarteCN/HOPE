# 2026-06-17 Stage 3 Baseline Resource Profile

## Scope

This note closes the original-HOPE SAC baseline run that started at `2026-06-17 23:38:10 +08:00`.

The run was originally launched toward 40,000 episodes, then intentionally stopped at the user's updated target of about 36,500 episodes after nearly 24 hours of wall-clock training. Treat this as the current local baseline for acceleration research, not as a completed 40,000-episode run.

## Run Identity

- Training run directory: `D:\Github\HOPE\src\log\exp\sac_20260617_233812`
- Runtime metadata: `D:\Github\HOPE\src\log\exp\stage3_baseline_40k_20260617_233810.meta.json`
- Final stop TensorBoard summary: `D:\Github\HOPE\src\log\exp\stage3_baseline_36500_stop.tb_summary.json`
- Resource CSV: `D:\Github\HOPE\src\log\exp\stage3_resource_40k_20260617_233810.csv`
- Final evaluation directory: `D:\Github\HOPE\src\log\eval\20260618_233234`
- Last periodic checkpoint evaluated: `D:\Github\HOPE\src\log\exp\sac_20260617_233812\SAC_35999.pt`
- Best checkpoint recorded during training: `D:\Github\HOPE\src\log\exp\sac_20260617_233812\SAC_best.pt`
- Best checkpoint note: `epoch: 34152, success rate: 1.0 1.0 0.92 0.95`

## Stop Result

The run was stopped at `2026-06-18 23:29:01 +08:00`.

- Episode count: `36502`
- Environment step count: `2753956`
- Estimated SAC updates after warmup: `274371`
- Stop target met for this run: `true` for `36500`
- Original 40,000-episode target: superseded for this run by the user's stop instruction
- Training child process, launcher process, and resource monitor process were all stopped.
- The obsolete heartbeat automation `hope-40k-baseline-monitor` was deleted.

Important caveat: because the process was stopped externally, the script did not run its built-in final evaluation block and did not save a checkpoint at exactly episode 36,502. The closest periodic checkpoint is `SAC_35999.pt`; this is what was evaluated below.

## TensorBoard Health

Final summary at 36,502 episodes:

- `total_reward`: last `6.2589`, mean `4.7650`, first-to-last-window delta `+4.9452`
- `avg_reward`: last `0.1381`, mean `0.0750`, first-to-last-window delta `+0.1427`
- `actor_loss`: last `-1.4654`, finite; no NaN/Inf
- `critic_loss`: last `0.1102`, finite; no NaN/Inf
- `alpha`: last `0.002560`
- `action_std0`: last log std `-0.5938`
- `action_std1`: last log std `-0.8362`
- Latest training moving success rates: Normal `0.99`, Complex `1.00`, Extrem `0.94`, DLP `0.90`
- `step_num`: last `36`, mean `75.45`, trend delta `-75.32`

Interpretation: the run remained healthy at stop time. There was no loss divergence, NaN/Inf, or latest-step timeout warning at the final snapshot.

## Resource Profile

Hardware observed:

- CPU: AMD Ryzen 7 9700X, 8 cores / 16 logical processors
- GPU: NVIDIA GeForce RTX 4080 SUPER, 16,376 MiB

Whole-run resource CSV:

- Samples: `16963`
- Sample window: `2026-06-17T23:38:20.8441316+08:00` to `2026-06-18T23:28:58.1670389+08:00`
- Duration: `23.84` hours
- Average normalized process CPU: `32.38%`
- Peak normalized process CPU: `49.09%`
- Average whole-GPU utilization: `25.76%`
- Peak whole-GPU utilization: `97%`
- Peak whole-GPU memory used: `3740 MB`
- Resource monitor sampling rate: about `11.86` samples/minute

Six-hour resource windows:

| Window | Samples | Avg CPU | Max CPU | Avg Whole-GPU | Max Whole-GPU |
|---|---:|---:|---:|---:|---:|
| 06-17 23:38 to 06-18 05:38 | 4256 | 44.49 | 49.09 | 31.26 | 95 |
| 06-18 05:38 to 06-18 11:38 | 4262 | 37.47 | 48.83 | 28.10 | 97 |
| 06-18 11:38 to 06-18 17:38 | 4278 | 26.77 | 45.18 | 24.27 | 85 |
| 06-18 17:38 to 06-18 23:28 | 4167 | 20.56 | 38.38 | 19.26 | 95 |

Interpretation: the training run did not saturate the CPU or GPU. Utilization fell as the policy improved and episodes shortened. The objective for acceleration should therefore be better time-to-good-policy, not artificially increasing resource occupancy.

## Final Checkpoint Evaluation

Command:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' '.\evaluation\eval_mix_scene.py' '.\log\exp\sac_20260617_233812\SAC_35999.pt' --eval_episode 200 --visualize "" --verbose ""
```

Results from `D:\Github\HOPE\src\log\eval\20260618_233234`:

| Scene | Success Rate | Success Step Num Mean | Step Std |
|---|---:|---:|---:|
| Normal | 1.000 | 21.715 | 16.825 |
| Complex | 0.985 | 27.487 | 22.965 |
| Extrem | 0.915 | 61.863 | 43.265 |
| DLP | 0.955 | 33.335 | 26.924 |

The evaluation supports treating this run as a strong local baseline despite the early stop.

## Bottleneck Evidence

Macro evidence:

- 36,502 episodes required about 23.84 hours.
- Average throughput was about 1,531 episodes/hour.
- Average environment-step throughput was about 32.1 environment steps/second.
- GPU utilization was low on average, but the run quality was high. GPU saturation is therefore not the success criterion.

Micro-profile evidence from a 500-step, no-source-change diagnostic:

| Component | Total Seconds | Count | ms/call |
|---|---:|---:|---:|
| `env.step_total` | 4.8749 | 500 | 9.750 |
| `env.render_total` | 3.1185 | 504 | 6.188 |
| `agent.action_total` | 1.7032 | 500 | 3.406 |
| `env.rs_probe` | 1.3022 | 381 | 3.418 |
| `agent.get_action` | 1.2783 | 400 | 3.196 |
| `env.action_mask` | 0.8457 | 504 | 1.678 |
| `agent.update` | 0.7773 | 20 | 38.863 |
| `env.img_capture` | 0.6698 | 504 | 1.329 |
| `env.img_process` | 0.5723 | 504 | 1.135 |
| `env.draw_pygame` | 0.4768 | 504 | 0.946 |
| `env.lidar` | 0.1938 | 504 | 0.385 |

Interpretation: environment stepping and observation/render work are the primary wall-clock costs, with RS probing and action-mask computation visible inside that path. SAC update cost is meaningful but not the dominant whole-run explanation by itself.

## Safe Acceleration Implications

Lowest-risk changes should first remove overhead that is not part of the learning semantics:

1. Use correct false-y CLI values: `--verbose ""` and, after parity checks, `--visualize ""`.
2. Avoid repeated `reward.png` generation and verbose printing during long quality runs.
3. Keep TensorBoard scalar logging because it provides training-health signals.
4. Add external profiling wrappers before editing original training code.

Render-mode parity check:

- Compared `render_mode=None` versus `render_mode='rgb_array'` under SDL dummy on Normal, Complex, Extrem, and DLP scripted-action cases.
- Max differences were `0.0` for `img`, `lidar`, `target`, `action_mask`, rewards, and statuses in the tested cases.
- This supports treating `--visualize ""` as a candidate for future quality runs, but it should still be validated on a broader scripted-case set before becoming default.

Higher-risk changes, such as larger batch size, altered update ratio, vectorized environment collection, replay tensorization, mixed precision, or `torch.compile`, should be treated as research experiments and validated against this baseline with matched evaluation.

## Training Failure Modes To Watch

Early warning indicators based on this baseline:

- `actor_loss` or `critic_loss` contains NaN/Inf.
- `critic_loss` repeatedly spikes far beyond the baseline maximum around `1.69`.
- `step_num` remains near `200` for several consecutive monitored snapshots while success rates stagnate.
- Normal/Complex success improves while Extrem/DLP success remains flat or degrades.
- Reward rises but held-out evaluation success does not improve.
- `alpha` or log std collapses much faster than baseline while exploration-heavy scenes underperform.
- Throughput improves only because observation, reward, action mask, or RS behavior changed.

## Next Decision

Recommended next run:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
& 'D:\Github\HOPE\.venv\Scripts\python.exe' '.\train\train_HOPE_sac.py' --train_episode 36500 --eval_episode 200 --visualize "" --verbose ""
```

This keeps the original training structure intact while disabling avoidable CLI/logging overhead. Compare throughput, TensorBoard curves, and final 200-episode evaluation against this baseline before attempting source-level optimization.
