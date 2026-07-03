# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `original`
- Detail: `env-step`
- Render mode: `human`
- Verbose env: `True`
- Episodes: `16`
- Max steps per episode: `160`
- Updates requested/completed: `20` / `20`
- Total seconds: `24.498539`
- Measured component seconds: `64.737776`
- Transitions collected: `1544`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 5.848073 | 1544 | 3.788 | 9.03% |
| `SACAgent.update` | 1.034719 | 20 | 51.736 | 1.60% |
| `env.raw_step.total` | 14.487786 | 1560 | 9.287 | 22.38% |
| `env.render.action_mask` | 2.777167 | 1560 | 1.780 | 4.29% |
| `env.render.action_mask_post_process` | 0.090905 | 1560 | 0.058 | 0.14% |
| `env.render.draw` | 2.006710 | 1560 | 1.286 | 3.10% |
| `env.render.img_capture` | 2.205665 | 1560 | 1.414 | 3.41% |
| `env.render.img_process` | 1.891055 | 1560 | 1.212 | 2.92% |
| `env.render.lidar_fast_calc` | 0.336498 | 1560 | 0.216 | 0.52% |
| `env.render.lidar_get_observation` | 0.994453 | 1560 | 0.637 | 1.54% |
| `env.render.lidar_rotate_filter` | 0.634539 | 1560 | 0.407 | 0.98% |
| `env.render.lidar_total` | 1.001424 | 1560 | 0.642 | 1.55% |
| `env.render.target` | 0.044171 | 1560 | 0.028 | 0.07% |
| `env.render.total` | 10.438847 | 1560 | 6.692 | 16.12% |
| `env.reset` | 0.540914 | 16 | 33.807 | 0.84% |
| `env.reward.components` | 0.051188 | 1549 | 0.033 | 0.08% |
| `env.reward.total` | 0.055243 | 1560 | 0.035 | 0.09% |
| `env.rs.calc_all_paths` | 0.462669 | 449 | 1.030 | 0.71% |
| `env.rs.find_path` | 1.083039 | 449 | 2.412 | 1.67% |
| `env.rs.traj_valid` | 0.581481 | 1272 | 0.457 | 0.90% |
| `env.sim.kinematic_step` | 0.707843 | 13779 | 0.051 | 1.09% |
| `env.sim.vehicle_retreat` | 0.000355 | 262 | 0.001 | 0.00% |
| `env.sim.vehicle_step` | 1.533428 | 13779 | 0.111 | 2.37% |
| `env.status.check_arrived` | 0.317120 | 15328 | 0.021 | 0.49% |
| `env.status.check_time_exceeded` | 0.001306 | 1549 | 0.001 | 0.00% |
| `env.status.detect_collision` | 0.977932 | 15333 | 0.064 | 1.51% |
| `env.status.detect_outbound` | 0.008148 | 1557 | 0.005 | 0.01% |
| `env.status.total` | 0.203962 | 1557 | 0.131 | 0.32% |
| `env.step` | 14.261317 | 1544 | 9.237 | 22.03% |
| `env.wrapper.action_rescale` | 0.049201 | 1544 | 0.032 | 0.08% |
| `env.wrapper.observation_rescale` | 0.003002 | 1560 | 0.002 | 0.00% |
| `env.wrapper.reward_shaping` | 0.007803 | 1544 | 0.005 | 0.01% |
| `replay.push` | 0.096325 | 1544 | 0.062 | 0.15% |
| `replay.sample` | 0.003489 | 21 | 0.166 | 0.01% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.226604`
- Episode 2 `Complex`: steps=`60`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.584850`
- Episode 3 `Extrem`: steps=`63`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.216808`
- Episode 4 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.785440`
- Episode 5 `Normal`: steps=`50`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.619759`
- Episode 6 `Complex`: steps=`111`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.761848`
- Episode 7 `Extrem`: steps=`99`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.721933`
- Episode 8 `dlp`: steps=`131`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.841153`
- Episode 9 `Normal`: steps=`19`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.159230`
- Episode 10 `Complex`: steps=`37`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.569747`
- Episode 11 `Extrem`: steps=`63`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.640649`
- Episode 12 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.410780`
- Episode 13 `Normal`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-1.104217`
- Episode 14 `Complex`: steps=`65`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.644199`
- Episode 15 `Extrem`: steps=`46`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.520239`
- Episode 16 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.659869`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
- detail timings include nested calls; do not sum them as exclusive wall-clock percentages
