# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `command-only`
- Detail: `env-step`
- Render mode: `rgb_array`
- Verbose env: `False`
- Episodes: `16`
- Max steps per episode: `160`
- Updates requested/completed: `20` / `20`
- Total seconds: `24.420082`
- Measured component seconds: `65.171722`
- Transitions collected: `1496`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 5.622490 | 1496 | 3.758 | 8.63% |
| `SACAgent.update` | 1.064441 | 20 | 53.222 | 1.63% |
| `env.raw_step.total` | 14.568482 | 1512 | 9.635 | 22.35% |
| `env.render.action_mask` | 2.689294 | 1512 | 1.779 | 4.13% |
| `env.render.action_mask_post_process` | 0.089505 | 1512 | 0.059 | 0.14% |
| `env.render.draw` | 1.913603 | 1512 | 1.266 | 2.94% |
| `env.render.img_capture` | 2.187811 | 1512 | 1.447 | 3.36% |
| `env.render.img_process` | 1.780057 | 1512 | 1.177 | 2.73% |
| `env.render.lidar_fast_calc` | 0.359365 | 1512 | 0.238 | 0.55% |
| `env.render.lidar_get_observation` | 0.977537 | 1512 | 0.647 | 1.50% |
| `env.render.lidar_rotate_filter` | 0.595163 | 1512 | 0.394 | 0.91% |
| `env.render.lidar_total` | 0.984657 | 1512 | 0.651 | 1.51% |
| `env.render.target` | 0.038975 | 1512 | 0.026 | 0.06% |
| `env.render.total` | 10.010370 | 1512 | 6.621 | 15.36% |
| `env.reset` | 0.548796 | 16 | 34.300 | 0.84% |
| `env.reward.components` | 0.051052 | 1502 | 0.034 | 0.08% |
| `env.reward.total` | 0.054432 | 1512 | 0.036 | 0.08% |
| `env.rs.calc_all_paths` | 0.726441 | 796 | 0.913 | 1.11% |
| `env.rs.find_path` | 2.001082 | 796 | 2.514 | 3.07% |
| `env.rs.traj_valid` | 1.205821 | 2631 | 0.458 | 1.85% |
| `env.sim.kinematic_step` | 0.609379 | 11869 | 0.051 | 0.94% |
| `env.sim.vehicle_retreat` | 0.000451 | 423 | 0.001 | 0.00% |
| `env.sim.vehicle_step` | 1.324173 | 11869 | 0.112 | 2.03% |
| `env.status.check_arrived` | 0.299219 | 13371 | 0.022 | 0.46% |
| `env.status.check_time_exceeded` | 0.001407 | 1502 | 0.001 | 0.00% |
| `env.status.detect_collision` | 0.805599 | 13369 | 0.060 | 1.24% |
| `env.status.detect_outbound` | 0.007534 | 1506 | 0.005 | 0.01% |
| `env.status.total` | 0.199369 | 1506 | 0.132 | 0.31% |
| `env.step` | 14.303413 | 1496 | 9.561 | 21.95% |
| `env.wrapper.action_rescale` | 0.046110 | 1496 | 0.031 | 0.07% |
| `env.wrapper.observation_rescale` | 0.002679 | 1512 | 0.002 | 0.00% |
| `env.wrapper.reward_shaping` | 0.007691 | 1496 | 0.005 | 0.01% |
| `replay.push` | 0.091420 | 1496 | 0.061 | 0.14% |
| `replay.sample` | 0.003905 | 21 | 0.186 | 0.01% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`56`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.994364`
- Episode 2 `Complex`: steps=`65`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.980250`
- Episode 3 `Extrem`: steps=`11`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.421498`
- Episode 4 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.370757`
- Episode 5 `Normal`: steps=`25`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.654109`
- Episode 6 `Complex`: steps=`16`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.223220`
- Episode 7 `Extrem`: steps=`139`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.397632`
- Episode 8 `dlp`: steps=`97`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.859315`
- Episode 9 `Normal`: steps=`101`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.720582`
- Episode 10 `Complex`: steps=`18`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.000237`
- Episode 11 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.515317`
- Episode 12 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.751413`
- Episode 13 `Normal`: steps=`8`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.821339`
- Episode 14 `Complex`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.814872`
- Episode 15 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.951634`
- Episode 16 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.706679`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
- detail timings include nested calls; do not sum them as exclusive wall-clock percentages
