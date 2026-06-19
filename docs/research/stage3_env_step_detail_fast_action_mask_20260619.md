# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `command-only`
- Action mask mode: `fast`
- Detail: `env-step`
- Render mode: `rgb_array`
- Verbose env: `False`
- Episodes: `16`
- Max steps per episode: `160`
- Updates requested/completed: `20` / `20`
- Total seconds: `22.188303`
- Measured component seconds: `57.588616`
- Transitions collected: `1551`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 5.492288 | 1551 | 3.541 | 9.54% |
| `SACAgent.update` | 0.892566 | 20 | 44.628 | 1.55% |
| `env.raw_step.total` | 12.973043 | 1567 | 8.279 | 22.53% |
| `env.render.action_mask` | 1.286215 | 1567 | 0.821 | 2.23% |
| `env.render.action_mask_post_process` | 0.077881 | 1567 | 0.050 | 0.14% |
| `env.render.draw` | 1.704352 | 1567 | 1.088 | 2.96% |
| `env.render.img_capture` | 2.189364 | 1567 | 1.397 | 3.80% |
| `env.render.img_process` | 1.843781 | 1567 | 1.177 | 3.20% |
| `env.render.lidar_fast_calc` | 0.271765 | 1567 | 0.173 | 0.47% |
| `env.render.lidar_get_observation` | 0.752298 | 1567 | 0.480 | 1.31% |
| `env.render.lidar_rotate_filter` | 0.459639 | 1567 | 0.293 | 0.80% |
| `env.render.lidar_total` | 0.757207 | 1567 | 0.483 | 1.31% |
| `env.render.target` | 0.023948 | 1567 | 0.015 | 0.04% |
| `env.render.total` | 8.769604 | 1567 | 5.596 | 15.23% |
| `env.reset` | 0.533002 | 16 | 33.313 | 0.93% |
| `env.reward.components` | 0.045821 | 1557 | 0.029 | 0.08% |
| `env.reward.total` | 0.048890 | 1567 | 0.031 | 0.08% |
| `env.rs.calc_all_paths` | 0.730279 | 754 | 0.969 | 1.27% |
| `env.rs.find_path` | 1.807675 | 754 | 2.397 | 3.14% |
| `env.rs.traj_valid` | 1.016769 | 2374 | 0.428 | 1.77% |
| `env.sim.kinematic_step` | 0.611604 | 12721 | 0.048 | 1.06% |
| `env.sim.vehicle_retreat` | 0.000451 | 417 | 0.001 | 0.00% |
| `env.sim.vehicle_step` | 1.316972 | 12721 | 0.104 | 2.29% |
| `env.status.check_arrived` | 0.263953 | 14278 | 0.018 | 0.46% |
| `env.status.check_time_exceeded` | 0.001050 | 1557 | 0.001 | 0.00% |
| `env.status.detect_collision` | 0.699138 | 14284 | 0.049 | 1.21% |
| `env.status.detect_outbound` | 0.008075 | 1565 | 0.005 | 0.01% |
| `env.status.total` | 0.149076 | 1565 | 0.095 | 0.26% |
| `env.step` | 12.725504 | 1551 | 8.205 | 22.10% |
| `env.wrapper.action_rescale` | 0.040379 | 1551 | 0.026 | 0.07% |
| `env.wrapper.observation_rescale` | 0.002549 | 1567 | 0.002 | 0.00% |
| `env.wrapper.reward_shaping` | 0.007564 | 1551 | 0.005 | 0.01% |
| `replay.push` | 0.082773 | 1551 | 0.053 | 0.14% |
| `replay.sample` | 0.003141 | 21 | 0.150 | 0.01% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.096041`
- Episode 2 `Complex`: steps=`45`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.051478`
- Episode 3 `Extrem`: steps=`63`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.557215`
- Episode 4 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.916569`
- Episode 5 `Normal`: steps=`19`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.888341`
- Episode 6 `Complex`: steps=`79`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.621988`
- Episode 7 `Extrem`: steps=`68`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.154207`
- Episode 8 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-1.066561`
- Episode 9 `Normal`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.860961`
- Episode 10 `Complex`: steps=`16`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.570132`
- Episode 11 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.771210`
- Episode 12 `dlp`: steps=`38`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.490333`
- Episode 13 `Normal`: steps=`28`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.511088`
- Episode 14 `Complex`: steps=`157`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.597981`
- Episode 15 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-1.061621`
- Episode 16 `dlp`: steps=`78`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.630994`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
- detail timings include nested calls; do not sum them as exclusive wall-clock percentages
