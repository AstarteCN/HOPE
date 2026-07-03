# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `command-only`
- Action mask mode: `original`
- Detail: `env-step`
- Render mode: `rgb_array`
- Verbose env: `False`
- Episodes: `16`
- Max steps per episode: `160`
- Updates requested/completed: `20` / `20`
- Total seconds: `25.448460`
- Measured component seconds: `69.861138`
- Transitions collected: `1654`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 6.074745 | 1654 | 3.673 | 8.70% |
| `SACAgent.update` | 0.885288 | 20 | 44.264 | 1.27% |
| `env.raw_step.total` | 15.683584 | 1670 | 9.391 | 22.45% |
| `env.render.action_mask` | 2.828447 | 1670 | 1.694 | 4.05% |
| `env.render.action_mask_post_process` | 0.093815 | 1670 | 0.056 | 0.13% |
| `env.render.draw` | 2.047639 | 1670 | 1.226 | 2.93% |
| `env.render.img_capture` | 2.269666 | 1670 | 1.359 | 3.25% |
| `env.render.img_process` | 1.910586 | 1670 | 1.144 | 2.73% |
| `env.render.lidar_fast_calc` | 0.349570 | 1670 | 0.209 | 0.50% |
| `env.render.lidar_get_observation` | 1.005433 | 1670 | 0.602 | 1.44% |
| `env.render.lidar_rotate_filter` | 0.632118 | 1670 | 0.379 | 0.90% |
| `env.render.lidar_total` | 1.011500 | 1670 | 0.606 | 1.45% |
| `env.render.target` | 0.045548 | 1670 | 0.027 | 0.07% |
| `env.render.total` | 10.524653 | 1670 | 6.302 | 15.07% |
| `env.reset` | 0.517878 | 16 | 32.367 | 0.74% |
| `env.reward.components` | 0.052266 | 1660 | 0.031 | 0.07% |
| `env.reward.total` | 0.055649 | 1670 | 0.033 | 0.08% |
| `env.rs.calc_all_paths` | 0.837235 | 897 | 0.933 | 1.20% |
| `env.rs.find_path` | 2.434892 | 897 | 2.714 | 3.49% |
| `env.rs.traj_valid` | 1.525897 | 2788 | 0.547 | 2.18% |
| `env.sim.kinematic_step` | 0.665838 | 13647 | 0.049 | 0.95% |
| `env.sim.vehicle_retreat` | 0.000401 | 401 | 0.001 | 0.00% |
| `env.sim.vehicle_step` | 1.434706 | 13647 | 0.105 | 2.05% |
| `env.status.check_arrived` | 0.306579 | 15307 | 0.020 | 0.44% |
| `env.status.check_time_exceeded` | 0.001447 | 1660 | 0.001 | 0.00% |
| `env.status.detect_collision` | 0.850556 | 15307 | 0.056 | 1.22% |
| `env.status.detect_outbound` | 0.008165 | 1665 | 0.005 | 0.01% |
| `env.status.total` | 0.201577 | 1665 | 0.121 | 0.29% |
| `env.step` | 15.456403 | 1654 | 9.345 | 22.12% |
| `env.wrapper.action_rescale` | 0.044594 | 1654 | 0.027 | 0.06% |
| `env.wrapper.observation_rescale` | 0.003041 | 1670 | 0.002 | 0.00% |
| `env.wrapper.reward_shaping` | 0.007755 | 1654 | 0.005 | 0.01% |
| `replay.push` | 0.090594 | 1654 | 0.055 | 0.13% |
| `replay.sample` | 0.003075 | 21 | 0.146 | 0.00% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`119`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.703465`
- Episode 2 `Complex`: steps=`15`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.197545`
- Episode 3 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.617074`
- Episode 4 `dlp`: steps=`123`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.959676`
- Episode 5 `Normal`: steps=`104`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.632572`
- Episode 6 `Complex`: steps=`89`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.853635`
- Episode 7 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.398293`
- Episode 8 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.476615`
- Episode 9 `Normal`: steps=`20`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.374501`
- Episode 10 `Complex`: steps=`30`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.202382`
- Episode 11 `Extrem`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.505902`
- Episode 12 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-1.028950`
- Episode 13 `Normal`: steps=`44`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.497361`
- Episode 14 `Complex`: steps=`10`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.247116`
- Episode 15 `Extrem`: steps=`140`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.922690`
- Episode 16 `dlp`: steps=`160`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.707062`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
- detail timings include nested calls; do not sum them as exclusive wall-clock percentages
