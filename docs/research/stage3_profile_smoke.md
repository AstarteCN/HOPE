# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Episodes: `3`
- Max steps per episode: `50`
- Updates requested/completed: `5` / `5`
- Total seconds: `4.953086`
- Measured component seconds: `2.225658`
- Transitions collected: `116`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 0.606808 | 116 | 5.231 | 27.26% |
| `SACAgent.update` | 0.300466 | 5 | 60.093 | 13.50% |
| `env.reset` | 0.181781 | 3 | 60.594 | 8.17% |
| `env.step` | 1.128572 | 116 | 9.729 | 50.71% |
| `replay.push` | 0.007460 | 116 | 0.064 | 0.34% |
| `replay.sample` | 0.000571 | 6 | 0.095 | 0.03% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`50`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.103727`
- Episode 2 `Complex`: steps=`16`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.503948`
- Episode 3 `Extrem`: steps=`50`, done=`False`, status=`Status.CONTINUE`, total_reward=`0.145947`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
