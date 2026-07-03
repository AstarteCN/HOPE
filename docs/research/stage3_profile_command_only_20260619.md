# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `command-only`
- Render mode: `rgb_array`
- Verbose env: `False`
- Episodes: `12`
- Max steps per episode: `120`
- Updates requested/completed: `20` / `20`
- Total seconds: `18.967075`
- Measured component seconds: `16.224460`
- Transitions collected: `1171`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 3.834631 | 1171 | 3.275 | 23.63% |
| `SACAgent.update` | 0.907330 | 20 | 45.366 | 5.59% |
| `env.reset` | 0.525173 | 12 | 43.764 | 3.24% |
| `env.step` | 10.881386 | 1171 | 9.292 | 67.07% |
| `replay.push` | 0.072904 | 1171 | 0.062 | 0.45% |
| `replay.sample` | 0.003036 | 21 | 0.145 | 0.02% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`100`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.728597`
- Episode 2 `Complex`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.441097`
- Episode 3 `Extrem`: steps=`118`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.784120`
- Episode 4 `dlp`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.608923`
- Episode 5 `Normal`: steps=`41`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.725395`
- Episode 6 `Complex`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.412077`
- Episode 7 `Extrem`: steps=`114`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.687851`
- Episode 8 `dlp`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.270282`
- Episode 9 `Normal`: steps=`96`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.623642`
- Episode 10 `Complex`: steps=`14`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.984003`
- Episode 11 `Extrem`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.972053`
- Episode 12 `dlp`: steps=`88`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.706082`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
