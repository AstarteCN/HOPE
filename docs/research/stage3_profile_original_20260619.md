# Stage 3 Component Profile

- Status: `pass`
- Note: `measurement_only_not_quality_evidence`
- Profile mode: `original`
- Render mode: `human`
- Verbose env: `True`
- Episodes: `12`
- Max steps per episode: `120`
- Updates requested/completed: `20` / `20`
- Total seconds: `13.239007`
- Measured component seconds: `10.483979`
- Transitions collected: `630`

This is a bounded measurement-only diagnostic, not training-quality evidence.

## Components

| Component | Seconds | Calls | Avg ms | Percent measured wall |
| --- | ---: | ---: | ---: | ---: |
| `ParkingAgent.get_action` | 2.247208 | 630 | 3.567 | 21.43% |
| `SACAgent.update` | 1.162922 | 20 | 58.146 | 11.09% |
| `env.reset` | 0.516048 | 12 | 43.004 | 4.92% |
| `env.step` | 6.514738 | 630 | 10.341 | 62.14% |
| `replay.push` | 0.040075 | 630 | 0.064 | 0.38% |
| `replay.sample` | 0.002988 | 21 | 0.142 | 0.03% |

## Skipped Components

- None

## Episodes

- Episode 1 `Normal`: steps=`26`, done=`True`, status=`Status.ARRIVED`, total_reward=`6.130192`
- Episode 2 `Complex`: steps=`16`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.950639`
- Episode 3 `Extrem`: steps=`83`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.378748`
- Episode 4 `dlp`: steps=`92`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-6.017815`
- Episode 5 `Normal`: steps=`10`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.884480`
- Episode 6 `Complex`: steps=`15`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.968276`
- Episode 7 `Extrem`: steps=`36`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.796079`
- Episode 8 `dlp`: steps=`120`, done=`False`, status=`Status.CONTINUE`, total_reward=`-0.396139`
- Episode 9 `Normal`: steps=`32`, done=`True`, status=`Status.ARRIVED`, total_reward=`5.989271`
- Episode 10 `Complex`: steps=`33`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.573124`
- Episode 11 `Extrem`: steps=`55`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-5.507983`
- Episode 12 `dlp`: steps=`112`, done=`True`, status=`Status.OUTBOUND`, total_reward=`-6.101375`

## Notes

- measurement_only_not_quality_evidence
- bounded diagnostic only; do not use as Stage 3 training-quality evidence
- timing wrappers are installed only on objects created by this diagnostic process
