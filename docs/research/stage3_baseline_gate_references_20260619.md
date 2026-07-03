# Stage 3 Baseline Gate References

- Baseline run: `D:\Github\HOPE\src\log\exp\sac_20260617_233812`
- Purpose: reference metrics for command-only 0-100K gated validation.
- Scalar loading: TensorBoard `EventAccumulator(..., size_guidance={"scalars": 0})` full scalar read.

## Target 20000

- Episode scalar count to target: `20001`
- Env steps to target: `1821805`
- Estimated SAC updates after warmup: `181180`

| Scalar | Last | Last100 Mean | Last500 Mean | Has Nonfinite |
|--------|------|--------------|--------------|---------------|
| `total_reward` | 6.0459 | 4.51466 | 5.12935 | False |
| `avg_reward` | 0.0648426 | 0.0601137 | 0.0803388 | False |
| `actor_loss` | -0.26927 | -0.634152 | -0.526619 | False |
| `critic_loss` | 0.0679047 | 0.121498 | 0.137464 | False |
| `action_std0` | -0.267503 | -0.266454 | -0.261956 | False |
| `action_std1` | -0.51281 | -0.511532 | -0.50658 | False |
| `alpha` | 0.00406539 | 0.00407342 | 0.00410061 | False |
| `success_rate_Normal` | 1 | 1 | 1 | False |
| `success_rate_Complex` | 0.94 | 0.9556 | 0.96462 | True |
| `success_rate_Extrem` | 0.73 | 0.735 | 0.76768 | True |
| `success_rate_dlp` | 0.78 | 0.8166 | 0.8132 | True |
| `step_num` | 31 | 78.79 | 67.2 | False |

## Target 36500

- Episode scalar count to target: `36501`
- Env steps to target: `2753920`
- Estimated SAC updates after warmup: `274392`

| Scalar | Last | Last100 Mean | Last500 Mean | Has Nonfinite |
|--------|------|--------------|--------------|---------------|
| `total_reward` | 6.2395 | 5.88681 | 5.72137 | False |
| `avg_reward` | 0.149186 | 0.155738 | 0.137636 | False |
| `actor_loss` | -1.46541 | -1.48329 | -1.25511 | False |
| `critic_loss` | 0.110175 | 0.157341 | 0.136912 | False |
| `action_std0` | -0.593834 | -0.593032 | -0.589803 | False |
| `action_std1` | -0.836208 | -0.835496 | -0.832458 | False |
| `alpha` | 0.00256019 | 0.00256281 | 0.00257373 | False |
| `success_rate_Normal` | 0.99 | 0.99 | 0.99138 | False |
| `success_rate_Complex` | 1 | 1 | 0.9974 | True |
| `success_rate_Extrem` | 0.94 | 0.94 | 0.9305 | True |
| `success_rate_dlp` | 0.9 | 0.9102 | 0.91498 | True |
| `step_num` | 12 | 39.32 | 43.14 | False |

## External Eval Reference

- 36.5K comparable checkpoint: `D:\Github\HOPE\src\log\exp\sac_20260617_233812\SAC_35999.pt`
- Eval episodes per scene: `200`
- Normal: `1.0`
- Complex: `0.985`
- Extrem: `0.915`
- DLP: `0.955`
- Mean: `0.96375`

## Gate Policy

- 20K is an early-warning gate, not a final equivalence gate.
- 36.5K is the equivalence gate before allowing the command-only run to continue to 100K without intervention.
