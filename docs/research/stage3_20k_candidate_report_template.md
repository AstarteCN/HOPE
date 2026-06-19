# Stage 3 20K Candidate Report

Date:
Run name:

## Candidate Evidence

- manifest path:
- candidate run dir:
- checkpoint path:
- TensorBoard summary JSON:
- TensorBoard summary Markdown:
- resource CSV:
- eval result dir(s):
- exact launch command:
- exact eval command:

## Decision

- Decision label:
- Baseline id: `stage3_command_only_20k_20260619`
- Candidate type:
- Changed knobs:
- Eligible for 36.5K or 100K follow-up:

Allowed decision labels and criteria:

- `pass`: quality and speed gates pass.
- `quality-pass-speed-neutral`: quality passes, speed gain below threshold.
- `investigate`: ambiguity requires rerun/deeper analysis.
- `reject`: hard reject or clear quality failure.

## Protected Boundaries

- `git diff -- src/train src/env src/model` output:
- Action mask semantics preserved:
- Curriculum and DLP scheduling preserved:
- Reward/terminal/observation semantics preserved:

## Speed

| Metric | Baseline 20K | Candidate 20K | Delta |
|---|---:|---:|---:|
| Wall time hours | `12.964` |  |  |
| Episodes/hour | `1545.71` |  |  |
| Env steps/sec | `39.10` |  |  |

## Quality

| Scene | Baseline Eval | Candidate Eval | Gate |
|---|---:|---:|---|
| Normal | `0.985` |  | `>=0.95` |
| Complex | `0.945` |  | `>=0.90` |
| Extrem | `0.655` |  | `>=0.58` |
| DLP | `0.960` |  | `>=0.91` |
| Mean | `0.88625` |  | `>=0.85625` |

## TensorBoard Signals

- Non-finite hard-reject metrics:
- Any watched non-finite metrics:
- Multi-scene collapse:
- Timeout/reward/success interaction:

## Resource Profile

- Process CPU:
- Whole-GPU utilization:
- Process RAM:
- GPU memory:
- Notes:

## Conclusion

State whether the candidate is accepted, neutral, requires investigation, or rejected.
