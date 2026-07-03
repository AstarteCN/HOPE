# Changelog

## 2026-06-20 - Stage 3 Fast Action-Mask 20K Candidate

- Added and validated the opt-in fast action-mask Stage 3 candidate workflow.
- Confirmed the 20K fast action-mask candidate passed the saved command-only 20K speed and quality gates.
- Recorded final speed metrics to `SAC_19999.pt`: `10.947863 h`, `1826.840564` episodes/hour, and `46.223481` environment steps/second.
- Recorded improvements over the command-only 20K baseline: `15.55%` wall-clock, `18.19%` episodes/hour, and `18.22%` environment steps/second.
- Recorded matched 200-episode external eval parity with the command-only 20K baseline: Normal `0.985`, Complex `0.945`, Extrem `0.655`, DLP `0.960`, mean `0.88625`.
- Documented the caveat that this is a 20K smoke-quality pass, not a 36.5K reproduction-equivalence proof.
- Preserved original `src/train` and `src/env` paths during the candidate reporting work.
