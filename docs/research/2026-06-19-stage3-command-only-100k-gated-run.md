# Stage 3 Command-Only 20K Smoke Stop Plan

## Objective

Run one continuous command-only HOPE SAC training process until about 20,000 episodes, then stop it as a smoke-quality check while preserving the original training script and training state up to the stop point.

The run uses the validated command-only flags:

```powershell
--visualize= --verbose=
```

This avoids the original `argparse type=bool` trap where `--visualize False --verbose False` still evaluates to `True`.

## 2026-06-19 Scope Update

This run was originally planned as a 0-100K gated validation with 20K, 36.5K, and 100K checkpoints. The user superseded that plan after the long-run speed improvement appeared limited: the run should now stop around 20K episodes and be treated as a smoke-quality check, not as a full 36.5K/100K validation.

The one-time heartbeat hook `hope-command-only-100k-gated-monitor` has been updated to check every 15 minutes, stop after `SAC_19999.pt` exists and TensorBoard is around 20K episodes, evaluate/report the stopped smoke run, start a research-only safe speed-architecture note if quality is not negatively impacted, and then delete itself.

## Why One Continuous Run

Do not split the training into a hard 0-36.5K process followed by a restart. Restarting from `--agent_ckpt` would reload parameters but not replay buffer, optimizer state, RNG state, scene chooser state, or training counters. That would make the 36.5K-to-100K continuation a different experiment.

Instead, this run keeps one continuous process until the new 20K stop target:

- Preserve replay buffer, optimizer state, RNG state, scene chooser state, and counters through the 20K smoke.
- Stop externally after `SAC_19999.pt` is present.
- Do not continue to 36.5K or 100K unless the user explicitly re-approves continuation.

## Command

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
D:\Github\HOPE\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 100000 --eval_episode 200 --visualize= --verbose=
```

`--eval_episode 200` keeps the final built-in eval comparable to the stopped 36.5K local baseline evaluation.

## Gate References

Reference file:

- `docs/research/stage3_baseline_gate_references_20260619.json`
- `docs/research/stage3_baseline_gate_references_20260619.md`

Local baseline run:

- `D:\Github\HOPE\src\log\exp\sac_20260617_233812`

Baseline checkpoints:

- 20K comparable checkpoint: `SAC_19999.pt`
- 36.5K comparable checkpoint: `SAC_35999.pt`

## 20K Smoke Stop And Quality Check

Trigger when the command-only run reaches about 20,000 TensorBoard episodes and `SAC_19999.pt` exists.

Compare against baseline 20K TensorBoard reference:

- `success_rate_Normal` latest around `1.00`
- `success_rate_Complex` latest around `0.94`
- `success_rate_Extrem` latest around `0.73`
- `success_rate_dlp` latest around `0.78`
- `avg_reward` latest around `0.0648`, last-500 mean around `0.0803`
- `critic_loss` and `actor_loss` finite
- `step_num` last-500 mean around `67.2`

Early-warning stop conditions:

- missing checkpoint after expected save point
- actor/critic/alpha/action std contains non-finite values
- total reward or average reward collapses while scene success rates also collapse
- multiple scene success rates are severely below baseline, especially if Normal or Complex is also poor

This gate is now the stop point for this smoke run. It is not a final 36.5K equivalence judgment. Borderline results should be reported with caveats, not automatically treated as failure.

## Superseded 36.5K Equivalence Gate

The following 36.5K plan is superseded for the current run unless the user re-approves continuation. Keep it only as historical reference for future validation design.

Compare TensorBoard against the local 36.5K baseline:

- latest Normal around `0.99`
- latest Complex around `1.00`
- latest Extrem around `0.94`
- latest DLP around `0.90`
- last-500 `step_num` around `43.14`
- finite actor/critic losses
- no new reward/success decoupling pattern

Run matched 200-episode evaluation for the command-only `SAC_35999.pt` and compare with the local baseline `SAC_35999.pt` eval:

| Scene | Baseline 36.5K Eval |
|-------|---------------------|
| Normal | 1.000 |
| Complex | 0.985 |
| Extrem | 0.915 |
| DLP | 0.955 |
| Mean | 0.96375 |

Working equivalence rule:

- Mean success should be within 0.05 absolute of baseline, and
- no individual scene should be worse by more than 0.10 absolute, and
- Normal/Complex should remain at least 0.95, and
- Extrem/DLP should remain at least 0.85.

If the result is clearly below this bar, stop the long run and investigate. If it is clearly above this bar, continue to 100K. If it is borderline, report and ask before stopping.

## Superseded 100K Final Comparison

The 100K final comparison is superseded for the current run unless the user re-approves continuation.

At 100K, compare:

- command-only `SAC_99999.pt`
- command-only `SAC_best.pt`
- author checkpoints `src/model/ckpt/HOPE_SAC0.pt` and `src/model/ckpt/HOPE_SAC1.pt`

Use the same evaluation budget and command style for all checkpoints:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
D:\Github\HOPE\.venv\Scripts\python.exe .\evaluation\eval_mix_scene.py <checkpoint> --eval_episode 200 --visualize= --verbose=
```

Primary KPI:

- Normal, Complex, Extrem, and DLP success rates.

Secondary KPI:

- average reward
- average steps per success/failure pattern
- TensorBoard reward/loss/alpha/action-std stability
- throughput and resource profile

## Boundary

This run still belongs to Stage 3 original-HOPE validation. It does not introduce OGM code and does not modify original `src/train`, `src/env`, or `src/model` source.

Future speed research must preserve the HOPE paper's core design intent. In particular, do not remove or weaken action masks, do not flatten or replace curriculum/scene scheduling, do not change observation/action/reward semantics or environment dynamics, and do not replace the hybrid policy/path-planning design with a different algorithmic idea. Speed work should focus on training methodology and engineering architecture under opt-in experiment boundaries.

## Final 20K Smoke Result

Stopped at:

- Local stop time: `2026-06-19T13:41:04+08:00`
- TensorBoard episodes at stop snapshot: `20038`
- Environment steps: `1824904`
- Estimated SAC updates after warmup: `181466`
- Stop checkpoint: `D:\Github\HOPE\src\log\exp\sac_20260619_004316\SAC_19999.pt`
- Command-only eval dir: `D:\Github\HOPE\src\log\eval\20260619_134153`
- Baseline 20K eval dir: `D:\Github\HOPE\src\log\eval\20260619_135349`

Resource and speed profile:

| Metric | Value |
|--------|-------|
| Wall time to stop | `12.964 h` |
| Episodes/hour | `1545.71` |
| Env steps/second | `39.10` |
| Resource samples | `9229` |
| Avg process CPU | `37.76%` |
| Avg whole-GPU util | `29.42%` |
| Recent process CPU | `26.49%` |
| Recent whole-GPU util | `19.00%` |
| Peak process CPU | `49.02%` |
| Peak whole-GPU util | `100.00%` |

TensorBoard comparison against the 20K baseline reference:

| Signal | Command-only latest | Baseline 20K latest | Delta | Command recent mean | Baseline 20K last-500 |
|--------|---------------------|---------------------|-------|---------------------|-----------------------|
| `avg_reward` | `0.076828` | `0.064843` | `+0.011985` | `0.061052` | `0.080339` |
| `actor_loss` | `-0.471469` | `-0.269270` | `-0.202199` | `-0.543129` | `-0.526619` |
| `critic_loss` | `0.080817` | `0.067905` | `+0.012913` | `0.103880` | `0.137464` |
| `success_rate_Normal` | `1.000000` | `1.000000` | `0.000000` | `1.000000` | `1.000000` |
| `success_rate_Complex` | `0.940000` | `0.940000` | `0.000000` | `0.940000` | `0.964620` |
| `success_rate_Extrem` | `0.690000` | `0.730000` | `-0.040000` | `0.710800` | `0.767680` |
| `success_rate_dlp` | `0.820000` | `0.780000` | `+0.040000` | `0.792400` | `0.813200` |
| `step_num` | `52.000000` | `31.000000` | `+21.000000` | `77.220000` | `67.200000` |

The scene success-rate summaries still contain early unavailable-window NaNs for Complex, Extrem, and DLP in the TensorBoard event stream. This is the same schema behavior already seen in the baseline references. Reward, actor loss, critic loss, alpha, action std, and step count did not contain non-finite values.

Matched 200-episode eval against the baseline 20K checkpoint:

| Scene | Command-only `SAC_19999.pt` | Baseline `SAC_19999.pt` | Delta |
|-------|------------------------------|--------------------------|-------|
| Normal | `0.985` | `0.985` | `0.000` |
| Complex | `0.945` | `0.945` | `0.000` |
| Extrem | `0.655` | `0.655` | `0.000` |
| DLP | `0.960` | `0.960` | `0.000` |
| Mean | `0.88625` | `0.88625` | `0.00000` |

Comparison against the stopped 36.5K baseline, with the required caveat that this is only a 20K smoke/early-quality check and not a full 36.5K equivalence proof:

| Scene | Command-only 20K eval | 36.5K baseline eval | Delta |
|-------|-----------------------|---------------------|-------|
| Normal | `0.985` | `1.000` | `-0.015` |
| Complex | `0.945` | `0.985` | `-0.040` |
| Extrem | `0.655` | `0.915` | `-0.260` |
| DLP | `0.960` | `0.955` | `+0.005` |
| Mean | `0.88625` | `0.96375` | `-0.07750` |

## Quality Judgment

The command-only flags did not show a negative training-quality impact at the 20K smoke point:

- The command-only checkpoint and matched baseline 20K checkpoint produced identical 200-episode eval results.
- TensorBoard reward/loss signals stayed finite and comparable to the 20K baseline reference.
- Normal and Complex were healthy in both TensorBoard and external eval.
- DLP was strong in external eval and comparable in TensorBoard.
- Extrem remained the weakest early-warning scene. This weakness is shared by the matched 20K baseline eval and should be treated as a 20K maturity limitation, not evidence that the command-only flags harmed quality.

This smoke result supports moving into research-only training-architecture speed analysis. It does not prove that a command-only run would be equivalent to the 36.5K baseline or author 100K checkpoints.

## Early Warning Notes

Observed early-warning signals to keep for future runs:

- Extrem is the most sensitive early-quality signal. It can lag while Normal/Complex/DLP look acceptable.
- `step_num=200` timeout samples recur during early and mid training and should be interpreted together with scene success and reward, not alone.
- Reward/loss non-finite values remain a hard stop condition, but this run did not exhibit them.
- A 20K eval can still be materially weaker than 36.5K on Extrem, so 20K should remain an early-warning gate, not an equivalence gate.
