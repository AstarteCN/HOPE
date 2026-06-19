# 2026-06-18 Stage 3 Safe Training Acceleration Research

## Question

How can HOPE SAC training be made faster without damaging training quality or breaking the original author's training structure?

This note combines the current original-HOPE Stage 3 baseline, the training code path, and prior on-disk experiment summaries. It does not approve OGM implementation or source changes under `src/train/`, `src/env/`, or `src/model/`.

## Current Baseline Evidence

Update after user-directed stop:

- The original SAC run was stopped at `36502` episodes on `2026-06-18 23:29 +08:00` and is now the current local acceleration baseline.
- Final TensorBoard health remained strong: finite actor/critic losses, positive reward trends, latest moving success rates Normal `0.99`, Complex `1.00`, Extrem `0.94`, DLP `0.90`, and latest `step_num=36`.
- Resource profile over `23.84` hours: average normalized process CPU `32.38%`, average whole-GPU utilization `25.76%`, peak whole-GPU memory `3740 MB`.
- The evaluated checkpoint is `SAC_35999.pt`; because the run was stopped externally, no `SAC_36502.pt` checkpoint exists.
- 200-episode final eval of `SAC_35999.pt`: Normal `1.00`, Complex `0.985`, Extrem `0.915`, DLP `0.955`.

Active original SAC run:

- Run directory: `D:\Github\HOPE\src\log\exp\sac_20260617_233812`
- Command family: original `src/train/train_HOPE_sac.py` with `--train_episode 40000 --eval_episode 200`
- Latest live TensorBoard summary: about `28525` training episodes, `2326377` summed environment steps, `training_budget_met=false`
- Throughput window: about `0.4669` episodes/s and `38.08` environment steps/s over the resource CSV window
- Latest watched signals: finite actor/critic losses, positive reward trend, `step_num` trend down from about `118.98` to `66.10`
- Latest success rates: Normal `0.99`, Complex `0.98`, Extrem `0.91`, DLP `0.75`
- Current best record: `epoch: 26209, success rate: 0.99 0.98 0.9 0.94`

Resource samples through `2026-06-18T16:32:21+08:00`:

- Samples: `12012`
- Average normalized process CPU: `37.14%`; peak `49.09%`
- Average whole-GPU utilization: `28.0%`; peak `97%`
- Peak whole-GPU memory used: about `3740 MB`
- GPU metrics are whole-device `nvidia-smi` samples, not PID-level attribution.

Interpretation: the original baseline is learning well before 40K episodes are complete, while the GPU is not consistently saturated. Chasing GPU occupancy is not the right objective; preserving this learning curve is.

## Training Design Findings

The SAC training loop is sequential and environment-first:

- `train_HOPE_sac.py` creates one `CarParking` environment and one `SAC` agent.
- Warmup uses random actions until `total_step_num > parking_agent.configs.memory_size`.
- `SACConfig.memory_size` is `10240`.
- After warmup, the loop performs exactly one `parking_agent.update()` every 10 environment steps.
- Actor/critic losses are logged every 200 environment steps.
- Per episode, TensorBoard logs reward, average reward, action stds, alpha, four scene success rates, and `step_num`.

Important configuration trap:

- `configs.py` currently says `BATCH_SIZE = 8192`.
- But `SACConfig` sets `self.batch_size = 32` after inheriting `ConfigBase`, and the training config does not pass `batch_size`.
- Therefore the current original SAC baseline is effectively using batch size `32`, not `8192`.
- Any old run that truly used larger batches must have changed SAC config or passed a different field, not merely changed the global `BATCH_SIZE`.

The per-step environment path is expensive and mostly CPU-side:

- `env.step()` advances the vehicle for `NUM_STEP = 10` physics substeps.
- It checks arrival, collision, outbound, timeout, and reward.
- It calls `render()` every step, even when the training is headless.
- `render()` draws with Pygame, processes image observations, computes lidar, computes the action mask, and calls `pygame.display.update()`.
- Near the target, the environment attempts Reeds-Shepp path generation and trajectory validation.
- Lidar and trajectory validity are already partly vectorized, but they still allocate arrays and use geometry operations per step.

The model update path is comparatively small:

- Actor and critic use multimodal embeddings plus an attention block.
- The pretrained image encoder is loaded and frozen, but still runs forward passes during training updates.
- Replay memory uses Python `deque` objects and samples by random index, converting sampled lists to arrays each update.
- The update runs one mini epoch with one sampled batch.

## Prior Run Evidence

The old run `D:\Github\HOPE\src\log\exp\sac_20260418_165251` is not a clean original-HOPE baseline. Its copied config contains BEV/ULS fields, action-mask/action-chunk profiles, larger visual encoder layers, and postprocess diagnostics:

- `USE_LIDAR = False`
- `C_CONV = [16, 32, 64]`
- `SIZE_FC = [512, 256]`
- BEV/ULS observation fields and additional action mask/chunk settings

Its own summaries report weak training quality at 40K episodes:

- `best_metric_value = 0.2625`
- selected sample eval `0.49`
- selected mean eval `0.215`
- selected DLP mean `0.2`
- selected Extrem mean `0.06`
- sample-mean gap `0.275`
- repeated reward/success decoupling and critic spikes
- deterministic control or generalization was assessed as the main bottleneck

Interpretation: that historical run mixed capacity, observation, action-mask/chunk, and postprocess changes. It is strong evidence against treating larger batch/layer count as a safe speed knob. It is not evidence that the original HOPE baseline is slow because the model is too small.

## Why Large Batch Or More Layers Is Risky Here

Large batch is not a neutral acceleration in this codebase.

- Update cadence stays fixed at one update per 10 environment steps unless the training loop changes.
- With `memory_size = 10240`, very large batches repeatedly sample a large fraction of the replay buffer.
- That reduces SGD noise, increases correlation with early random-policy data, and changes SAC's effective learning dynamics.
- Larger batches usually need coordinated retuning of learning rate, target update, entropy temperature behavior, and update ratio. That is a research change, not a safe speed change.
- If the environment is the wall-clock bottleneck, larger batches can use more GPU while not producing better environment samples per hour.

More layers are also not a safe acceleration.

- They increase compute per update and change the critic/actor function class.
- In SAC, critic stability and target-network lag matter. Increasing capacity without retuning can worsen value drift.
- The old run's critic spikes and reward/success decoupling match this risk.
- A larger model can appear faster only if fewer environment samples are needed, which must be proven with 40K-episode validation.

## Safe Acceleration Ladder

### Level 0: Finish The Original 40K Baseline

Do not stop or reinterpret the current run as complete until:

- `episode_count >= 40000`
- `training_budget_met = true`
- the resource CSV and TensorBoard summary are exported
- final evaluation results are recorded

This baseline is the quality anchor. Every acceleration experiment should compare against it.

### Level 1: Remove Logging And CLI Overhead Without Changing Training Semantics

The current script defines `--verbose` and `--visualize` as `type=bool`, so passing `False` parses as `True`. Verified locally:

```text
bool("False") == True
bool("") == False
```

Practical effect: `--verbose False` does not disable verbose mode. The current run is still updating `reward.png` about every 20 episodes and printing summaries about every 10 episodes.

Safest future command-side change:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
..\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 40000 --eval_episode 200 --visualize "" --verbose ""
```

Expected semantic impact:

- `--verbose ""` should only remove prints and repeated `reward.png` writes.
- `--visualize ""` switches the env constructor to `render_mode='rgb_array'`, which should keep hidden rendering. Before adopting it for quality runs, verify observation/reward/status parity on scripted actions.

If only one flag is adopted first, use `--verbose ""`. It is lower risk than changing render mode.

### Level 2: Add Measurement-Only Profiling

Before code changes, add an external profiling wrapper or diagnostic entry point that times:

- `env.reset`
- `env.step`
- `parking_agent.get_action`
- `parking_agent.push_memory`
- `parking_agent.update`
- TensorBoard scalar logging
- `reward.png` generation
- checkpoint save
- final evaluation

Short diagnostic runs are acceptable for timing only. They must not be used as training-quality evidence.

Target question: how much wall time is spent in environment observation/render/action-mask/RS versus SAC update? If environment time dominates, GPU-oriented changes are the wrong first lever.

### Level 3: Semantics-Preserving Environment Micro-Optimizations

Only after profiling identifies a hot path, propose a separate opt-in experiment entry point. Candidate changes need parity tests against the original environment on scripted action sequences.

Promising candidates:

- Skip `pygame.display.update()` when no human display is needed, if observation pixels are identical.
- Precompute lidar ray angles and reusable arrays in `LidarSimlator`.
- Replace repeated generic `deepcopy` of observations with explicit array copies where aliasing safety is preserved.
- Cache or reuse temporary arrays in action-mask and lidar calculations.
- Time Reeds-Shepp probing and add diagnostics before changing any planner gate.

Required parity gate:

- same seeded reset cases
- same scripted actions
- compare reward, status, `target`, `lidar`, `action_mask`, and image observation tolerances
- run enough cases across Normal, Complex, Extrem, and DLP

### Level 4: Training-Algorithm Experiments

These are not "safe speedups"; they are research changes and must be validated with the 40K rule.

Possible but high-risk:

- change batch size gradually, such as 32 to 64, with no other variable changed
- change update cadence or add more updates per collected environment step
- vectorize environment collection
- precompute frozen image embeddings in replay memory
- use mixed precision or `torch.compile` for update-only sections

Do not combine these. Each needs a matched 40K validation, TensorBoard comparison, and final evaluation comparison.

## Command-Only 1000-Episode Smoke Result

On 2026-06-18 to 2026-06-19, a user-approved parameter-only smoke used the original `train_HOPE_sac.py` with:

```powershell
--train_episode 1000 --eval_episode 1 --visualize= --verbose=
```

This run completed 1,000 training episodes and 127,470 environment steps. Based on TensorBoard wall time from launch to the last training scalar, it took 2,261.16 seconds, or 37.69 minutes.

Key measured values:

- Episodes/hour: `1592.10`
- Environment steps/second: `56.37`
- Average normalized CPU: `44.74%`
- Average whole-GPU utilization: `36.75%`
- Peak whole-GPU memory: `3519 MB`

Fair comparison against the original baseline's first 1,000 episodes:

- Original first 1K: `459.58` episodes/hour and `14.99` environment steps/second.
- Command-only 1K: `1592.10` episodes/hour and `56.37` environment steps/second.
- Ratio: `3.46x` episodes/hour and `3.76x` environment steps/second.

This supports using the command-only flags as the next low-risk long-run candidate. It does not prove long-run quality equivalence; the run is intentionally too short for that.

Detailed note: `docs/research/2026-06-19-stage3-command-only-1000-smoke.md`.

## Updated Recommended Next Decision

1. Treat the stopped `36502`-episode run as the current local baseline because the user explicitly requested this stop point.
2. Treat the 1,000-episode command-only run as a successful speed/resource smoke, not a training-quality validation.
3. For the next long run, use command-only acceleration: `--verbose ""` and `--visualize ""`, or the equivalent `--verbose= --visualize=`, with the same original training script and a matched evaluation budget.
4. Compare against the baseline on throughput, TensorBoard curves, final scene success rates, and failure-mode indicators.
5. Add a measurement-only profiling wrapper before source-level changes.
6. Only after a measured hot path is confirmed, propose one opt-in source-level experiment at a time.

Do not increase layer count or jump batch size again as a speed strategy. In this repository, those changes are more likely to change training quality than improve time-to-good-policy.

## Final Research Position

The safest acceleration path is not to make the GPU busier. It is to remove non-learning overhead first, then measure the environment hot path, then preserve exact observation/reward/status parity for any source-level optimization.

Recommended approach order:

1. Command-only overhead reduction: `--verbose ""`, then `--visualize ""` after parity checks.
2. External profiling and parity harnesses.
3. Opt-in environment micro-optimizations, especially display update suppression, reusable lidar/action-mask arrays, and RS-probe diagnostics.
4. Algorithmic speed experiments only after the above, each with its own matched long validation.
