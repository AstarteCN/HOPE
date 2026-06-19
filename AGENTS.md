# Agent Development Rules For HOPE

This repository is now maintained under the user's fork for incremental research and development. Future agents must follow this file before making changes.

## Current Objective

The current development objective is to reproduce and validate the original HOPE project before introducing any RL-OGM-Parking changes.

Work must proceed in this order:

1. Configure an isolated project environment for this repository.
2. Run the original repository code with the provided checkpoints to confirm the environment works.
3. Retrain an agent locally using the original HOPE method, study local hardware utilization, and attempt to reproduce the original paper's reported behavior.
4. Only after the original HOPE baseline is understood and reproduced, study how to adapt the OGM paper's design.

Do not skip ahead to OGM implementation.

## Hard Boundaries

- Do not introduce OGM-specific source code until stages 1-3 are complete and the user explicitly approves moving to stage 4.
- Do not change the original HOPE training logic while validating checkpoint execution unless the change is required to make the unmodified project run in the isolated environment.
- Do not overwrite, delete, or regenerate the provided checkpoints in `src/model/ckpt/`.
- Do not commit training artifacts, TensorBoard logs, generated checkpoints, large datasets, or local environment folders unless the user explicitly asks.
- Do not install packages into the global Python environment. Use a project-local environment or a named environment dedicated to this repository.
- Do not push to the original upstream repository. The default push remote must remain the user's fork.
- Do not optimize or refactor the training pipeline before collecting a baseline resource-utilization profile.
- Do not treat very short reinforcement-learning runs as training-quality evidence. Smoke and optimization-validation runs for Stage 3 should cover at least 40,000 training episodes (`--train_episode 40000`), roughly 40% of the original author's 100,000-episode training scale, unless the user explicitly approves a shorter diagnostic run.
- Do not break the original author's training framework. During Stage 3, do not edit original HOPE source paths under `src/train/`, `src/env/`, or `src/model/` for performance work. Any acceleration experiment must be proposed as a separate, user-approved plan using external wrappers, monitoring tools, or clearly new opt-in experiment entry points.
- Do not rewrite the HOPE paper's core design intent when researching speed. Preserve action-mask behavior, curriculum and scene scheduling, observation/action/reward semantics, environment dynamics, and the original hybrid policy/path-planning framing. Speed research should focus on training methodology and engineering architecture, not on replacing the paper's algorithmic ideas.

## Environment Rules

This is primarily a Windows/PowerShell workspace.

Use a project-specific Python environment. Preferred options:

- A local virtual environment at `.venv/`.
- A dedicated Conda environment named for this project, such as `HOPE`.

Avoid relying on packages already installed globally on the machine. When installing dependencies, install into the selected project environment only.

Recommended setup flow:

```powershell
cd D:\Github\HOPE
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Install PyTorch explicitly for the local CUDA/CPU environment from the official PyTorch instructions. Record the exact PyTorch install command used in a docs note before running long training jobs.

When running project scripts, prefer:

```powershell
cd D:\Github\HOPE\src
..\.venv\Scripts\python.exe .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 10 --visualize False
```

If using Conda, record the environment name and `python --version`, `pip freeze`, and CUDA availability in project docs.

## Stage 1: Environment Configuration

Goal: create a reproducible local runtime that is isolated from the user's global machine state.

Required checks:

- Confirm Python version.
- Confirm dependencies install inside the project environment.
- Confirm PyTorch imports and report CUDA availability.
- Confirm `pygame`, `shapely`, `gym`, `torch`, and `cv2` import from the project environment.
- Record setup commands and important environment details in `docs/research/ogm_experiment_log.md` or a dedicated setup note under `docs/research/`.

Do not modify source files during this stage unless a compatibility issue blocks the original code from importing.

## Stage 2: Checkpoint Validation

Goal: prove the original HOPE code can run with author-provided checkpoints.

Use checkpoints under:

```text
src/model/ckpt/
```

Start with:

```text
src/model/ckpt/HOPE_SAC0.pt
```

Recommended smoke command:

```powershell
cd D:\Github\HOPE\src
..\.venv\Scripts\python.exe .\evaluation\eval_mix_scene.py .\model\ckpt\HOPE_SAC0.pt --eval_episode 10 --visualize False
```

Record:

- Command used.
- Checkpoint used.
- Whether the run completed.
- Success rate and any warnings/errors.
- Hardware and CUDA status.

Keep this stage focused on running the original code path.

## Stage 3: Original HOPE Local Retraining And Resource Study

Goal: train locally using the original HOPE method, understand whether the current code fully uses this machine's hardware, and compare behavior against the original paper and repository baseline.

Start with SAC because the repository includes SAC checkpoints and `train_HOPE_sac.py`.

Required progression:

1. Establish an unmodified baseline run from the original training entry point.
2. Profile CPU, GPU, memory, disk, wall-clock throughput, and TensorBoard curves during training.
3. Identify whether the run is underutilizing local hardware.
4. If hardware is underutilized, locate the training bottleneck before making changes.
5. Propose resource-utilization improvements that preserve training quality and the original HOPE framework.
6. Validate any optimization against the baseline with the same quality indicators and at least 40,000 training episodes.
7. Only after this evidence exists, run longer reproduction training.

Commands should use the isolated project Python executable. Example:

```powershell
cd D:\Github\HOPE\src
$env:SDL_VIDEODRIVER='dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'
..\.venv\Scripts\python.exe .\train\train_HOPE_sac.py --train_episode 40000 --eval_episode 200 --visualize False --verbose True
```

For each run, record:

- Command.
- Git commit or working-tree state.
- Environment details.
- Training episode count and evaluation episode count.
- Measured training episode count from TensorBoard `step_num` event count.
- Measured environment step count from the sum of TensorBoard `step_num` values as a secondary throughput and training-health metric.
- TensorBoard/log directory.
- Final checkpoint path.
- Normal, Complex, Extrem, and DLP evaluation results when available.
- Any deviations from the paper's setup.
- Hardware utilization observations: GPU utilization, GPU memory, CPU utilization, RAM, disk activity, and wall-clock throughput.
- TensorBoard observations during the run, not only final numbers.
- Unless a PID-level GPU monitor is added, GPU utilization fields from `nvidia-smi` are whole-device samples and should be described as coarse trends, not proof of per-process GPU attribution.

Generated logs and checkpoints should stay local unless the user asks to track a specific artifact.

### Stage 3 Resource-Utilization Rules

- First measure, then optimize. Do not guess the bottleneck.
- Treat Stage 3's 40K budget as training episodes, not environment interaction steps or replay transitions.
- Use at least 40,000 training episodes for Stage 3 smoke/validation runs and for validating any performance modification. A shorter run is allowed only for command syntax, monitoring, or crash diagnostics, and must not be used to judge training quality.
- Continue recording environment interaction steps from TensorBoard `step_num`; use that value for throughput, SAC-update estimates, and failure-mode interpretation, not as the budget gate.
- Preserve the original training behavior as the comparison baseline. Stage 3 performance work must use opt-in wrappers, monitoring scripts, or clearly new experiment entry points; do not modify original training defaults or existing HOPE source paths without explicit user approval for a separate research change.
- Change one performance variable at a time so the effect can be attributed.
- Keep reward shaping, environment dynamics, action masking, RS fallback behavior, and model architecture unchanged unless the user explicitly approves a separate research change.
- Current command-only long run scope was narrowed by the user on 2026-06-19: stop around 20,000 episodes as a smoke-quality check because speed gains were limited. Do not continue that run to 36.5K or 100K unless the user explicitly re-approves it.

### Stage 3 TensorBoard Monitoring

Monitor these scalars during training:

- `total_reward`
- `avg_reward`
- `actor_loss`
- `critic_loss`
- `action_std0`
- `action_std1`
- `alpha`
- `success_rate_Normal`
- `success_rate_Complex`
- `success_rate_Extrem`
- `success_rate_dlp`
- `step_num`

Summarize early warning patterns after each meaningful run:

- Reward does not improve after enough training episodes.
- Success rate stays low or improves only in easy scenes.
- `step_num` remains near timeout, suggesting the agent is not learning efficient parking.
- Actor or critic loss becomes NaN, explodes, or oscillates without reward improvement.
- `alpha` or action standard deviations collapse too early, reducing exploration.
- DLP performance diverges from synthetic scenes.
- GPU utilization remains low while CPU usage is high, suggesting environment or preprocessing bottlenecks.
- GPU memory is available but batch/update work does not keep the GPU busy.

## Stage 4: OGM Adaptation Research

Do not start this stage until the user explicitly approves it after stages 1-3.

When approved, use the prior research notes and plan:

- `docs/research/2026-06-17-rl-ogm-paper-code-research.md`
- `docs/superpowers/plans/2026-06-17-rl-ogm-parking-integration.md`

The first OGM step should be design/research only unless the user asks for implementation. The main open question is how to introduce OGM observations while preserving a fair HOPE baseline.

## Git And Artifact Hygiene

- Keep `origin` pointed at the user's fork.
- Keep `upstream` as read-only reference to the original author repository.
- Before any commit, run `git status -sb` and inspect changed files.
- Stage only intentional files.
- Do not stage `.venv/`, Conda environments, training logs, TensorBoard runs, generated checkpoints, or raw local experiment dumps.
- Prefer documentation commits for setup and experiment records.

## Documentation Expectations

Each completed stage should leave a short written record under `docs/research/`.

Use concise entries with:

- Date.
- Exact commands.
- Environment.
- Result.
- Next decision.

The repository should remain understandable to a future agent that has not seen the prior conversation.
