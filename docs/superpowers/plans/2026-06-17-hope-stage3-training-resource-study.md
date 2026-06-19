# HOPE Stage 3 Training Resource Study Implementation Plan

> **For agentic workers:** Use `superpowers:subagent-driven-development` or `superpowers:executing-plans` to execute this plan task by task. Keep `planning-with-files` state synchronized after each major action.

**Goal:** Run original HOPE SAC retraining locally at a meaningful 40,000-training-episode scale, monitor hardware and TensorBoard signals, identify bottlenecks, and gate any optimization behind evidence.

**Architecture:** Keep `src/train/train_HOPE_sac.py` as the unmodified baseline entry point. Use external Stage 3 tooling under `tools/stage3/` for Windows resource sampling and TensorBoard scalar summarization. Write research notes under `docs/research/`. Stage 3 performance/resource work must not edit original HOPE source paths; any source-changing optimization requires a separate user-approved plan.

**Tech Stack:** Windows PowerShell, project-local `.venv`, Python 3.13, PyTorch CUDA, TensorBoard event files, `nvidia-smi`, original HOPE SAC training code.

---

## Scope Check

This plan covers one subsystem: Stage 3 baseline retraining and resource study for the original HOPE SAC code path. OGM adaptation, reward redesign, environment dynamics changes, and model architecture changes remain out of scope.

## Budget Definition

- Stage 3's 40K budget means 40,000 training episodes, aligned with the original repository's `--train_episode 100000` scale.
- A meaningful smoke or optimization-validation run must use `--train_episode 40000` unless the user explicitly approves a shorter diagnostic.
- TensorBoard `step_num` values are environment-step counts for each episode. Their sum is `env_step_count`, a secondary throughput and health metric.
- `env_step_count` is useful for SAC-update estimates, timeout detection, and throughput, but it is not the training-budget gate.
- Any shorter run, including the earlier 300-episode diagnostic, is command/resource-pipeline evidence only and must not be used to judge training quality.
- Resource monitor GPU columns are whole-GPU state from `nvidia-smi`, not PID-level attribution. Treat them as coarse utilization trends unless a later user-approved monitor adds PID-level GPU sampling.

## File Structure

- Existing: `tools/stage3/monitor_stage3_resources.ps1`
  - Samples one training process plus system GPU state to CSV without importing or changing HOPE Python source code.
- Existing: `tools/stage3/tensorboard_stage3_summary.py`
  - Reads TensorBoard event files, counts training episodes from `step_num` events, sums `step_num` into `env_step_count`, and emits JSON/Markdown summaries plus early-warning flags.
- Create during execution: `docs/research/2026-06-17-stage3-diagnostic-dry-run.md`
  - Records the interrupted 300-episode run as diagnostic only.
- Create during execution: `docs/research/2026-06-17-stage3-baseline-resource-profile.md`
  - Records the 40,000-episode baseline command, budget result, resource utilization, TensorBoard summary, and quality warnings.
- Create during execution: `docs/research/2026-06-17-stage3-bottleneck-analysis.md`
  - Records the bottleneck decision and whether a separate optimization plan is justified.
- Modify during execution: `AGENTS.md`, `task_plan.md`, `findings.md`, `progress.md`
  - Keep persistent planning and future-agent boundaries synchronized.
- Do not modify for Stage 3 performance/resource work: `src/train/train_HOPE_sac.py`, `src/model/**`, `src/env/**`
  - Baseline evidence must come from the original training path, and optimization proposals must use wrappers or new opt-in entry points unless separately approved by the user.

## Invariants

- Run all Python commands through `D:\Github\HOPE\.venv\Scripts\python.exe`.
- Set `$env:SDL_VIDEODRIVER='dummy'` before headless training.
- Set `$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD='1'` for trusted author checkpoints or image encoder loading under modern PyTorch.
- Keep TensorBoard open at `http://127.0.0.1:6006` for visual monitoring.
- Keep generated logs, TensorBoard event files, checkpoints, and resource CSV files local and untracked unless the user asks otherwise.
- Do not edit original HOPE source files during Stage 3 performance/resource work unless the user explicitly approves a separate source-change plan.

## Task 1: Verify External Tooling

**Files:**
- Read: `tools/stage3/monitor_stage3_resources.ps1`
- Read: `tools/stage3/tensorboard_stage3_summary.py`

### Step 1: Verify the resource monitor samples CPU and GPU fields

Run:

```powershell
cd D:\Github\HOPE
$samplePath = "D:\Github\HOPE\tmp\stage3_resource_monitor_check.csv"
$monitor = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "D:\Github\HOPE\tools\stage3\monitor_stage3_resources.ps1", "-ProcessId", $PID, "-OutputPath", $samplePath, "-IntervalSeconds", "1", "-MaxSamples", "3") -WindowStyle Hidden -PassThru
Wait-Process -Id $monitor.Id
Import-Csv -LiteralPath $samplePath | Format-Table
```

Expected: the CSV exists, contains three samples, includes CPU and memory columns, and includes GPU columns when `nvidia-smi` is available.

### Step 2: Verify the TensorBoard summary script with episode semantics

Run:

```powershell
cd D:\Github\HOPE
Remove-Item -LiteralPath .\tmp\stage3_tb_test -Recurse -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\tmp\stage3_tb_summary.json -Force -ErrorAction SilentlyContinue
Remove-Item -LiteralPath .\tmp\stage3_tb_summary.md -Force -ErrorAction SilentlyContinue
.\.venv\Scripts\python.exe -c "from torch.utils.tensorboard import SummaryWriter; from pathlib import Path; p=Path('tmp/stage3_tb_test'); p.mkdir(parents=True, exist_ok=True); w=SummaryWriter(str(p)); [w.add_scalar('step_num', v, i) for i, v in enumerate([200, 150])]; [w.add_scalar('avg_reward', v, i) for i, v in enumerate([-10.0, -8.0])]; w.close(); print(p.resolve())"
.\.venv\Scripts\python.exe .\tools\stage3\tensorboard_stage3_summary.py .\tmp\stage3_tb_test --json .\tmp\stage3_tb_summary.json --markdown .\tmp\stage3_tb_summary.md --min-episodes 40000
.\.venv\Scripts\python.exe -c "import json; data=json.load(open('tmp/stage3_tb_summary.json', encoding='utf-8')); print(data['episode_count']); print(data['training_budget_met']); print(data['env_step_count'])"
```

Expected:

```text
2
False
350
```

### Step 3: Verify source remains unmodified

Run:

```powershell
cd D:\Github\HOPE
git diff -- src
```

Expected: no output.

## Task 2: Record The 300-Episode Diagnostic Dry Run

**Files:**
- Create: `docs/research/2026-06-17-stage3-diagnostic-dry-run.md`
- Modify: `findings.md`
- Modify: `progress.md`

Record the earlier run as diagnostic only:

- Command used `--train_episode 300 --eval_episode 200`.
- TensorBoard run directory was `D:\Github\HOPE\src\log\exp\sac_20260617_230340`.
- The run reached 300 training episodes and about 35,689 summed environment steps before it was stopped.
- One partial DLP evaluation file showed success rate `0.44`; because the run was short and final evaluation was interrupted, it is not a baseline quality result.
- A launcher PID versus child Python PID mismatch was found during monitoring; future runs should monitor the child Python PID when present.
- No `src/` source diff should be present.

## Task 3: Start The 40,000-Episode Original SAC Baseline

**Files:**
- Runtime output: `src/log/exp/stage3_baseline_40k_<timestamp>.stdout.log`
- Runtime output: `src/log/exp/stage3_baseline_40k_<timestamp>.stderr.log`
- Runtime output: `src/log/exp/stage3_resource_40k_<timestamp>.csv`
- Runtime output: `src/log/exp/stage3_baseline_40k_<timestamp>.meta.json`

### Step 1: Confirm TensorBoard is available

TensorBoard should be open in the Codex side browser at:

```text
http://127.0.0.1:6006
```

If no TensorBoard process is running, start it:

```powershell
cd D:\Github\HOPE
Start-Process -FilePath "D:\Github\HOPE\.venv\Scripts\tensorboard.exe" -ArgumentList @("--logdir", "D:\Github\HOPE\src\log\exp", "--port", "6006", "--host", "127.0.0.1") -WindowStyle Hidden
```

### Step 2: Start the unmodified training entry point

Run:

```powershell
cd D:\Github\HOPE
git diff -- src

$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$logRoot = "D:\Github\HOPE\src\log\exp"
[System.IO.Directory]::CreateDirectory($logRoot) | Out-Null
$trainOut = Join-Path $logRoot "stage3_baseline_40k_$stamp.stdout.log"
$trainErr = Join-Path $logRoot "stage3_baseline_40k_$stamp.stderr.log"
$resourceCsv = Join-Path $logRoot "stage3_resource_40k_$stamp.csv"
$metaPath = Join-Path $logRoot "stage3_baseline_40k_$stamp.meta.json"
$env:SDL_VIDEODRIVER = 'dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$launcher = Start-Process -FilePath "D:\Github\HOPE\.venv\Scripts\python.exe" -ArgumentList @(".\train\train_HOPE_sac.py", "--train_episode", "40000", "--eval_episode", "200", "--visualize", "False", "--verbose", "True") -WorkingDirectory "D:\Github\HOPE\src" -RedirectStandardOutput $trainOut -RedirectStandardError $trainErr -WindowStyle Hidden -PassThru
Start-Sleep -Seconds 8
if ($launcher.HasExited) {
    $stderrTail = if (Test-Path -LiteralPath $trainErr) { Get-Content -LiteralPath $trainErr -Tail 80 } else { @() }
    throw "Training exited before monitoring could start. ExitCode=$($launcher.ExitCode). StderrTail=$($stderrTail -join [Environment]::NewLine)"
}
$child = Get-CimInstance Win32_Process | Where-Object { $_.ParentProcessId -eq $launcher.Id -and $_.CommandLine -like '*train_HOPE_sac.py*' } | Select-Object -First 1
$monitorPid = if ($child) { [int]$child.ProcessId } else { [int]$launcher.Id }
$monitor = Start-Process -FilePath "powershell" -ArgumentList @("-NoProfile", "-ExecutionPolicy", "Bypass", "-File", "D:\Github\HOPE\tools\stage3\monitor_stage3_resources.ps1", "-ProcessId", "$monitorPid", "-OutputPath", $resourceCsv, "-IntervalSeconds", "5") -WindowStyle Hidden -PassThru
$runDir = Get-ChildItem -LiteralPath "D:\Github\HOPE\src\log\exp" -Directory |
    Where-Object { $_.Name -like 'sac_*' -and $_.CreationTime -ge $launcher.StartTime.AddSeconds(-5) } |
    Sort-Object CreationTime -Descending |
    Select-Object -First 1
if (-not $runDir) {
    $stdoutTail = if (Test-Path -LiteralPath $trainOut) { Get-Content -LiteralPath $trainOut -Tail 80 } else { @() }
    $printedRun = $stdoutTail | Select-String -Pattern "tensorboard --log-dir (?<path>\S+)" | Select-Object -Last 1
    if ($printedRun) {
        $candidate = $printedRun.Matches[0].Groups['path'].Value
        $runDirPath = if ([System.IO.Path]::IsPathRooted($candidate)) { $candidate } else { Join-Path "D:\Github\HOPE\src" $candidate }
        if (Test-Path -LiteralPath $runDirPath) {
            $runDir = Get-Item -LiteralPath $runDirPath
        }
    }
}
if (-not $runDir) {
    throw "Could not lock the TensorBoard run directory for this training process. Do not summarize by newest sac_* directory."
}
[pscustomobject]@{
    stamp = $stamp
    train_episode = 40000
    eval_episode = 200
    launcher_pid = $launcher.Id
    monitored_pid = $monitorPid
    monitor_pid = $monitor.Id
    stdout = $trainOut
    stderr = $trainErr
    resource_csv = $resourceCsv
    run_dir = $runDir.FullName
    tensorboard = "http://127.0.0.1:6006"
} | ConvertTo-Json | Set-Content -LiteralPath $metaPath -Encoding UTF8
Get-Content -LiteralPath $metaPath
```

Expected: PowerShell prints metadata with launcher, monitored process, monitor PID, output paths, and the locked `run_dir`. `git diff -- src` must print no diff before the run. If training exits within the initial sleep window, the command throws with exit code and stderr tail instead of writing misleading monitoring metadata.

### Step 3: Monitor while training runs

Use these checks periodically:

```powershell
cd D:\Github\HOPE
$meta = Get-ChildItem -LiteralPath .\src\log\exp -Filter "stage3_baseline_40k_*.meta.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$m = Get-Content -LiteralPath $meta.FullName | ConvertFrom-Json
Get-Process -Id $m.monitored_pid -ErrorAction SilentlyContinue
Import-Csv -LiteralPath $m.resource_csv | Select-Object -Last 5 | Format-Table
$m.run_dir
Get-ChildItem -LiteralPath $m.run_dir | Select-Object Name, LastWriteTime, Length
```

Expected: the training process stays alive, the resource CSV grows, and TensorBoard shows the locked `run_dir` from metadata.

## Task 4: Export TensorBoard Signals During And After The 40K Run

**Files:**
- Read: latest `src/log/exp/sac_<timestamp>/events.out.tfevents*`
- Create: `docs/research/stage3_baseline_40k_tensorboard_<timestamp>.json`
- Create: `docs/research/stage3_baseline_40k_tensorboard_<timestamp>.md`

Run while the baseline is active and again after it exits:

```powershell
cd D:\Github\HOPE
$meta = Get-ChildItem -LiteralPath .\src\log\exp -Filter "stage3_baseline_40k_*.meta.json" | Sort-Object LastWriteTime -Descending | Select-Object -First 1
$m = Get-Content -LiteralPath $meta.FullName | ConvertFrom-Json
$runDir = $m.run_dir
$stamp = Get-Date -Format "yyyyMMdd_HHmmss"
$summaryJson = "D:\Github\HOPE\docs\research\stage3_baseline_40k_tensorboard_$stamp.json"
$summaryMd = "D:\Github\HOPE\docs\research\stage3_baseline_40k_tensorboard_$stamp.md"
.\.venv\Scripts\python.exe .\tools\stage3\tensorboard_stage3_summary.py $runDir --json $summaryJson --markdown $summaryMd --min-episodes 40000
Get-Content -LiteralPath $summaryMd
```

Expected during training: `training_budget_met` is usually `False` until 40,000 `step_num` scalar events exist.

Expected after training: `episode_count` is at least `40000`, `training_budget_met` is `True`, and `env_step_count` is recorded for throughput and training-health analysis.

## Task 5: Write Baseline Resource Profile

**Files:**
- Create: `docs/research/2026-06-17-stage3-baseline-resource-profile.md`

Create the report only after the 40,000-episode baseline has completed or after a user-approved stop. Include:

- Exact command.
- Git working-tree state.
- TensorBoard run directory.
- Stdout, stderr, metadata, and resource CSV paths.
- `episode_count`, required minimum episodes, `training_budget_met`, and `env_step_count`.
- Average and peak CPU utilization.
- Average and peak whole-GPU utilization from `nvidia-smi`.
- Peak GPU memory usage.
- Note that GPU columns are whole-device samples, not PID-level GPU attribution.
- Episodes per second and environment steps per second when calculable.
- TensorBoard warnings and manual observations.
- Final evaluation results when available.
- Explicit note if the run was stopped early.

## Task 6: Write Bottleneck Analysis And Optimization Gate

**Files:**
- Create: `docs/research/2026-06-17-stage3-bottleneck-analysis.md`
- Read: `docs/research/2026-06-17-stage3-baseline-resource-profile.md`
- Read: `src/train/train_HOPE_sac.py`
- Read: `src/env/car_parking_base.py`
- Read: `src/model/agent/sac_agent.py`
- Read: `src/model/replay_memory.py`

Use measured evidence to choose one primary classification:

```text
CPU/environment-bound:
  AvgGpuPercent < 30 and MaxGpuMemoryUsedMb is far below GPU capacity, while CPU usage or one Python process dominates wall time.

GPU-update-bound:
  AvgGpuPercent >= 70 during post-warmup training and TensorBoard losses update regularly.

Logging/evaluation-bound:
  Training slows mainly near figure generation, checkpoint saves, or final evaluation directories, with resource CSV spikes aligned to those periods.

Training-quality-bound:
  TensorBoard warnings show reward plateau, NaN/Inf losses, alpha/action_std collapse, or success isolated to easy scenes.

Inconclusive:
  Resource sample count is too low, episode budget is not met, or stdout/stderr indicates a runtime anomaly.
```

Do not propose edits to existing original HOPE source paths. If evidence suggests acceleration is possible, propose a separate optimization plan that preserves the baseline and uses wrappers or new opt-in entry points unless the user explicitly approves changing original source files.

## Task 7: Sync Planning Files And Prepare Handoff

**Files:**
- Modify: `task_plan.md`
- Modify: `findings.md`
- Modify: `progress.md`
- Modify: `AGENTS.md` if new boundaries are learned

Mark Phase 2 complete only if:

```text
1. Original SAC 40,000-episode baseline command completed.
2. TensorBoard summary reports training_budget_met true.
3. Resource CSV exists and contains samples.
4. Baseline resource profile document exists.
```

If the 40K run is still active, keep Phase 2 in progress and record the latest metadata paths, episode count, resource observations, and TensorBoard warnings.

## Final Verification

Run before claiming any Stage 3 milestone is complete:

```powershell
cd D:\Github\HOPE
$reservedMarkers = @('TB' + 'D', 'TO' + 'DO', 'place' + 'holder', 'fill' + ' in', 'implement' + ' later')
foreach ($marker in $reservedMarkers) {
    rg -n $marker AGENTS.md task_plan.md findings.md progress.md docs\research docs\superpowers\plans\2026-06-17-hope-stage3-training-resource-study.md
}
$staleBudgetTerms = @(
    'environment interaction ' + 'nodes',
    '40,000 ' + 'nodes',
    '40,000-' + 'node',
    '--min-' + 'nodes',
    'node_' + 'budget_met',
    'min_' + 'nodes',
    'node_' + 'count'
)
foreach ($term in $staleBudgetTerms) {
    rg -n $term AGENTS.md task_plan.md findings.md progress.md docs\superpowers\specs\2026-06-17-hope-stage3-training-resource-design.md docs\superpowers\plans\2026-06-17-hope-stage3-training-resource-study.md
}
git diff -- src
git status -sb
```

Expected:

```text
No reserved incomplete-marker matches.
No stale node-budget matches in active Stage 3 docs.
No source diff under src.
Git status shows only intentional docs/tooling changes plus ignored runtime artifacts.
```

## Self-Review

- Spec coverage: resource utilization, bottleneck analysis, quality-preserving acceleration gate, original-framework preservation, 40,000-episode rule, and TensorBoard early-warning monitoring are each mapped to a task.
- Budget consistency: `episode_count` and `training_budget_met` are the budget fields; `env_step_count` is observational.
- Boundary check: this plan creates external monitoring and summary tools only; it does not edit original HOPE training, environment, model, reward, or checkpoint files.
