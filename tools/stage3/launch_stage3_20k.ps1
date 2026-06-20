param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [Parameter(Mandatory=$true)][string]$CandidateType,
    [Parameter(Mandatory=$true)][string]$ChangedKnobsJson,
    [int]$TrainEpisode = 20000,
    [int]$EvalEpisode = 200,
    [switch]$FastActionMask
)

$ErrorActionPreference = 'Stop'

function Resolve-RequiredPath {
    param(
        [Parameter(Mandatory=$true)][string]$Path,
        [Parameter(Mandatory=$true)][string]$Description
    )

    if (-not (Test-Path -LiteralPath $Path)) {
        throw "${Description} not found: $Path"
    }
    return (Resolve-Path -LiteralPath $Path).Path
}

function ConvertTo-SafeFileName {
    param([Parameter(Mandatory=$true)][string]$Value)

    $safe = $Value -replace '[^A-Za-z0-9_.-]', '_'
    if ([string]::IsNullOrWhiteSpace($safe)) {
        return 'stage3_run'
    }
    return $safe
}

function Quote-ProcessArgument {
    param([Parameter(Mandatory=$true)][string]$Value)

    $escaped = $Value.Replace('"', '\"')
    if ($escaped -match '[\s"]') {
        return '"' + $escaped + '"'
    }
    return $escaped
}

function Get-CurrentPowerShellExecutable {
    $currentProcess = Get-Process -Id $PID
    if ($currentProcess.Path -and (Test-Path -LiteralPath $currentProcess.Path)) {
        return $currentProcess.Path
    }

    $pwsh = Get-Command pwsh -ErrorAction SilentlyContinue
    if ($null -ne $pwsh) {
        return $pwsh.Source
    }

    $powershell = Get-Command powershell -ErrorAction SilentlyContinue
    if ($null -ne $powershell) {
        return $powershell.Source
    }

    throw 'Could not locate a PowerShell executable for the resource monitor.'
}

function Wait-Stage3RunDirectory {
    param(
        [Parameter(Mandatory=$true)][string]$ExpDir,
        [Parameter(Mandatory=$true)][hashtable]$BeforeRunDirs,
        [Parameter(Mandatory=$true)][datetime]$LaunchStartUtc,
        [Parameter(Mandatory=$true)][int]$WorkloadPid,
        [int]$TimeoutSeconds = 300
    )

    $deadline = (Get-Date).AddSeconds($TimeoutSeconds)
    while ((Get-Date) -lt $deadline) {
        $candidates = @(
            Get-ChildItem -LiteralPath $ExpDir -Directory -Filter 'sac_*' -ErrorAction SilentlyContinue |
                Where-Object {
                    -not $BeforeRunDirs.ContainsKey($_.FullName) -and
                    $_.CreationTimeUtc -ge $LaunchStartUtc
                } |
                Sort-Object -Property CreationTimeUtc, LastWriteTimeUtc, Name -Descending
        )
        if ($candidates.Count -eq 1) {
            return $candidates[0]
        }
        if ($candidates.Count -gt 1) {
            $candidateList = ($candidates | ForEach-Object { $_.FullName }) -join '; '
            throw "Ambiguous new sac_* run directories created after launch start under ${ExpDir}: $candidateList"
        }

        $workload = Get-Process -Id $WorkloadPid -ErrorAction SilentlyContinue
        if ($null -eq $workload) {
            throw "Training workload PID $WorkloadPid exited before a new sac_* run directory was created. Check stdout/stderr under $ExpDir."
        }

        Start-Sleep -Seconds 2
    }

    throw "Timed out after $TimeoutSeconds seconds waiting for a new sac_* run directory under $ExpDir."
}

try {
    $changedKnobsForValidation = $ChangedKnobsJson | ConvertFrom-Json -ErrorAction Stop
} catch {
    throw "ChangedKnobsJson must be valid JSON. Parser error: $($_.Exception.Message)"
}
if ($null -eq $changedKnobsForValidation -or $changedKnobsForValidation -is [array] -or $changedKnobsForValidation -isnot [pscustomobject]) {
    throw 'ChangedKnobsJson must decode to a JSON object.'
}

$RepoRoot = Resolve-RequiredPath -Path (Join-Path -Path $PSScriptRoot -ChildPath '..\..') -Description 'Repository root'
$SrcDir = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'src') -Description 'HOPE src directory'
$Python = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe') -Description 'Project Python executable'
$TrainScript = Resolve-RequiredPath -Path (Join-Path -Path $SrcDir -ChildPath 'train\train_HOPE_sac.py') -Description 'Original HOPE SAC training script'
$FastActionMaskTrainScript = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\train_HOPE_sac_fast_action_mask.py') -Description 'Stage 3 fast action-mask training wrapper'
$MonitorScript = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\monitor_stage3_resources.ps1') -Description 'Stage 3 resource monitor script'
Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\stage3_manifest.py') -Description 'Stage 3 manifest helper' | Out-Null

$ExpDir = Join-Path -Path $SrcDir -ChildPath 'log\exp'
[System.IO.Directory]::CreateDirectory($ExpDir) | Out-Null
$beforeRunDirs = @{}
Get-ChildItem -LiteralPath $ExpDir -Directory -Filter 'sac_*' -ErrorAction SilentlyContinue |
    ForEach-Object { $beforeRunDirs[$_.FullName] = $true }

$launchStartUtc = (Get-Date).ToUniversalTime()
$launchStartUtcText = $launchStartUtc.ToString('o')
$launchStamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$safeRunName = ConvertTo-SafeFileName -Value $RunName
$logPrefix = "${safeRunName}_${launchStamp}"
$StdoutPath = Join-Path -Path $ExpDir -ChildPath "$logPrefix.stdout.log"
$StderrPath = Join-Path -Path $ExpDir -ChildPath "$logPrefix.stderr.log"
$ResourceCsvPath = Join-Path -Path $ExpDir -ChildPath "$logPrefix.resources.csv"
$ManifestPath = Join-Path -Path $ExpDir -ChildPath "$logPrefix.meta.json"

$env:SDL_VIDEODRIVER = 'dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'

$trainingEntryPoint = '.\train\train_HOPE_sac.py'
if ($FastActionMask) {
    $trainingEntryPoint = $FastActionMaskTrainScript
}

$trainingArgs = @(
    $trainingEntryPoint,
    '--train_episode',
    "$TrainEpisode",
    '--eval_episode',
    "$EvalEpisode",
    '--visualize=',
    '--verbose='
)

$workload = Start-Process `
    -FilePath $Python `
    -ArgumentList $trainingArgs `
    -WorkingDirectory $SrcDir `
    -RedirectStandardOutput $StdoutPath `
    -RedirectStandardError $StderrPath `
    -WindowStyle Hidden `
    -PassThru
$workloadStartedAtUtcText = (Get-Date).ToUniversalTime().ToString('o')

try {
    $runDirItem = Wait-Stage3RunDirectory -ExpDir $ExpDir -BeforeRunDirs $beforeRunDirs -LaunchStartUtc $launchStartUtc -WorkloadPid $workload.Id
} catch {
    $activeWorkload = Get-Process -Id $workload.Id -ErrorAction SilentlyContinue
    if ($null -ne $activeWorkload) {
        Stop-Process -Id $workload.Id -ErrorAction SilentlyContinue
    }
    throw
}

$powerShellExe = Get-CurrentPowerShellExecutable
$monitorArgs = @(
    '-NoProfile',
    '-ExecutionPolicy',
    'Bypass',
    '-File',
    $MonitorScript,
    '-ProcessId',
    "$($workload.Id)",
    '-OutputPath',
    $ResourceCsvPath
) | ForEach-Object { Quote-ProcessArgument -Value $_ }

$monitor = Start-Process `
    -FilePath $powerShellExe `
    -ArgumentList $monitorArgs `
    -WorkingDirectory $RepoRoot `
    -WindowStyle Hidden `
    -PassThru
$monitorStartedAtUtcText = (Get-Date).ToUniversalTime().ToString('o')

$commandForManifest = @($Python) + $trainingArgs
$commandJson = $commandForManifest | ConvertTo-Json -Compress
$environmentJson = @{
    SDL_VIDEODRIVER = $env:SDL_VIDEODRIVER
    TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = $env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD
    STAGE3_FAST_ACTION_MASK = if ($FastActionMask) { '1' } else { '0' }
} | ConvertTo-Json -Compress

$manifestCode = @'
import json
import sys
from pathlib import Path

from tools.stage3.stage3_manifest import build_manifest, write_manifest

repo_root = Path(sys.argv[1])
manifest_path = Path(sys.argv[2])
run_name = sys.argv[3]
candidate_type = sys.argv[4]
changed_knobs = json.loads(sys.argv[5])
if not isinstance(changed_knobs, dict):
    raise SystemExit("ChangedKnobsJson must decode to a JSON object.")
command = json.loads(sys.argv[6])
environment = json.loads(sys.argv[7])

manifest = build_manifest(
    repo_root=repo_root,
    run_name=run_name,
    candidate_type=candidate_type,
    changed_knobs=changed_knobs,
    command=command,
    python_executable=sys.argv[8],
    env=environment,
    run_dir=sys.argv[9],
    stdout_path=sys.argv[10],
    stderr_path=sys.argv[11],
    resource_csv_path=sys.argv[12],
)
manifest.update(
    {
        "launcher_pid": int(sys.argv[13]),
        "workload_pid": int(sys.argv[14]),
        "resource_monitor_pid": int(sys.argv[15]),
        "gate_status": "running",
        "launcher_started_at_utc": sys.argv[16],
        "workload_started_at_utc": sys.argv[17],
        "resource_monitor_started_at_utc": sys.argv[18],
        "train_episode": int(sys.argv[19]),
        "eval_episode": int(sys.argv[20]),
    }
)
write_manifest(manifest, manifest_path)
print(str(manifest_path))
'@

$manifestExitCode = 0
Push-Location -LiteralPath $RepoRoot
try {
    & $Python -c $manifestCode `
        $RepoRoot `
        $ManifestPath `
        $RunName `
        $CandidateType `
        $ChangedKnobsJson `
        $commandJson `
        $environmentJson `
        $Python `
        $runDirItem.FullName `
        $StdoutPath `
        $StderrPath `
        $ResourceCsvPath `
        "$PID" `
        "$($workload.Id)" `
        "$($monitor.Id)" `
        $launchStartUtcText `
        $workloadStartedAtUtcText `
        $monitorStartedAtUtcText `
        "$TrainEpisode" `
        "$EvalEpisode"
    $manifestExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($manifestExitCode -ne 0) {
    Stop-Process -Id $monitor.Id -ErrorAction SilentlyContinue
    Stop-Process -Id $workload.Id -ErrorAction SilentlyContinue
    throw "Manifest write failed with exit code $manifestExitCode. The workload PID was $($workload.Id), monitor PID was $($monitor.Id); both were stopped to avoid an untracked long run."
}

Write-Output "Manifest: $ManifestPath"
Write-Output "Run directory: $($runDirItem.FullName)"
Write-Output "Expected 20K checkpoint: $(Join-Path -Path $runDirItem.FullName -ChildPath 'SAC_19999.pt')"
Write-Output "Workload PID: $($workload.Id)"
Write-Output "Resource monitor PID: $($monitor.Id)"
Write-Output "Stdout: $StdoutPath"
Write-Output "Stderr: $StderrPath"
Write-Output "Resource CSV: $ResourceCsvPath"
