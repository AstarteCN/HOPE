param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [int]$MinEpisodes = 20000
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

function Stop-ManifestProcess {
    param(
        [Parameter(Mandatory=$false)]$PidValue,
        [Parameter(Mandatory=$true)][string]$Label,
        [Parameter(Mandatory=$true)][string[]]$ExpectedProcessNames,
        [Parameter(Mandatory=$false)][string]$StartedAtUtcText,
        [Parameter(Mandatory=$false)][string]$ExpectedExecutablePath,
        [Parameter(Mandatory=$false)][string[]]$ExpectedCommandFragments = @(),
        [int]$StartToleranceMinutes = 2
    )

    if ($null -eq $PidValue) {
        throw "${Label} PID is not recorded."
    }

    $pidText = [string]$PidValue
    if ([string]::IsNullOrWhiteSpace($pidText)) {
        throw "${Label} PID is blank."
    }

    $pidNumber = 0
    if (-not [int]::TryParse($pidText, [ref]$pidNumber) -or $pidNumber -le 0) {
        throw "${Label} PID is not a positive integer: $pidText"
    }

    if ($pidNumber -eq $PID) {
        throw "${Label} PID $pidNumber is the current stop script process."
    }

    $process = Get-Process -Id $pidNumber -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        return [pscustomobject]@{
            label = $Label
            pid = $pidNumber
            status = 'already_inactive'
            message = "${Label} PID $pidNumber is not active."
        }
    }

    $expectedNames = @($ExpectedProcessNames | ForEach-Object { $_.ToLowerInvariant() })
    $actualName = $process.ProcessName.ToLowerInvariant()
    if ($expectedNames -notcontains $actualName) {
        throw "${Label} PID $pidNumber process name '$($process.ProcessName)' is not one of: $($ExpectedProcessNames -join ', ')."
    }

    if ([string]::IsNullOrWhiteSpace($StartedAtUtcText)) {
        throw "${Label} PID $pidNumber manifest start time is missing."
    }

    $recordedStartOffset = [datetimeoffset]::MinValue
    if (-not [datetimeoffset]::TryParse($StartedAtUtcText, [ref]$recordedStartOffset)) {
        throw "${Label} PID $pidNumber manifest start time could not be parsed: $StartedAtUtcText"
    }

    try {
        $processStartUtc = $process.StartTime.ToUniversalTime()
    } catch {
        throw "${Label} PID $pidNumber process StartTime is unavailable: $($_.Exception.Message)"
    }

    $earliestAllowedStartUtc = $recordedStartOffset.UtcDateTime.AddMinutes(-1 * $StartToleranceMinutes)
    $latestAllowedStartUtc = $recordedStartOffset.UtcDateTime.AddMinutes($StartToleranceMinutes)
    if ($processStartUtc -lt $earliestAllowedStartUtc -or $processStartUtc -gt $latestAllowedStartUtc) {
        throw "${Label} PID $pidNumber process start time $($processStartUtc.ToString('o')) is outside manifest start window $($earliestAllowedStartUtc.ToString('o')) to $($latestAllowedStartUtc.ToString('o'))."
    }

    $processDetails = Get-CimInstance -ClassName Win32_Process -Filter "ProcessId = $pidNumber" -ErrorAction SilentlyContinue
    if ($null -eq $processDetails) {
        throw "${Label} PID $pidNumber Win32_Process details are unavailable."
    }

    $actualExecutablePath = [string]$processDetails.ExecutablePath
    $commandLine = [string]$processDetails.CommandLine
    if ([string]::IsNullOrWhiteSpace($actualExecutablePath) -or [string]::IsNullOrWhiteSpace($commandLine)) {
        throw "${Label} PID $pidNumber ExecutablePath or CommandLine is unavailable."
    }

    if (-not [string]::IsNullOrWhiteSpace($ExpectedExecutablePath)) {
        $expectedResolvedExecutable = $ExpectedExecutablePath
        $actualResolvedExecutable = $actualExecutablePath
        try {
            $expectedResolvedExecutable = (Resolve-Path -LiteralPath $ExpectedExecutablePath -ErrorAction Stop).Path
        } catch {
            throw "${Label} PID $pidNumber expected executable path cannot be resolved: $ExpectedExecutablePath"
        }
        try {
            $actualResolvedExecutable = (Resolve-Path -LiteralPath $actualExecutablePath -ErrorAction Stop).Path
        } catch {
            throw "${Label} PID $pidNumber actual executable path cannot be resolved: $actualExecutablePath"
        }

        if ($expectedResolvedExecutable.ToLowerInvariant() -ne $actualResolvedExecutable.ToLowerInvariant()) {
            throw "${Label} PID $pidNumber executable path '$actualResolvedExecutable' does not match expected '$expectedResolvedExecutable'."
        }
    }

    foreach ($fragment in $ExpectedCommandFragments) {
        if ([string]::IsNullOrWhiteSpace($fragment)) {
            continue
        }
        if ($commandLine.IndexOf($fragment, [StringComparison]::OrdinalIgnoreCase) -lt 0) {
            throw "${Label} PID $pidNumber command line is missing expected fragment: $fragment"
        }
    }

    Stop-Process -Id $pidNumber -ErrorAction Stop
    $process.WaitForExit(5000) | Out-Null
    $processAfterStop = Get-Process -Id $pidNumber -ErrorAction SilentlyContinue
    if ($null -ne $processAfterStop) {
        throw "${Label} PID $pidNumber was signaled to stop but is still active."
    }

    return [pscustomobject]@{
        label = $Label
        pid = $pidNumber
        status = 'stopped'
        message = "Stopped ${Label} PID $pidNumber ($($process.ProcessName))."
    }
}

$RepoRoot = Resolve-RequiredPath -Path (Join-Path -Path $PSScriptRoot -ChildPath '..\..') -Description 'Repository root'
$Python = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe') -Description 'Project Python executable'
$SummaryScript = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\tensorboard_stage3_summary.py') -Description 'Stage 3 TensorBoard summary script'
Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\stage3_manifest.py') -Description 'Stage 3 manifest helper' | Out-Null
$ResolvedManifestPath = Resolve-RequiredPath -Path $ManifestPath -Description 'Stage 3 manifest'

$manifest = Get-Content -Raw -LiteralPath $ResolvedManifestPath | ConvertFrom-Json
if ($null -eq $manifest.run_dir -or [string]::IsNullOrWhiteSpace([string]$manifest.run_dir)) {
    throw "Manifest does not contain a locked run_dir: $ResolvedManifestPath"
}

$RunDir = Resolve-RequiredPath -Path ([string]$manifest.run_dir) -Description 'Locked Stage 3 run directory'
$expectedCheckpoint = [string]$manifest.expected_checkpoint_20k
if ([string]::IsNullOrWhiteSpace($expectedCheckpoint)) {
    $expectedCheckpoint = Join-Path -Path $RunDir -ChildPath 'SAC_19999.pt'
}

$manifestDirectory = Split-Path -Parent $ResolvedManifestPath
$manifestStem = [System.IO.Path]::GetFileNameWithoutExtension($ResolvedManifestPath)
$TensorboardSummaryJson = Join-Path -Path $manifestDirectory -ChildPath "$manifestStem.tensorboard_summary.json"
$TensorboardSummaryMarkdown = Join-Path -Path $manifestDirectory -ChildPath "$manifestStem.tensorboard_summary.md"

& $Python $SummaryScript `
    $RunDir `
    '--json' `
    $TensorboardSummaryJson `
    '--markdown' `
    $TensorboardSummaryMarkdown `
    '--min-episodes' `
    "$MinEpisodes"
if ($LASTEXITCODE -ne 0) {
    throw "TensorBoard summary failed with exit code $LASTEXITCODE for run_dir $RunDir."
}

$summary = Get-Content -Raw -LiteralPath $TensorboardSummaryJson | ConvertFrom-Json
if (-not [bool]$summary.training_budget_met) {
    throw "TensorBoard episode count $($summary.episode_count) is below MinEpisodes $MinEpisodes. Not stopping Stage 3 processes."
}

if (-not (Test-Path -LiteralPath $expectedCheckpoint)) {
    throw "Expected 20K checkpoint is missing. Not stopping Stage 3 processes: $expectedCheckpoint"
}

$expectedWorkloadExecutable = [string]$manifest.python_executable
if ([string]::IsNullOrWhiteSpace($expectedWorkloadExecutable)) {
    $expectedWorkloadExecutable = $Python
}

$stopResults = @()
$workloadStopResult = Stop-ManifestProcess `
    -PidValue $manifest.workload_pid `
    -Label 'workload' `
    -ExpectedProcessNames @('python', 'pythonw') `
    -StartedAtUtcText ([string]$manifest.workload_started_at_utc) `
    -ExpectedExecutablePath $expectedWorkloadExecutable `
    -ExpectedCommandFragments @('train_HOPE_sac.py', '--train_episode')
$stopResults += $workloadStopResult
Write-Output $workloadStopResult.message
Write-Output 'launcher_pid is treated as metadata only and will not be stopped.'
Start-Sleep -Seconds 2
$monitorCommandFragments = @(
    'monitor_stage3_resources.ps1',
    ([string]$manifest.workload_pid)
)
if (-not [string]::IsNullOrWhiteSpace([string]$manifest.resource_csv_path)) {
    $monitorCommandFragments += [string]$manifest.resource_csv_path
}
$monitorStopResult = Stop-ManifestProcess `
    -PidValue $manifest.resource_monitor_pid `
    -Label 'resource monitor' `
    -ExpectedProcessNames @('powershell', 'pwsh') `
    -StartedAtUtcText ([string]$manifest.resource_monitor_started_at_utc) `
    -ExpectedCommandFragments $monitorCommandFragments
$stopResults += $monitorStopResult
Write-Output $monitorStopResult.message

$allowedStopStatuses = @('stopped', 'already_inactive')
$unsafeStopResults = @($stopResults | Where-Object { $_.status -notin $allowedStopStatuses })
if ($unsafeStopResults.Count -gt 0) {
    $unsafeSummary = ($unsafeStopResults | ForEach-Object { "$($_.label) PID $($_.pid): $($_.status)" }) -join '; '
    throw "Unsafe process stop results prevent manifest update: $unsafeSummary"
}

$stoppedAtUtc = (Get-Date).ToUniversalTime().ToString('o')
$updatesJson = @{
    stopped_at_utc = $stoppedAtUtc
    tensorboard_summary_json = $TensorboardSummaryJson
    tensorboard_summary_markdown = $TensorboardSummaryMarkdown
    gate_status = 'stopped_at_20k'
    launcher_pid_metadata_only = $true
    process_stop_results = @($stopResults | ForEach-Object {
        @{
            label = $_.label
            pid = $_.pid
            status = $_.status
            message = $_.message
        }
    })
} | ConvertTo-Json -Depth 4 -Compress

$updateCode = @'
import json
import sys
from pathlib import Path

from tools.stage3.stage3_manifest import update_manifest

update_manifest(Path(sys.argv[1]), json.loads(sys.argv[2]))
'@

$updateExitCode = 0
Push-Location -LiteralPath $RepoRoot
try {
    & $Python -c $updateCode $ResolvedManifestPath $updatesJson
    $updateExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($updateExitCode -ne 0) {
    throw "Manifest update failed with exit code ${updateExitCode}: $ResolvedManifestPath"
}

Write-Output "Stopped Stage 3 run at/after $MinEpisodes episodes."
Write-Output "Manifest: $ResolvedManifestPath"
Write-Output "TensorBoard JSON: $TensorboardSummaryJson"
Write-Output "TensorBoard Markdown: $TensorboardSummaryMarkdown"
Write-Output "Preserved checkpoint: $expectedCheckpoint"
