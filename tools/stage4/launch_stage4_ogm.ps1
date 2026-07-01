param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [int]$TrainEpisode = 100000,
    [int]$StartEpisode = 0,
    [int]$EvalEpisode = 200,
    [string]$ResumeCheckpoint,
    [string]$ResumeState,
    [string]$ExpDirOverride,
    [switch]$DryRun,
    [ValidateSet('legacy-coscos','corrected-cossin')][string]$TargetMode = 'legacy-coscos',
    [ValidateSet('off','diagnostic','direction_hold')][string]$ManeuverStabilityMode = 'off',
    [ValidateSet('rl','rl-rs','all')][string]$ManeuverStabilityApplyTo = 'rl',
    [double]$ManeuverStabilityMinSpeed = 0.000001,
    [int]$ManeuverStabilityHoldSteps = 1,
    [Nullable[double]]$ManeuverStabilityMaxHoldSpeed = $null,
    [string]$ChangedKnobsJson = '{"policy_inputs":"target+action_mask+ogm","rgb_bev_policy":false,"internal_lidar_for_action_mask":true}'
)

$ErrorActionPreference = 'Stop'

if (-not [string]::IsNullOrWhiteSpace($ResumeState) -and -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    throw "-ResumeState cannot be combined with -ResumeCheckpoint. Use one continuation source."
}
if (-not [string]::IsNullOrWhiteSpace($ResumeState) -and $StartEpisode -le 0) {
    throw "-ResumeState requires -StartEpisode greater than 0."
}
if ($TargetMode -eq 'corrected-cossin' -and (
        -not [string]::IsNullOrWhiteSpace($ResumeCheckpoint) -or
        ($StartEpisode -ne 0 -and [string]::IsNullOrWhiteSpace($ResumeState))
    )) {
    throw "-TargetMode corrected-cossin must start from a fresh run or matching -ResumeState; do not pass -ResumeCheckpoint or continue without -ResumeState."
}

function ConvertTo-ProcessArgument {
    param([AllowNull()][string]$Argument)

    if ($null -eq $Argument -or $Argument.Length -eq 0) {
        return '""'
    }
    if ($Argument -notmatch '[\s"]') {
        return $Argument
    }

    $escaped = $Argument -replace '(\\*)"', '$1$1\"'
    $escaped = $escaped -replace '(\\+)$', '$1$1'
    return '"' + $escaped + '"'
}

function Join-ProcessArgumentList {
    param([string[]]$ArgumentList)

    return (($ArgumentList | ForEach-Object { ConvertTo-ProcessArgument $_ }) -join ' ')
}

$changedKnobs = $ChangedKnobsJson | ConvertFrom-Json -ErrorAction Stop
$changedKnobs | Add-Member -NotePropertyName target_mode -NotePropertyValue $TargetMode -Force
$changedKnobs | Add-Member -NotePropertyName maneuver_stability_mode -NotePropertyValue $ManeuverStabilityMode -Force
$changedKnobs | Add-Member -NotePropertyName maneuver_stability_apply_to -NotePropertyValue $ManeuverStabilityApplyTo -Force
$changedKnobs | Add-Member -NotePropertyName maneuver_stability_min_speed -NotePropertyValue $ManeuverStabilityMinSpeed -Force
$changedKnobs | Add-Member -NotePropertyName maneuver_stability_hold_steps -NotePropertyValue $ManeuverStabilityHoldSteps -Force
$changedKnobs | Add-Member -NotePropertyName maneuver_stability_max_hold_speed -NotePropertyValue $ManeuverStabilityMaxHoldSpeed -Force

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$SrcDir = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'src')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$TrainScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage4\train_HOPE_sac_ogm.py')).Path
$MonitorScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\monitor_stage3_resources.ps1')).Path
$ExpDir = if ([string]::IsNullOrWhiteSpace($ExpDirOverride)) {
    Join-Path -Path $SrcDir -ChildPath 'log\exp'
} else {
    $ExpDirOverride
}
[System.IO.Directory]::CreateDirectory($ExpDir) | Out-Null
$ExpDir = (Resolve-Path -LiteralPath $ExpDir).Path

$env:SDL_VIDEODRIVER = 'dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'

$stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$safeRunName = $RunName -replace '[^A-Za-z0-9_.-]', '_'
$prefix = "${safeRunName}_${stamp}"
$RunDir = Join-Path -Path $ExpDir -ChildPath "sac_ogm_${prefix}"
$stdout = Join-Path -Path $ExpDir -ChildPath "$prefix.stdout.log"
$stderr = Join-Path -Path $ExpDir -ChildPath "$prefix.stderr.log"
$resourceCsv = Join-Path -Path $ExpDir -ChildPath "$prefix.resources.csv"
$manifest = Join-Path -Path $ExpDir -ChildPath "$prefix.meta.json"

$args = @(
    $TrainScript,
    '--train_episode', "$TrainEpisode",
    '--start_episode', "$StartEpisode",
    '--eval_episode', "$EvalEpisode",
    '--target_mode', $TargetMode,
    '--maneuver_stability_mode', $ManeuverStabilityMode,
    '--maneuver_stability_apply_to', $ManeuverStabilityApplyTo,
    '--maneuver_stability_min_speed', "$ManeuverStabilityMinSpeed",
    '--maneuver_stability_hold_steps', "$ManeuverStabilityHoldSteps",
    '--run_dir', $RunDir,
    '--visualize=',
    '--verbose='
)
if ($null -ne $ManeuverStabilityMaxHoldSpeed) {
    $args += @('--maneuver_stability_max_hold_speed', "$ManeuverStabilityMaxHoldSpeed")
}
if (-not [string]::IsNullOrWhiteSpace($ResumeCheckpoint)) {
    $args += @('--resume_checkpoint', $ResumeCheckpoint)
}
if (-not [string]::IsNullOrWhiteSpace($ResumeState)) {
    $args += @('--resume_state', $ResumeState)
}
$runnerArgumentString = Join-ProcessArgumentList $args
$resumeCheckpointForManifest = if ([string]::IsNullOrWhiteSpace($ResumeCheckpoint)) { $null } else { $ResumeCheckpoint }
$resumeStateForManifest = if ([string]::IsNullOrWhiteSpace($ResumeState)) { $null } else { $ResumeState }
$continuationType = if ($null -ne $resumeStateForManifest) {
    'full_state'
} elseif ($null -ne $resumeCheckpointForManifest) {
    'checkpoint_based'
} else {
    'fresh'
}

$initialWorkloadPid = $null
$workloadPid = $null
$workloadPidSource = 'dry_run'
$monitorPid = $null

if (-not $DryRun) {
    $workload = Start-Process -FilePath $Python -ArgumentList $runnerArgumentString -WorkingDirectory $SrcDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

    $initialWorkloadPid = $workload.Id
    $workloadPid = $initialWorkloadPid
    $workloadPidSource = 'initial_process'
    $deadline = (Get-Date).AddSeconds(10)
    do {
        $childPython = Get-CimInstance -ClassName Win32_Process -Filter "ParentProcessId = $initialWorkloadPid" |
            Where-Object { $_.Name -like 'python*.exe' } |
            Sort-Object -Property CreationDate |
            Select-Object -First 1
        if ($null -ne $childPython) {
            $workloadPid = [int]$childPython.ProcessId
            $workloadPidSource = 'child_python'
            break
        }
        Start-Sleep -Milliseconds 250
    } while ((Get-Date) -lt $deadline)

}

$monitorArgs = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $MonitorScript,
    '-ProcessId', "$workloadPid",
    '-OutputPath', $resourceCsv
)
$monitorArgumentString = Join-ProcessArgumentList $monitorArgs

if (-not $DryRun) {
    $monitor = Start-Process -FilePath powershell -ArgumentList $monitorArgumentString -WindowStyle Hidden -PassThru
    $monitorPid = $monitor.Id
}

$meta = [ordered]@{
    schema_version = 1
    run_name = $RunName
    candidate_type = 'stage4_ogm_proxy'
    train_episode = $TrainEpisode
    start_episode = $StartEpisode
    eval_episode = $EvalEpisode
    target_mode = $TargetMode
    resume_checkpoint = $resumeCheckpointForManifest
    resume_state = $resumeStateForManifest
    continuation_type = $continuationType
    dry_run = [bool]$DryRun
    runner_args = $args
    runner_argument_string = $runnerArgumentString
    monitor_args = $monitorArgs
    monitor_argument_string = $monitorArgumentString
    workload_pid = $workloadPid
    initial_workload_pid = $initialWorkloadPid
    workload_pid_source = $workloadPidSource
    monitor_pid = $monitorPid
    stdout_path = $stdout
    stderr_path = $stderr
    resource_csv_path = $resourceCsv
    manifest_path = $manifest
    run_dir = $RunDir
    changed_knobs = $changedKnobs
    first_gate_episode = 20000
    progress_gate_interval = 10000
    max_train_episode = 120000
    created_at = (Get-Date).ToUniversalTime().ToString('o')
}
$meta | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $manifest -Encoding UTF8
Write-Output $manifest
