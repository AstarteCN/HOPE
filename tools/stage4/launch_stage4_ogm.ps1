param(
    [Parameter(Mandatory=$true)][string]$RunName,
    [int]$TrainEpisode = 100000,
    [int]$EvalEpisode = 200,
    [string]$ChangedKnobsJson = '{"policy_inputs":"target+action_mask+ogm","rgb_bev_policy":false,"internal_lidar_for_action_mask":true}'
)

$ErrorActionPreference = 'Stop'

$changedKnobs = $ChangedKnobsJson | ConvertFrom-Json -ErrorAction Stop

$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$SrcDir = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'src')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$TrainScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage4\train_HOPE_sac_ogm.py')).Path
$MonitorScript = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\monitor_stage3_resources.ps1')).Path
$ExpDir = Join-Path -Path $SrcDir -ChildPath 'log\exp'
[System.IO.Directory]::CreateDirectory($ExpDir) | Out-Null

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
    '--eval_episode', "$EvalEpisode",
    '--run_dir', $RunDir,
    '--visualize=',
    '--verbose='
)

$workload = Start-Process -FilePath $Python -ArgumentList $args -WorkingDirectory $SrcDir -RedirectStandardOutput $stdout -RedirectStandardError $stderr -WindowStyle Hidden -PassThru

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

$monitor = Start-Process -FilePath powershell -ArgumentList @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', $MonitorScript,
    '-ProcessId', "$workloadPid",
    '-OutputPath', $resourceCsv
) -WindowStyle Hidden -PassThru

$meta = [ordered]@{
    schema_version = 1
    run_name = $RunName
    candidate_type = 'stage4_ogm_proxy'
    train_episode = $TrainEpisode
    eval_episode = $EvalEpisode
    workload_pid = $workloadPid
    initial_workload_pid = $initialWorkloadPid
    workload_pid_source = $workloadPidSource
    monitor_pid = $monitor.Id
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
