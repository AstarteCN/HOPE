param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)][int]$GateEpisode,
    [string]$OutputDir = ''
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$SummaryScript = Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\tensorboard_stage3_summary.py'
$SummaryOutputDir = Join-Path -Path $RepoRoot -ChildPath 'docs\research'
if (-not [string]::IsNullOrWhiteSpace($OutputDir)) {
    $SummaryOutputDir = $OutputDir
}
$GateJson = Join-Path -Path $SummaryOutputDir -ChildPath "stage4_ogm_gate_${GateEpisode}.tensorboard.json"
$GateMd = Join-Path -Path $SummaryOutputDir -ChildPath "stage4_ogm_gate_${GateEpisode}.tensorboard.md"
$ResolvedManifestPath = (Resolve-Path -LiteralPath $ManifestPath).Path
$ResolvedRunDir = (Resolve-Path -LiteralPath $RunDir).Path
$Meta = Get-Content -LiteralPath $ResolvedManifestPath -Raw | ConvertFrom-Json

if ($null -ne $Meta.run_dir -and -not [string]::IsNullOrWhiteSpace($Meta.run_dir)) {
    $ManifestRunDirPath = [string]$Meta.run_dir
    if (-not [System.IO.Path]::IsPathRooted($ManifestRunDirPath)) {
        $ManifestDir = Split-Path -Parent $ResolvedManifestPath
        $ManifestRunDirPath = Join-Path -Path $ManifestDir -ChildPath $ManifestRunDirPath
    }
    $ManifestRunDir = (Resolve-Path -LiteralPath $ManifestRunDirPath).Path
    if (-not [string]::Equals($ManifestRunDir, $ResolvedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest run_dir does not match supplied RunDir. Manifest: $ManifestRunDir Supplied: $ResolvedRunDir"
    }
}

& $Python $SummaryScript $ResolvedRunDir --min-episodes $GateEpisode --json $GateJson --markdown $GateMd
if ($LASTEXITCODE -ne 0) {
    throw "TensorBoard summary failed with exit code $LASTEXITCODE for gate ${GateEpisode}: $ResolvedRunDir"
}

$Summary = Get-Content -LiteralPath $GateJson -Raw | ConvertFrom-Json
$EpisodeCount = [int]$Summary.episode_count
$TrainingBudgetMet = $EpisodeCount -ge $GateEpisode
if ($null -ne $Summary.training_budget_met) {
    $TrainingBudgetMet = [bool]$Summary.training_budget_met
}
$LastGlobalStep = $null
if ($null -ne $Summary.scalars -and $null -ne $Summary.scalars.step_num -and $null -ne $Summary.scalars.step_num.last_step) {
    $LastGlobalStep = [int]$Summary.scalars.step_num.last_step
}
$GlobalGateMet = $false
if ($null -ne $LastGlobalStep) {
    $GlobalGateMet = $LastGlobalStep -ge ($GateEpisode - 1)
}
$GateReady = $TrainingBudgetMet -or ($EpisodeCount -ge $GateEpisode) -or $GlobalGateMet

if (-not $GateReady) {
    Write-Output "TensorBoard summary: $GateJson"
    Write-Output "episode_count: $EpisodeCount"
    if ($null -ne $LastGlobalStep) {
        Write-Output "last_global_step: $LastGlobalStep"
    }
    Write-Output "training_budget_met: $($TrainingBudgetMet.ToString().ToLowerInvariant())"
    Write-Output "gate_ready: false"
    Write-Output "Gate pending: checkpoint validation waits until gate $GateEpisode is reached."
    exit 0
}

$checkpointEpisode = $GateEpisode - 1
$checkpoint = Join-Path -Path $ResolvedRunDir -ChildPath "SAC_${checkpointEpisode}.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Expected checkpoint missing at gate ${GateEpisode}: $checkpoint"
}
$stateSnapshot = Join-Path -Path $ResolvedRunDir -ChildPath "stage4_state_${checkpointEpisode}.pt"
if (-not (Test-Path -LiteralPath $stateSnapshot)) {
    throw "Expected state snapshot missing at gate ${GateEpisode}: $stateSnapshot"
}

Write-Output "TensorBoard summary: $GateJson"
Write-Output "episode_count: $EpisodeCount"
if ($null -ne $LastGlobalStep) {
    Write-Output "last_global_step: $LastGlobalStep"
}
Write-Output "training_budget_met: $($TrainingBudgetMet.ToString().ToLowerInvariant())"
Write-Output "gate_ready: true"
Write-Output "Checkpoint: $checkpoint"
Write-Output "StateSnapshot: $stateSnapshot"
