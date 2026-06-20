param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$RunDir,
    [Parameter(Mandatory=$true)][int]$GateEpisode
)

$ErrorActionPreference = 'Stop'
$RepoRoot = (Resolve-Path -LiteralPath (Join-Path -Path $PSScriptRoot -ChildPath '..\..')).Path
$Python = (Resolve-Path -LiteralPath (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe')).Path
$SummaryScript = Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\tensorboard_stage3_summary.py'
$GateJson = Join-Path -Path $RepoRoot -ChildPath "docs\research\stage4_ogm_gate_${GateEpisode}.tensorboard.json"
$GateMd = Join-Path -Path $RepoRoot -ChildPath "docs\research\stage4_ogm_gate_${GateEpisode}.tensorboard.md"
$ResolvedManifestPath = (Resolve-Path -LiteralPath $ManifestPath).Path
$ResolvedRunDir = (Resolve-Path -LiteralPath $RunDir).Path
$Meta = Get-Content -LiteralPath $ResolvedManifestPath -Raw | ConvertFrom-Json

if ($null -ne $Meta.run_dir -and -not [string]::IsNullOrWhiteSpace($Meta.run_dir)) {
    $ManifestRunDir = (Resolve-Path -LiteralPath $Meta.run_dir).Path
    if (-not [string]::Equals($ManifestRunDir, $ResolvedRunDir, [StringComparison]::OrdinalIgnoreCase)) {
        throw "Manifest run_dir does not match supplied RunDir. Manifest: $ManifestRunDir Supplied: $ResolvedRunDir"
    }
}

& $Python $SummaryScript $ResolvedRunDir --min-episodes $GateEpisode --json $GateJson --markdown $GateMd

$checkpointEpisode = $GateEpisode - 1
$checkpoint = Join-Path -Path $ResolvedRunDir -ChildPath "SAC_${checkpointEpisode}.pt"
if (-not (Test-Path -LiteralPath $checkpoint)) {
    throw "Expected checkpoint missing at gate ${GateEpisode}: $checkpoint"
}

Write-Output "TensorBoard summary: $GateJson"
Write-Output "Checkpoint: $checkpoint"
