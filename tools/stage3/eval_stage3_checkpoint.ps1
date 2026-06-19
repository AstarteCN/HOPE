param(
    [Parameter(Mandatory=$true)][string]$ManifestPath,
    [Parameter(Mandatory=$true)][string]$CheckpointPath,
    [int]$EvalEpisode = 200
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

function Test-EvalResultDirectory {
    param([Parameter(Mandatory=$true)][string]$Path)

    $requiredResults = @(
        'extreme\result.txt',
        'dlp\result.txt',
        'complex\result.txt',
        'normalize\result.txt'
    )

    foreach ($relativePath in $requiredResults) {
        if (-not (Test-Path -LiteralPath (Join-Path -Path $Path -ChildPath $relativePath))) {
            return $false
        }
    }
    return $true
}

function Find-NewEvaluationDirectory {
    param(
        [Parameter(Mandatory=$true)][string]$EvalRoot,
        [Parameter(Mandatory=$true)]$BeforeDirs,
        [Parameter(Mandatory=$true)][datetime]$EvalStartUtc
    )

    if (-not (Test-Path -LiteralPath $EvalRoot)) {
        throw "Evaluation root was not created: $EvalRoot"
    }

    $afterDirs = @(Get-ChildItem -LiteralPath $EvalRoot -Directory -ErrorAction SilentlyContinue)
    $newDirs = @(
        $afterDirs |
            Where-Object {
                -not $BeforeDirs.ContainsKey($_.FullName) -and
                $_.CreationTimeUtc -ge $EvalStartUtc -and
                (Test-EvalResultDirectory -Path $_.FullName)
            } |
            Sort-Object -Property CreationTimeUtc, LastWriteTimeUtc, Name -Descending
    )
    if ($newDirs.Count -eq 1) {
        return $newDirs[0]
    }
    if ($newDirs.Count -gt 1) {
        $candidateList = ($newDirs | ForEach-Object { $_.FullName }) -join '; '
        throw "Ambiguous evaluation output directories created by the eval command under ${EvalRoot}: $candidateList"
    }

    $updatedDirs = @(
        $afterDirs |
            Where-Object {
                $_.LastWriteTimeUtc -ge $EvalStartUtc -and
                (Test-EvalResultDirectory -Path $_.FullName)
            } |
            Sort-Object -Property LastWriteTimeUtc, Name -Descending
    )
    if ($updatedDirs.Count -eq 1) {
        return $updatedDirs[0]
    }
    if ($updatedDirs.Count -gt 1) {
        $candidateList = ($updatedDirs | ForEach-Object { $_.FullName }) -join '; '
        throw "Ambiguous updated evaluation output directories matched after eval under ${EvalRoot}: $candidateList"
    }

    throw "Could not discover a completed evaluation output directory created by the eval command under $EvalRoot."
}

$RepoRoot = Resolve-RequiredPath -Path (Join-Path -Path $PSScriptRoot -ChildPath '..\..') -Description 'Repository root'
$SrcDir = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'src') -Description 'HOPE src directory'
$Python = Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath '.venv\Scripts\python.exe') -Description 'Project Python executable'
$EvalScript = Resolve-RequiredPath -Path (Join-Path -Path $SrcDir -ChildPath 'evaluation\eval_mix_scene.py') -Description 'Original HOPE mixed-scene evaluator'
Resolve-RequiredPath -Path (Join-Path -Path $RepoRoot -ChildPath 'tools\stage3\stage3_manifest.py') -Description 'Stage 3 manifest helper' | Out-Null
$ResolvedManifestPath = Resolve-RequiredPath -Path $ManifestPath -Description 'Stage 3 manifest'
$ResolvedCheckpointPath = Resolve-RequiredPath -Path $CheckpointPath -Description 'Checkpoint'

$EvalRoot = Join-Path -Path $SrcDir -ChildPath 'log\eval'
$beforeDirs = @{}
if (Test-Path -LiteralPath $EvalRoot) {
    Get-ChildItem -LiteralPath $EvalRoot -Directory -ErrorAction SilentlyContinue |
        ForEach-Object { $beforeDirs[$_.FullName] = $true }
}

$env:SDL_VIDEODRIVER = 'dummy'
$env:TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD = '1'
$evalStartUtc = (Get-Date).ToUniversalTime()
$evalArgs = @(
    '.\evaluation\eval_mix_scene.py',
    $ResolvedCheckpointPath,
    '--eval_episode',
    "$EvalEpisode",
    '--visualize=',
    '--verbose='
)

Push-Location -LiteralPath $SrcDir
try {
    & $Python @evalArgs
    if ($LASTEXITCODE -ne 0) {
        throw "Evaluation failed with exit code $LASTEXITCODE for checkpoint $ResolvedCheckpointPath."
    }
} finally {
    Pop-Location
}

$evalDir = Find-NewEvaluationDirectory -EvalRoot $EvalRoot -BeforeDirs $beforeDirs -EvalStartUtc $evalStartUtc

$updateCode = @'
import sys
from pathlib import Path

from tools.stage3.stage3_manifest import read_manifest, utc_timestamp, write_manifest

manifest_path = Path(sys.argv[1])
eval_dir = sys.argv[2]
manifest = read_manifest(manifest_path)
eval_result_dirs = manifest.get("eval_result_dirs")
if not isinstance(eval_result_dirs, dict):
    eval_result_dirs = {}
eval_result_dirs["candidate_20k"] = eval_dir
manifest["eval_result_dirs"] = eval_result_dirs
manifest["updated_at_utc"] = utc_timestamp()
write_manifest(manifest, manifest_path)
'@

$updateExitCode = 0
Push-Location -LiteralPath $RepoRoot
try {
    & $Python -c $updateCode $ResolvedManifestPath $evalDir.FullName
    $updateExitCode = $LASTEXITCODE
} finally {
    Pop-Location
}
if ($updateExitCode -ne 0) {
    throw "Manifest eval_result_dirs update failed with exit code ${updateExitCode}: $ResolvedManifestPath"
}

Write-Output "Evaluation directory: $($evalDir.FullName)"
Write-Output "Updated manifest: $ResolvedManifestPath"
