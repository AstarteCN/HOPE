param(
    [Parameter(Mandatory = $true)]
    [int]$ProcessId,

    [Parameter(Mandatory = $true)]
    [string]$OutputPath,

    [ValidateRange(1, 2147483647)]
    [int]$IntervalSeconds = 5,

    [int]$MaxSamples = 0
)

$ErrorActionPreference = 'Stop'

$resolvedOutput = [System.IO.Path]::GetFullPath($OutputPath)
$outputDirectory = Split-Path -Parent $resolvedOutput
if (-not [string]::IsNullOrWhiteSpace($outputDirectory) -and -not (Test-Path -LiteralPath $outputDirectory)) {
    [System.IO.Directory]::CreateDirectory($outputDirectory) | Out-Null
}

$columns = @(
    'timestamp',
    'pid',
    'process_name',
    'cpu_percent',
    'working_set_mb',
    'private_memory_mb',
    'gpu_util_percent',
    'gpu_memory_used_mb',
    'gpu_memory_total_mb',
    'gpu_power_w'
)
$header = $columns -join ','
if (-not (Test-Path -LiteralPath $resolvedOutput) -or (Get-Item -LiteralPath $resolvedOutput).Length -eq 0) {
    Set-Content -LiteralPath $resolvedOutput -Value $header -Encoding UTF8
} else {
    $existingHeader = Get-Content -LiteralPath $resolvedOutput -TotalCount 1
    if ($existingHeader -ne $header) {
        throw "Existing CSV header does not match expected monitor schema: $resolvedOutput"
    }
}

$processorCount = [Environment]::ProcessorCount
$lastCpuSeconds = $null
$lastTimestamp = $null
$sampleCount = 0

while ($true) {
    $now = Get-Date
    $process = Get-Process -Id $ProcessId -ErrorAction SilentlyContinue
    if ($null -eq $process) {
        break
    }

    $cpuPercent = 0.0
    if ($null -ne $lastCpuSeconds -and $null -ne $lastTimestamp) {
        $elapsed = ($now - $lastTimestamp).TotalSeconds
        $cpuDelta = $process.CPU - $lastCpuSeconds
        if ($elapsed -gt 0 -and $processorCount -gt 0) {
            $cpuPercent = [Math]::Round(($cpuDelta / $elapsed / $processorCount) * 100.0, 2)
        }
    }

    $gpuUtil = ''
    $gpuMemoryUsed = ''
    $gpuMemoryTotal = ''
    $gpuPower = ''
    $nvidiaSmi = Get-Command nvidia-smi -ErrorAction SilentlyContinue
    if ($null -ne $nvidiaSmi) {
        $gpuOutput = & nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,power.draw --format=csv,noheader,nounits 2>$null
        $gpuLine = $gpuOutput | Select-Object -First 1
        if (-not [string]::IsNullOrWhiteSpace($gpuLine)) {
            $parts = $gpuLine -split ',' | ForEach-Object { $_.Trim() }
            if ($parts.Count -ge 4) {
                $gpuUtil = $parts[0]
                $gpuMemoryUsed = $parts[1]
                $gpuMemoryTotal = $parts[2]
                $gpuPower = $parts[3]
            }
        }
    }

    $rowObject = [pscustomobject][ordered]@{
        timestamp = $now.ToString('o')
        pid = $process.Id
        process_name = $process.ProcessName
        cpu_percent = $cpuPercent
        working_set_mb = [Math]::Round($process.WorkingSet64 / 1MB, 2)
        private_memory_mb = [Math]::Round($process.PrivateMemorySize64 / 1MB, 2)
        gpu_util_percent = $gpuUtil
        gpu_memory_used_mb = $gpuMemoryUsed
        gpu_memory_total_mb = $gpuMemoryTotal
        gpu_power_w = $gpuPower
    }
    $row = $rowObject | ConvertTo-Csv -NoTypeInformation | Select-Object -Skip 1
    Add-Content -LiteralPath $resolvedOutput -Value $row -Encoding UTF8

    $lastCpuSeconds = $process.CPU
    $lastTimestamp = $now
    $sampleCount += 1

    if ($MaxSamples -gt 0 -and $sampleCount -ge $MaxSamples) {
        break
    }

    Start-Sleep -Seconds $IntervalSeconds
}
