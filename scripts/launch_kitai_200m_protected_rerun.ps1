<#!
.SYNOPSIS
Launches the protected replacement KitAI 197.6M production pretraining run.

.DESCRIPTION
This script starts a new production run from random initialization in a separate,
hidden Windows process with redirected stdout and stderr. It deliberately does
not use --resume or --load-model, and it uses new log/checkpoint directories so
the terminated uncheckpointed attempt remains preserved for auditability.
#>
[CmdletBinding()]
param()

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot

$python = 'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe'
$config = '.\configs\kitai_200m_scratch_pretrain_protected_rerun.yaml'
$dataDirectory = '.\datasets\kitai_200m_scratch\pretrain_fineweb_edu'
$validationReport = '.\datasets\kitai_200m_scratch\validation_report.json'
$runLogDirectory = '.\logs\kitai_200m_scratch_pretrain_protected_rerun'
$runCheckpointDirectory = '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun'

if (-not (Test-Path $python)) { throw "Python executable not found: $python" }
if (-not (Test-Path $config)) { throw "Protected production config not found: $config" }
if (-not (Test-Path $validationReport)) { throw "Validated production data report not found: $validationReport" }

$validation = Get-Content $validationReport -Raw | ConvertFrom-Json
if (-not $validation.passed) { throw 'Validated production data report does not declare passed=true.' }

$activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts\.train' }
if ($activeTraining) {
    $activeIds = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Refusing to launch while a KitAI training command is already active (PID(s): $activeIds)."
}

if ((Test-Path $runLogDirectory) -or (Test-Path $runCheckpointDirectory)) {
    throw "Refusing to overwrite protected-rerun output directories. Inspect or archive $runLogDirectory and $runCheckpointDirectory first."
}

$shards = Get-ChildItem $dataDirectory -Filter 'fineweb_edu_*.txt' -File | Sort-Object Name
if ($shards.Count -ne 15) { throw "Expected exactly 15 preserved FineWeb shards; found $($shards.Count)." }
$trainingData = (($shards | Select-Object -First 14 | ForEach-Object FullName) -join ',')
$validationData = ($shards | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Path $runLogDirectory -Force | Out-Null
$stdoutLog = Join-Path $runLogDirectory 'training_stdout.log'
$stderrLog = Join-Path $runLogDirectory 'training_stderr.log'
$launchManifest = Join-Path $runLogDirectory 'launch_manifest.json'

$arguments = @(
    '-u', '-m', 'scripts.train',
    '--config', $config,
    '--data', $trainingData,
    '--val-data', $validationData,
    '--dataset-kind', 'streaming_document'
)

$manifest = [ordered]@{
    launched_at_local = (Get-Date).ToString('o')
    initialization = 'random; no --resume and no --load-model'
    python = $python
    config = (Resolve-Path $config).Path
    training_shards = @($shards | Select-Object -First 14 | ForEach-Object FullName)
    validation_shard = $validationData
    command_arguments = $arguments
    stdout_log = $stdoutLog
    stderr_log = $stderrLog
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $launchManifest -Encoding utf8

# Start-Process creates a separate hidden process with independent redirection.
# It is intentionally not attached to this task's terminal, so closing a visible
# terminal or disconnecting this desktop-control session cannot send it a close event.
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

Start-Sleep -Seconds 5
if ($process.HasExited) {
    $exitCode = $process.ExitCode
    $details = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 60 | Out-String } else { '' }
    throw "Protected production run exited immediately with code $exitCode.`n$details"
}

Write-Output ("Protected random-initialization production pretraining started. PID: {0}" -f $process.Id)
Write-Output ("Logs: {0}" -f $runLogDirectory)
Write-Output ("Checkpoints: {0}" -f $runCheckpointDirectory)
