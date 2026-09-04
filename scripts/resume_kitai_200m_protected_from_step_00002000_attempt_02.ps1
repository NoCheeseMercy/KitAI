<#
.SYNOPSIS
Resumes the protected KitAI 197.6M production pretraining run from its verified
step-2000 checkpoint after an external interruption.

.DESCRIPTION
This recovery launcher restores the complete training state with --resume:
model weights, AdamW optimizer state, scheduler state, AMP scaler, global step,
and epoch. It does not create random weights, does not use --load-model, and
does not modify the original protected-run stdout/stderr logs. Recovery stdout
and stderr are written to a dedicated audit directory.
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
$checkpoint = '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun\checkpoint_step_00002000.pt'
$recoveryLogDirectory = '.\logs\kitai_200m_scratch_pretrain_protected_rerun_resume_from_00002000_attempt_02'

if (-not (Test-Path $python)) { throw "Python executable not found: $python" }
if (-not (Test-Path $config)) { throw "Protected production config not found: $config" }
if (-not (Test-Path $validationReport)) { throw "Validated production data report not found: $validationReport" }
if (-not (Test-Path $checkpoint)) { throw "Verified step-2000 checkpoint not found: $checkpoint" }

$validation = Get-Content $validationReport -Raw | ConvertFrom-Json
if (-not $validation.passed) { throw 'Validated production data report does not declare passed=true.' }

$checkpointItem = Get-Item $checkpoint
if ($checkpointItem.Length -lt 2000000000) {
    throw "Refusing suspiciously small step-2000 checkpoint ($($checkpointItem.Length) bytes)."
}

$activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts\.train' }
if ($activeTraining) {
    $activeIds = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Refusing to resume while a KitAI training command is already active (PID(s): $activeIds)."
}

if (Test-Path $recoveryLogDirectory) {
    throw "Refusing to overwrite existing recovery logs: $recoveryLogDirectory"
}

$shards = Get-ChildItem $dataDirectory -Filter 'fineweb_edu_*.txt' -File | Sort-Object Name
if ($shards.Count -ne 15) { throw "Expected exactly 15 preserved FineWeb shards; found $($shards.Count)." }
$trainingData = (($shards | Select-Object -First 14 | ForEach-Object FullName) -join ',')
$validationData = ($shards | Select-Object -Last 1).FullName

New-Item -ItemType Directory -Path $recoveryLogDirectory -Force | Out-Null
$stdoutLog = Join-Path $recoveryLogDirectory 'training_stdout.log'
$stderrLog = Join-Path $recoveryLogDirectory 'training_stderr.log'
$manifestPath = Join-Path $recoveryLogDirectory 'resume_manifest.json'

$arguments = @(
    '-u', '-m', 'scripts.train',
    '--config', $config,
    '--data', $trainingData,
    '--val-data', $validationData,
    '--dataset-kind', 'streaming_document',
    '--resume', (Resolve-Path $checkpoint).Path
)

$manifest = [ordered]@{
    launched_at_local = (Get-Date).ToString('o')
    recovery_mode = 'full-state checkpoint continuation via --resume'
    source_checkpoint = (Resolve-Path $checkpoint).Path
    source_checkpoint_size_bytes = $checkpointItem.Length
    restored_state = @('model weights', 'optimizer', 'scheduler', 'AMP scaler', 'global step', 'epoch')
    prohibited_options_absent = @('--load-model')
    python = $python
    config = (Resolve-Path $config).Path
    training_shards = @($shards | Select-Object -First 14 | ForEach-Object FullName)
    validation_shard = $validationData
    command_arguments = $arguments
    original_protected_stdout_preserved = (Resolve-Path '.\logs\kitai_200m_scratch_pretrain_protected_rerun\training_stdout.log').Path
    original_protected_stderr_preserved = (Resolve-Path '.\logs\kitai_200m_scratch_pretrain_protected_rerun\training_stderr.log').Path
    recovery_stdout = $stdoutLog
    recovery_stderr = $stderrLog
}
$manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8

# A hidden detached process prevents the prior visible-console close failure mode.
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru

Start-Sleep -Seconds 20
if ($process.HasExited) {
    $details = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 80 | Out-String } else { '' }
    throw "Protected checkpoint recovery exited immediately with code $($process.ExitCode).`n$details"
}

Write-Output ("Protected pretraining resumed from full state at step 2000. PID: {0}" -f $process.Id)
Write-Output ("Recovery logs: {0}" -f $recoveryLogDirectory)
Write-Output ("Checkpoint source: {0}" -f (Resolve-Path $checkpoint).Path)
