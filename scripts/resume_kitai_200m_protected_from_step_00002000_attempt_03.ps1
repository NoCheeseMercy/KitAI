<#
.SYNOPSIS
Launches a third audited full-state recovery of KitAI protected pretraining.

.DESCRIPTION
This launcher resumes only the model's own verified protected step-2000 checkpoint
through --resume. It restores model, optimizer, scheduler, AMP scaler, step, and
epoch; it does not use --load-model and does not create random weights. Existing
logs and earlier recovery attempts remain untouched.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$python = 'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe'
$config = '.\configs\kitai_200m_scratch_pretrain_protected_rerun.yaml'
$dataDirectory = '.\datasets\kitai_200m_scratch\pretrain_fineweb_edu'
$validationReport = '.\datasets\kitai_200m_scratch\validation_report.json'
$checkpoint = '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun\checkpoint_step_00002000.pt'
$recoveryLogDirectory = '.\logs\kitai_200m_scratch_pretrain_protected_rerun_resume_from_00002000_attempt_03'

foreach ($path in @($python, $config, $validationReport, $checkpoint)) {
    if (-not (Test-Path $path)) { throw "Required path not found: $path" }
}
$validation = Get-Content $validationReport -Raw | ConvertFrom-Json
if (-not $validation.passed) { throw 'Validated production data report does not declare passed=true.' }
$checkpointItem = Get-Item $checkpoint
if ($checkpointItem.Length -lt 2000000000) {
    throw "Refusing suspiciously small protected checkpoint ($($checkpointItem.Length) bytes)."
}
if (Test-Path $recoveryLogDirectory) {
    throw "Refusing to overwrite existing attempt-03 audit logs: $recoveryLogDirectory"
}

$activeTraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object { $_.CommandLine -match 'scripts\.train' }
if ($activeTraining) {
    $ids = ($activeTraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Refusing to resume while a KitAI training process is active (PID(s): $ids)."
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

[ordered]@{
    launched_at_local = (Get-Date).ToString('o')
    recovery_mode = 'full-state protected continuation via --resume'
    source_checkpoint = (Resolve-Path $checkpoint).Path
    source_checkpoint_size_bytes = $checkpointItem.Length
    restored_state = @('model weights', 'optimizer', 'scheduler', 'AMP scaler', 'global step', 'epoch')
    prohibited_options_absent = @('--load-model')
    config = (Resolve-Path $config).Path
    training_shards = @($shards | Select-Object -First 14 | ForEach-Object FullName)
    validation_shard = $validationData
    command_arguments = $arguments
    milestone_requested_by_user = 9000
    prior_attempt_logs_preserved = @(
        (Resolve-Path '.\logs\kitai_200m_scratch_pretrain_protected_rerun_resume_from_00002000').Path,
        (Resolve-Path '.\logs\kitai_200m_scratch_pretrain_protected_rerun_resume_from_00002000_attempt_02').Path
    )
} | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
Start-Sleep -Seconds 20
if ($process.HasExited) {
    $details = if (Test-Path $stderrLog) { Get-Content $stderrLog -Tail 100 | Out-String } else { '' }
    throw "Attempt-03 recovery exited immediately with code $($process.ExitCode).`n$details"
}

[pscustomobject]@{
    recovery_pid = $process.Id
    recovery_log_directory = (Resolve-Path $recoveryLogDirectory).Path
    source_checkpoint = (Resolve-Path $checkpoint).Path
    source_step = 2000
    recovery_mode = 'full-state --resume only'
    target_milestone = 9000
} | Format-List
