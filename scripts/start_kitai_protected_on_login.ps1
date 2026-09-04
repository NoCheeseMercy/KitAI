<#
.SYNOPSIS
Starts KitAI protected pretraining at Windows login using the newest valid checkpoint.

.DESCRIPTION
This launcher is intended for Windows Task Scheduler at user logon. It dynamically
selects the newest full protected checkpoint with a companion metrics file, checks
that the validated 15-shard FineWeb-Edu inventory is present, and launches the
existing trainer with --resume. It refuses to start a duplicate KitAI trainer.
It never uses --load-model and never changes model parameters.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$python = 'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe'
$config = Join-Path $projectRoot 'configs\kitai_200m_scratch_pretrain_protected_rerun.yaml'
$dataDirectory = Join-Path $projectRoot 'datasets\kitai_200m_scratch\pretrain_fineweb_edu'
$validationReport = Join-Path $projectRoot 'datasets\kitai_200m_scratch\validation_report.json'
$checkpointDirectory = Join-Path $projectRoot 'checkpoints\kitai_200m_scratch_pretrain_protected_rerun'
$startupLogRoot = Join-Path $projectRoot 'logs\kitai_startup_launcher'

New-Item -ItemType Directory -Path $startupLogRoot -Force | Out-Null
$timestamp = Get-Date -Format 'yyyyMMdd_HHmmss'
$stdoutLog = Join-Path $startupLogRoot "training_stdout_$timestamp.log"
$stderrLog = Join-Path $startupLogRoot "training_stderr_$timestamp.log"
$manifestPath = Join-Path $startupLogRoot "launch_manifest_$timestamp.json"
$launcherLog = Join-Path $startupLogRoot 'launcher.log'

function Write-LauncherLog([string]$Message) {
    "$(Get-Date -Format o) | $Message" | Tee-Object -FilePath $launcherLog -Append
}

if (-not (Test-Path -LiteralPath $python)) { throw "Python executable not found: $python" }
foreach ($required in @($config, $validationReport, $dataDirectory, $checkpointDirectory)) {
    if (-not (Test-Path -LiteralPath $required)) { throw "Required path not found: $required" }
}

$validation = Get-Content -LiteralPath $validationReport -Raw | ConvertFrom-Json
if (-not $validation.passed) { throw 'Validated production data report does not declare passed=true.' }

$activeTraining = @(Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" | Where-Object {
    $_.CommandLine -match 'scripts\.train' -and
    $_.CommandLine -match 'kitai_200m_scratch_pretrain_protected_rerun\.yaml'
})
if ($activeTraining.Count -gt 0) {
    Write-LauncherLog "Refusing duplicate launch; active KitAI trainer PID(s): $($activeTraining.ProcessId -join ', ')"
    exit 0
}

$candidates = @(Get-ChildItem -LiteralPath $checkpointDirectory -File -Filter 'checkpoint_step_*.pt' | Where-Object {
    $_.BaseName -match '^checkpoint_step_(\d{8})$' -and
    $_.Length -ge 2200000000
} | ForEach-Object {
    $step = [int64]$Matches[1]
    $metrics = Join-Path $checkpointDirectory ("metrics_step_{0:D8}.json" -f $step)
    if ((Test-Path -LiteralPath $metrics) -and ((Get-Item -LiteralPath $metrics).Length -gt 0)) {
        [pscustomobject]@{ Item = $_; Step = $step; Metrics = $metrics }
    }
} | Sort-Object Step -Descending)
if ($candidates.Count -eq 0) { throw 'No valid protected full-state checkpoint with metrics was found.' }
$checkpoint = $candidates[0]

$shards = @(Get-ChildItem -LiteralPath $dataDirectory -Filter 'fineweb_edu_*.txt' -File | Sort-Object Name)
if ($shards.Count -ne 15) { throw "Expected exactly 15 preserved FineWeb shards; found $($shards.Count)." }
$trainingData = (($shards | Select-Object -First 14 | ForEach-Object FullName) -join ',')
$validationData = ($shards | Select-Object -Last 1).FullName
$arguments = @(
    '-u', '-m', 'scripts.train',
    '--config', $config,
    '--data', $trainingData,
    '--val-data', $validationData,
    '--dataset-kind', 'streaming_document',
    '--resume', $checkpoint.Item.FullName
)

[ordered]@{
    launched_at_local = (Get-Date).ToString('o')
    trigger = 'Windows logon/startup'
    recovery_mode = 'full-state protected continuation via --resume'
    source_checkpoint = $checkpoint.Item.FullName
    source_checkpoint_bytes = $checkpoint.Item.Length
    source_step = $checkpoint.Step
    source_metrics = $checkpoint.Metrics
    config = $config
    training_shards = @($shards | Select-Object -First 14 | ForEach-Object FullName)
    validation_shard = $validationData
    command_arguments = $arguments
    prohibited_options_absent = @('--load-model')
} | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $manifestPath -Encoding utf8

Write-LauncherLog "Launching protected trainer from step $($checkpoint.Step) with --resume."
$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -WindowStyle Hidden -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog -PassThru
Start-Sleep -Seconds 20
if ($process.HasExited) {
    $details = if (Test-Path -LiteralPath $stderrLog) { Get-Content -LiteralPath $stderrLog -Tail 100 | Out-String } else { '' }
    throw "Startup trainer exited immediately with code $($process.ExitCode).`n$details"
}
Write-LauncherLog "Started protected trainer PID $($process.Id)."
Write-Output "STARTED_PID=$($process.Id)"
Write-Output "SOURCE_STEP=$($checkpoint.Step)"
Write-Output "MANIFEST=$manifestPath"
