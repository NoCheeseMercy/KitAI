<#!
.SYNOPSIS
Starts the prepared final assistant-only SFT phase for the real KitAI 197.6M
model after production pretraining has completed.

.DESCRIPTION
This script intentionally refuses to run while the production pretraining
process is still active. It requires an explicit production checkpoint path,
loads only the model weights with --load-model, streams the preserved filtered
Smol-SmolTalk JSONL without materializing it, and retains bounded final
user/assistant pairs for overlength records. It never uses --resume.
#>
[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$ProductionCheckpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot

$activePretraining = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object {
        $_.CommandLine -match 'kitai_200m_scratch_pretrain(?:_protected_rerun)?\.yaml' -and
        $_.CommandLine -match 'scripts\.train'
    }
if ($activePretraining) {
    $activeIds = ($activePretraining | Select-Object -ExpandProperty ProcessId) -join ', '
    throw "Refusing to start final SFT: production pretraining is still active (PID(s): $activeIds)."
}

$checkpointPath = Resolve-Path $ProductionCheckpoint -ErrorAction Stop
$checkpointRoot = (Resolve-Path '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun').Path
if (-not $checkpointPath.Path.StartsWith($checkpointRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing non-protected-production checkpoint: $($checkpointPath.Path) is not under $checkpointRoot"
}
if ((Get-Item $checkpointPath.Path).Length -le 0) {
    throw "Refusing empty production checkpoint: $($checkpointPath.Path)"
}

$python = 'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe'
if (-not (Test-Path $python)) {
    throw "Python executable not found: $python"
}

$config = '.\configs\kitai_200m_scratch_final_sft.yaml'
$chatData = '.\datasets\kitai_200m_scratch\chat_sft\smol_smoltalk_filtered.jsonl'
$logDir = '.\logs\kitai_200m_scratch_final_sft'
New-Item -ItemType Directory -Force -Path $logDir | Out-Null
$stdoutLog = Join-Path $logDir 'training_stdout.log'
$stderrLog = Join-Path $logDir 'training_stderr.log'

if ((Test-Path $stdoutLog) -or (Test-Path $stderrLog)) {
    throw "Refusing to overwrite existing SFT logs in $logDir. Archive or inspect them before intentionally starting a new SFT attempt."
}

$arguments = @(
    '-u', '-m', 'scripts.train',
    '--config', $config,
    '--data', $chatData,
    '--val-data', $chatData,
    '--dataset-kind', 'streaming_chat_sft',
    '--chat-overlength-strategy', 'final_pair',
    '--chat-max-prompt-tokens', '96',
    '--chat-holdout-modulus', '20',
    '--load-model', $checkpointPath.Path
)

Write-Host 'Launching final assistant-only SFT from the verified production checkpoint.'
Write-Host "Checkpoint: $($checkpointPath.Path)"
Write-Host 'Training stream: records with remainders 1-19 modulo 20; validation: remainder 0.'
Write-Host "Logs: $logDir"

$process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $projectRoot -NoNewWindow -PassThru -RedirectStandardOutput $stdoutLog -RedirectStandardError $stderrLog
Write-Host "Final SFT started with PID $($process.Id)."
