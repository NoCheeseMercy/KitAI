<#
.SYNOPSIS
Stops the active KitAI protected-pretraining recovery only after verifying that a
full, resumable protected checkpoint already exists.

.DESCRIPTION
The active Windows recovery trainer is detached and does not register a Windows
signal handler that would create a new checkpoint on Ctrl+C. This helper first
verifies the latest existing full-state checkpoint and metrics artifact, then
attempts a console CTRL_BREAK for a clean Python shutdown. If the detached
process does not exit, it performs a controlled termination only after writing
a stop manifest documenting the verified resume source. Any in-memory work
since the last checkpoint is intentionally discarded; the verified checkpoint
remains intact for a later --resume continuation.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$checkpointDirectory = Join-Path $projectRoot 'checkpoints\kitai_200m_scratch_pretrain_protected_rerun'
$auditDirectory = Join-Path $projectRoot 'logs\kitai_200m_scratch_pretrain_protected_rerun_resume_from_00002000_attempt_02'

$active = Get-CimInstance Win32_Process -Filter "Name = 'python.exe'" |
    Where-Object {
        $_.CommandLine -match 'scripts\.train' -and
        $_.CommandLine -match 'kitai_200m_scratch_pretrain_protected_rerun\.yaml' -and
        $_.CommandLine -match '\-\-resume'
    }
if (-not $active) {
    throw 'No active protected KitAI resume trainer was found; refusing to stop an unrelated process.'
}
if (@($active).Count -ne 1) {
    throw "Expected exactly one protected KitAI resume trainer; found $(@($active).Count)."
}

$checkpoint = Get-ChildItem $checkpointDirectory -File -Filter 'checkpoint_step_*.pt' |
    Sort-Object {
        if ($_.BaseName -match 'checkpoint_step_(\d+)$') { [int64]$Matches[1] } else { -1 }
    } -Descending |
    Select-Object -First 1
if (-not $checkpoint) {
    throw 'No protected checkpoint exists; refusing to stop because there is no resume source.'
}
if ($checkpoint.Length -lt 2000000000) {
    throw "Latest checkpoint is suspiciously small ($($checkpoint.Length) bytes); refusing to stop."
}
if ($checkpoint.BaseName -notmatch 'checkpoint_step_(\d+)$') {
    throw "Unexpected checkpoint name: $($checkpoint.Name)"
}
$checkpointStep = $Matches[1]
$metrics = Join-Path $checkpointDirectory ("metrics_step_{0}.json" -f $checkpointStep)
if (-not (Test-Path $metrics)) {
    throw "Missing companion metrics artifact: $metrics"
}

$manifestPath = Join-Path $auditDirectory ("controlled_stop_{0}.json" -f (Get-Date -Format 'yyyyMMdd_HHmmss'))
$manifest = [ordered]@{
    requested_at_local = (Get-Date).ToString('o')
    trainer_pid = [int]$active.ProcessId
    trainer_command = $active.CommandLine
    verified_resume_checkpoint = $checkpoint.FullName
    verified_resume_checkpoint_bytes = $checkpoint.Length
    verified_resume_step = [int64]$checkpointStep
    companion_metrics = (Resolve-Path $metrics).Path
    stop_method = $null
    note = 'The next run must use --resume with this protected checkpoint. In-memory work after this checkpoint is not durable.'
}

Add-Type @'
using System;
using System.Runtime.InteropServices;
public static class KitAIConsoleControl {
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool FreeConsole();
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool AttachConsole(uint processId);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool SetConsoleCtrlHandler(IntPtr handler, bool add);
    [DllImport("kernel32.dll", SetLastError=true)] public static extern bool GenerateConsoleCtrlEvent(uint controlEvent, uint processGroupId);
}
'@

$softStopSent = $false
try {
    [void][KitAIConsoleControl]::FreeConsole()
    if ([KitAIConsoleControl]::AttachConsole([uint32]$active.ProcessId)) {
        [void][KitAIConsoleControl]::SetConsoleCtrlHandler([IntPtr]::Zero, $true)
        $softStopSent = [KitAIConsoleControl]::GenerateConsoleCtrlEvent(1, 0)
    }
} catch {
    $softStopSent = $false
}

if ($softStopSent) {
    $manifest.stop_method = 'CTRL_BREAK sent to the detached trainer console; waiting for normal Python shutdown.'
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8
    Start-Sleep -Seconds 12
}

$stillActive = Get-Process -Id $active.ProcessId -ErrorAction SilentlyContinue
if ($stillActive) {
    $manifest.stop_method = if ($softStopSent) {
        'CTRL_BREAK did not stop the detached trainer within 12 seconds; controlled termination used after checkpoint verification.'
    } else {
        'Detached trainer console could not receive CTRL_BREAK; controlled termination used after checkpoint verification.'
    }
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8
    Stop-Process -Id $active.ProcessId -Force
    Start-Sleep -Seconds 5
} else {
    $manifest.stop_method = 'CTRL_BREAK produced a normal trainer process exit.'
    $manifest | ConvertTo-Json -Depth 4 | Set-Content -Path $manifestPath -Encoding utf8
}

if (Get-Process -Id $active.ProcessId -ErrorAction SilentlyContinue) {
    throw "Trainer PID $($active.ProcessId) is still active after the guarded stop attempt."
}

$checkpointAfter = Get-Item $checkpoint.FullName
if ($checkpointAfter.Length -ne $checkpoint.Length) {
    throw 'Verified resume checkpoint size changed during stop; inspect before resuming.'
}

[pscustomobject]@{
    stopped_trainer_pid = [int]$active.ProcessId
    verified_resume_checkpoint = $checkpointAfter.FullName
    verified_resume_step = [int64]$checkpointStep
    verified_resume_checkpoint_bytes = $checkpointAfter.Length
    companion_metrics = (Resolve-Path $metrics).Path
    stop_manifest = $manifestPath
    stop_method = $manifest.stop_method
} | Format-List
