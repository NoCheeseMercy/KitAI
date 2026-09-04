<#
.SYNOPSIS
Exports a validated stopped KitAI protected-pretraining checkpoint to HF and F16 GGUF.

.DESCRIPTION
This produces a clearly labeled intermediate diagnostic artifact only. It does
not start training, alter the checkpoint, register an Ollama model, or run a
chat test. The source checkpoint remains the required full-state --resume
artifact for later protected-pretraining continuation.
#>

[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Checkpoint
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$projectRoot = (Resolve-Path (Join-Path $PSScriptRoot '..')).Path
Set-Location $projectRoot
$python = 'C:\Users\abdel\AppData\Local\Programs\Python\Python314\python.exe'
$checkpointRoot = (Resolve-Path '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun').Path
$tokenizer = (Resolve-Path '.\tokenizer\kitai_200m_chat_32k.json').Path

if (-not (Test-Path $python)) { throw "Python executable not found: $python" }
$checkpointItem = Get-Item (Resolve-Path $Checkpoint -ErrorAction Stop) -ErrorAction Stop
if (-not $checkpointItem.FullName.StartsWith($checkpointRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing a non-protected checkpoint: $($checkpointItem.FullName)"
}
if ($checkpointItem.Name -notmatch '^checkpoint_step_(\d{8})\.pt$') {
    throw "Refusing an unrecognized checkpoint filename: $($checkpointItem.Name)"
}
if ($checkpointItem.Length -lt 2000000000) {
    throw "Checkpoint is suspiciously small ($($checkpointItem.Length) bytes)."
}

$step = [int64]$Matches[1]
$metrics = Join-Path $checkpointRoot ("metrics_step_{0:D8}.json" -f $step)
if (-not (Test-Path $metrics)) {
    throw "Missing metrics companion for the proposed source checkpoint: $metrics"
}

$tag = ('step-{0:D8}' -f $step)
$exportRoot = Join-Path $projectRoot ("exports\kitai_200m_protected_rerun_{0}_stopped_intermediate" -f $tag)
$hfOutput = Join-Path $exportRoot 'hf'
$ggufOutput = Join-Path $exportRoot ("kitai_200m_protected_rerun_{0}_f16.gguf" -f $tag)
$logDirectory = Join-Path $exportRoot 'logs'
if (Test-Path $exportRoot) {
    throw "Refusing to overwrite an existing export: $exportRoot"
}
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

@"
# KitAI stopped-run intermediate export

This directory was exported from the genuine **protected pretraining step $step**
checkpoint after the active recovery was stopped. It is an intermediate
pretraining-only diagnostic artifact, **not a final or chat-ready KitAI model**.
It has no final assistant-only SFT and may be incoherent or unsuitable for chat.

The source `.pt` checkpoint remains unchanged and contains the model, optimizer,
scheduler, AMP state, global step, and epoch required for later continuation via
`--resume`. Do not use this export as the source for resumed pretraining.
"@ | Set-Content -Path (Join-Path $exportRoot 'INTERMEDIATE_ARTIFACT_ONLY.md') -Encoding utf8

$metadata = [ordered]@{
    exported_at_local = (Get-Date).ToString('o')
    source_checkpoint = $checkpointItem.FullName
    source_checkpoint_bytes = $checkpointItem.Length
    source_step = $step
    source_metrics = (Resolve-Path $metrics).Path
    hf_output = $hfOutput
    gguf_output = $ggufOutput
    status = 'intermediate pretraining-only export; no Ollama import or chat test was run'
}
$metadata | ConvertTo-Json -Depth 3 | Set-Content -Path (Join-Path $exportRoot 'export_manifest.json') -Encoding utf8

$exportLog = Join-Path $logDirectory 'hf_export.log'
$convertLog = Join-Path $logDirectory 'gguf_convert.log'
$savedPreference = $ErrorActionPreference

$ErrorActionPreference = 'Continue'
& $python -u .\scripts\export_checkpoint_to_hf.py --checkpoint $checkpointItem.FullName --tokenizer $tokenizer --output-dir $hfOutput 2>&1 | Tee-Object -FilePath $exportLog
$exportExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($exportExit -ne 0) { throw "Hugging Face export failed with exit code $exportExit" }
if (-not (Test-Path (Join-Path $hfOutput 'model.safetensors'))) {
    throw 'Hugging Face export is missing model.safetensors.'
}

$ErrorActionPreference = 'Continue'
& $python -u .\tools\llama.cpp\convert_hf_to_gguf.py $hfOutput --outfile $ggufOutput --outtype f16 2>&1 | Tee-Object -FilePath $convertLog
$convertExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($convertExit -ne 0) { throw "GGUF conversion failed with exit code $convertExit" }
if (-not (Test-Path $ggufOutput) -or (Get-Item $ggufOutput).Length -lt 100MB) {
    throw "GGUF output is missing or implausibly small: $ggufOutput"
}

[pscustomobject]@{
    source_checkpoint = $checkpointItem.FullName
    source_step = $step
    hf_directory = (Resolve-Path $hfOutput).Path
    gguf_file = (Resolve-Path $ggufOutput).Path
    gguf_bytes = (Get-Item $ggufOutput).Length
    export_manifest = (Resolve-Path (Join-Path $exportRoot 'export_manifest.json')).Path
    note = 'Intermediate pretraining-only export. No Ollama import or chat test was run.'
} | Format-List
