<#
.SYNOPSIS
Exports a completed protected KitAI pretraining checkpoint to Ollama while the
separate production trainer continues running.

.DESCRIPTION
This procedure is deliberately inference/export-only: it never stops, resumes,
reconfigures, or writes to the active trainer, its source checkpoint, or its
optimizer state. It refuses incomplete, non-protected, or duplicate artifacts.
It imports an explicitly labeled intermediate pretraining model and performs no
text-generation smoke test.
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
$ollama = 'C:\Users\abdel\AppData\Local\Programs\Ollama\ollama.exe'
$checkpointRoot = (Resolve-Path '.\checkpoints\kitai_200m_scratch_pretrain_protected_rerun').Path
$tokenizer = (Resolve-Path '.\tokenizer\kitai_200m_chat_32k.json').Path

if (-not (Test-Path -LiteralPath $python)) { throw "Python executable not found: $python" }
if (-not (Test-Path -LiteralPath $ollama)) { throw "Ollama executable not found: $ollama" }

$checkpointPath = Get-Item -LiteralPath (Resolve-Path -LiteralPath $Checkpoint -ErrorAction Stop) -ErrorAction Stop
if (-not $checkpointPath.FullName.StartsWith($checkpointRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing a non-protected checkpoint: $($checkpointPath.FullName)"
}
if ($checkpointPath.Name -notmatch '^checkpoint_step_(\d{8})\.pt$') {
    throw "Refusing an unrecognized checkpoint filename: $($checkpointPath.Name)"
}
if ($checkpointPath.Length -lt 2200000000) {
    throw "Checkpoint is too small to be a fully written 197.6M protected checkpoint."
}

$step = [int]$Matches[1]
$metricsName = (($checkpointPath.Name -replace '^checkpoint_', 'metrics_') -replace '\.pt$', '.json')
$metricsPath = Join-Path $checkpointRoot $metricsName
if (-not (Test-Path -LiteralPath $metricsPath)) {
    throw "Companion metrics artifact is missing: $metricsPath"
}

$tag = ('step-{0:D8}' -f $step)
$exportRoot = Join-Path $projectRoot ("exports\kitai_200m_protected_rerun_{0}_live_intermediate" -f $tag)
$hfOutput = Join-Path $exportRoot 'hf'
$ggufOutput = Join-Path $exportRoot ("kitai_200m_protected_rerun_{0}_f16.gguf" -f $tag)
$logDirectory = Join-Path $exportRoot 'logs'
$modelName = ("kitai-200m-scratch-intermediate-{0}" -f $tag)

if (Test-Path -LiteralPath $exportRoot) {
    throw "Refusing to overwrite existing export directory: $exportRoot"
}
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

@"
# KitAI intermediate pretraining export

- **Source:** protected full-state checkpoint $($checkpointPath.Name)
- **Checkpoint bytes:** $($checkpointPath.Length)
- **Companion metrics:** $metricsName
- **Ollama model:** `$modelName`
- **Export policy:** performed while the independent trainer remained active; no
  training state, source checkpoint, configuration, or dataset was changed.

This is an **intermediate pretraining-only artifact**, not final KitAI. It has
not had final assistant-only chat supervision and must not be used to resume
training or represented as a normal chat-ready model.
"@ | Set-Content -LiteralPath (Join-Path $exportRoot 'INTERMEDIATE_ONLY.md') -Encoding utf8

$exportLog = Join-Path $logDirectory 'hf_export.log'
$convertLog = Join-Path $logDirectory 'gguf_convert.log'
$ollamaCreateLog = Join-Path $logDirectory 'ollama_create.log'

$oldPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -u .\scripts\export_checkpoint_to_hf.py --checkpoint $checkpointPath.FullName --tokenizer $tokenizer --output-dir $hfOutput 2>&1 | Tee-Object -FilePath $exportLog
$exportExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
if ($exportExit -ne 0) { throw "Hugging Face export failed with exit code $exportExit" }

$ErrorActionPreference = 'Continue'
& $python -u .\tools\llama.cpp\convert_hf_to_gguf.py $hfOutput --outfile $ggufOutput --outtype f16 2>&1 | Tee-Object -FilePath $convertLog
$convertExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
if ($convertExit -ne 0) { throw "GGUF conversion failed with exit code $convertExit" }
if (-not (Test-Path -LiteralPath $ggufOutput) -or (Get-Item -LiteralPath $ggufOutput).Length -lt 100MB) {
    throw "GGUF output is missing or implausibly small: $ggufOutput"
}

$modelfile = Join-Path $exportRoot 'Modelfile'
@"
FROM $ggufOutput
PARAMETER num_ctx 256
PARAMETER temperature 0.7
PARAMETER top_p 0.9
TEMPLATE """{{ .Prompt }}"""
"@ | Set-Content -LiteralPath $modelfile -Encoding utf8

$ErrorActionPreference = 'Continue'
& $ollama create $modelName -f $modelfile 2>&1 | Tee-Object -FilePath $ollamaCreateLog
$ollamaCreateExit = $LASTEXITCODE
$ErrorActionPreference = $oldPreference
if ($ollamaCreateExit -ne 0) { throw "Ollama import failed with exit code $ollamaCreateExit" }

@{
    source_checkpoint = $checkpointPath.FullName
    source_checkpoint_bytes = $checkpointPath.Length
    source_metrics = $metricsPath
    checkpoint_step = $step
    ollama_model = $modelName
    export_kind = 'live intermediate pretraining-only'
    no_training_mutation = $true
    no_generation_test = $true
    created_at = (Get-Date).ToString('o')
} | ConvertTo-Json | Set-Content -LiteralPath (Join-Path $exportRoot 'export_manifest.json') -Encoding utf8

Write-Host "Export and Ollama import complete: $modelName"
Write-Host "Intermediate artifact directory: $exportRoot"
