<#!
.SYNOPSIS
Exports one genuine protected-run pretraining checkpoint and performs a clearly
labeled local Ollama smoke test.

.DESCRIPTION
This is a test-artifact procedure only. It does not stop, resume, alter, or
replace the active production trainer. It refuses old/proof checkpoints,
existing export destinations, and checkpoint files that appear incomplete.
The resulting Ollama model name includes the exact checkpoint step and must
never be presented as the final KitAI model.
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

if (-not (Test-Path $python)) { throw "Python executable not found: $python" }
if (-not (Test-Path $ollama)) { throw "Ollama executable not found: $ollama" }

$checkpointPath = Get-Item (Resolve-Path $Checkpoint -ErrorAction Stop) -ErrorAction Stop
if (-not $checkpointPath.FullName.StartsWith($checkpointRoot, [System.StringComparison]::OrdinalIgnoreCase)) {
    throw "Refusing a non-protected-run checkpoint: $($checkpointPath.FullName)"
}
if ($checkpointPath.Name -notmatch '^checkpoint_step_(\d{8})\.pt$') {
    throw "Refusing an unrecognized checkpoint filename: $($checkpointPath.Name)"
}
if ($checkpointPath.Length -lt 100MB) {
    throw "Checkpoint is too small to be a completed 197.6M production checkpoint."
}

$step = [int]$Matches[1]
$tag = ('step-{0:D8}' -f $step)
$exportRoot = Join-Path $projectRoot ("exports\kitai_200m_protected_rerun_{0}_test" -f $tag)
$hfOutput = Join-Path $exportRoot 'hf'
$ggufOutput = Join-Path $exportRoot ("kitai_200m_protected_rerun_{0}_f16.gguf" -f $tag)
$logDirectory = Join-Path $exportRoot 'logs'
$modelName = ("kitai-200m-scratch-test-{0}" -f $tag)

if (Test-Path $exportRoot) {
    throw "Refusing to overwrite existing test export directory: $exportRoot"
}
New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null

$record = @"
# KitAI checkpoint smoke-test artifact

This export came from **production pretraining checkpoint $step** of the protected
random-initialization rerun. It is intentionally an intermediate diagnostic model,
not the final KitAI release. It has not received the required final assistant-only
chat-supervision phase and may produce incoherent or non-chat-like text.

Do not overwrite it, do not call it final, and do not use it as a source of
weights for later training phases.
"@
$record | Set-Content -Path (Join-Path $exportRoot 'TEST_ARTIFACT_ONLY.md') -Encoding utf8

$exportLog = Join-Path $logDirectory 'hf_export.log'
$convertLog = Join-Path $logDirectory 'gguf_convert.log'
$ollamaCreateLog = Join-Path $logDirectory 'ollama_create.log'
$ollamaTestLog = Join-Path $logDirectory 'ollama_hello_test.log'

Write-Host "Exporting genuine protected-run checkpoint step $step as a test artifact."
$savedPreference = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $python -u .\scripts\export_checkpoint_to_hf.py --checkpoint $checkpointPath.FullName --tokenizer $tokenizer --output-dir $hfOutput 2>&1 | Tee-Object -FilePath $exportLog
$exportExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($exportExit -ne 0) { throw "Hugging Face export failed with exit code $exportExit" }

$ErrorActionPreference = 'Continue'
& $python -u .\tools\llama.cpp\convert_hf_to_gguf.py $hfOutput --outfile $ggufOutput --outtype f16 2>&1 | Tee-Object -FilePath $convertLog
$convertExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($convertExit -ne 0) { throw "GGUF conversion failed with exit code $convertExit" }
if (-not (Test-Path $ggufOutput) -or (Get-Item $ggufOutput).Length -lt 100MB) {
    throw "GGUF output is missing or implausibly small: $ggufOutput"
}

$modelfile = Join-Path $exportRoot 'Modelfile'
@"
FROM $ggufOutput
PARAMETER num_ctx 256
PARAMETER temperature 0.7
PARAMETER top_p 0.9
TEMPLATE """{{ if .System }}<bos><|system|>{{ .System }}<|end|>
{{ end }}{{ range .Messages }}<|{{ .Role }}|>{{ .Content }}<|end|>
{{ end }}<|assistant|>"""
"@ | Set-Content -Path $modelfile -Encoding utf8

$ErrorActionPreference = 'Continue'
& $ollama create $modelName -f $modelfile 2>&1 | Tee-Object -FilePath $ollamaCreateLog
$ollamaCreateExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($ollamaCreateExit -ne 0) { throw "Ollama import failed with exit code $ollamaCreateExit" }

$ErrorActionPreference = 'Continue'
& $ollama run $modelName 'Hello' 2>&1 | Tee-Object -FilePath $ollamaTestLog
$ollamaTestExit = $LASTEXITCODE
$ErrorActionPreference = $savedPreference
if ($ollamaTestExit -ne 0) { throw "Ollama smoke test failed with exit code $ollamaTestExit" }

Write-Host "Test export and smoke test complete. Intermediate Ollama model: $modelName"
Write-Host "Artifact directory: $exportRoot"
