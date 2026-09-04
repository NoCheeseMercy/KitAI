[CmdletBinding()]
param(
    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$Model,

    [Parameter(Mandatory = $true)]
    [ValidateNotNullOrEmpty()]
    [string]$OutputPath
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'

$prompts = @(
    'Hello',
    'Who are you?',
    'What is 2 + 2?',
    'Tell me a short joke.'
)

$results = foreach ($prompt in $prompts) {
    $body = @{
        model = $Model
        stream = $false
        messages = @(@{ role = 'user'; content = $prompt })
        options = @{
            num_ctx = 256
            num_predict = 64
            temperature = 0.7
            top_p = 0.9
        }
    } | ConvertTo-Json -Depth 8

    $response = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/chat' -Method Post -ContentType 'application/json' -Body $body -TimeoutSec 180
    [pscustomobject]@{
        prompt = $prompt
        response = [string]$response.message.content
        done_reason = [string]$response.done_reason
        eval_count = $response.eval_count
        eval_duration_ns = $response.eval_duration
    }
}

$destination = [System.IO.Path]::GetFullPath($OutputPath)
$directory = Split-Path -Parent $destination
if ($directory) { New-Item -ItemType Directory -Path $directory -Force | Out-Null }
$results | ConvertTo-Json -Depth 6 | Set-Content -Path $destination -Encoding utf8
$results | Format-Table -AutoSize
