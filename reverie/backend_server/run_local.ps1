# run_local.ps1 -- set up the local-model environment and verify it.
#
#   .\run_local.ps1                            # preflight with the default model
#   .\run_local.ps1 -Model llama3.2:3b         # preflight with a specific model
#   .\run_local.ps1 -Model llama3.2:3b -Run    # preflight, then start reverie.py
#   .\run_local.ps1 -Sweep                     # validity across the size ladder
#
# Every variable set here is scoped to this PowerShell session. Close the window
# and the repo is back to its OpenAI configuration.

param(
    [string]$Model      = "qwen2.5:0.5b",
    [string]$EmbedModel = "nomic-embed-text",
    [string]$BaseUrl    = "http://localhost:11434/v1",
    [int]$Repeat        = 5,
    [switch]$Run,
    [switch]$Sweep
)

$ErrorActionPreference = "Stop"

$env:CRSEC_LLM_BACKEND       = "local"
$env:CRSEC_LOCAL_BASE_URL    = $BaseUrl
$env:CRSEC_LOCAL_CHAT_MODEL  = $Model
$env:CRSEC_LOCAL_EMBED_MODEL = $EmbedModel
$env:OPENAI_API_KEY          = "local"

# Reloading a model between calls costs seconds each time, thousands of times.
$env:OLLAMA_KEEP_ALIVE = "-1"

Write-Host "CRSEC local backend" -ForegroundColor Cyan
Write-Host "  server : $BaseUrl"
Write-Host "  chat   : $Model"
Write-Host "  embed  : $EmbedModel"
Write-Host ""

if (-not (Test-Path "local_llm\preflight.py")) {
    Write-Host "Run this from reverie\backend_server (local_llm\ must be here)." -ForegroundColor Red
    exit 1
}

# Offline tests first: they need no server, so a failure here is unambiguously
# a code problem rather than a configuration problem.
Write-Host "Offline tests..." -ForegroundColor Cyan
python local_llm\test_local_backend.py
if ($LASTEXITCODE -ne 0) {
    Write-Host "Offline tests failed. Fix before touching the server." -ForegroundColor Red
    exit 1
}
Write-Host ""

if ($Sweep) {
    # Validity against model size. Pull each tag first; missing ones are skipped.
    $ladder = @("qwen2.5:0.5b", "qwen3.5:0.8b", "llama3.2:1b", "llama3.2:3b", "qwen3.5:9b")
    $installed = (ollama list) -join "`n"
    foreach ($m in $ladder) {
        if ($installed -notmatch [regex]::Escape($m.Split(":")[0])) {
            Write-Host "skip $m (not pulled)" -ForegroundColor DarkGray
            continue
        }
        Write-Host ("=" * 68) -ForegroundColor Cyan
        Write-Host "  $m" -ForegroundColor Cyan
        Write-Host ("=" * 68) -ForegroundColor Cyan
        $env:CRSEC_LOCAL_CHAT_MODEL = $m
        python local_llm\preflight.py --repeat $Repeat
        Write-Host ""
    }
    exit 0
}

python local_llm\preflight.py --repeat $Repeat
$preflight = $LASTEXITCODE

if ($preflight -ne 0) {
    Write-Host "Preflight failed. Not starting the simulation." -ForegroundColor Red
    exit $preflight
}

if ($Run) {
    Write-Host ""
    Write-Host "Starting reverie.py against $Model" -ForegroundColor Cyan
    Write-Host "Local inference is slow. A full day is thousands of sequential calls." -ForegroundColor Yellow
    python reverie.py
}
