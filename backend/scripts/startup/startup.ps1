# Startup script to cache OpenRouter models (PowerShell)

$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..\..\..")
Set-Location $RepoRoot

Write-Host "🔄 Caching OpenRouter models..."

# Run the cache script
python backend/scripts/cache_openrouter_models.py

if ($LASTEXITCODE -eq 0) {
    Write-Host "✓ Model cache updated"
} else {
    Write-Host "⚠ Failed to cache models (continuing anyway)"
}

Write-Host "✓ Ready to start application"

