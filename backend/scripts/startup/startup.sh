#!/usr/bin/env bash
# Startup script to cache OpenRouter models

echo "🔄 Caching OpenRouter models..."

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../../.." && pwd)"
cd "$REPO_ROOT"

# Run the cache script
python3 backend/scripts/cache_openrouter_models.py

if [ $? -eq 0 ]; then
    echo "✓ Model cache updated"
else
    echo "⚠ Failed to cache models (continuing anyway)"
fi

# Continue with normal startup
echo "✓ Starting application..."

