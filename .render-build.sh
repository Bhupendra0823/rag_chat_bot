#!/usr/bin/env bash
set -e

echo "🚀 Starting Render build..."

# Install dependencies
pip install -r requirements.txt

# Install Playwright with specific browsers
echo "🎭 Installing Playwright browsers..."
playwright install chromium
playwright install firefox  # Optional

# Set proper permissions
export PLAYWRIGHT_BROWSERS_PATH=/opt/render/.cache/ms-playwright

# Create directories
mkdir -p /tmp/data /tmp/logs

echo "✅ Build complete!"