#!/bin/bash
# ─── NanoBot Daily Brief Runner ──────────────────────────────────────────────
# Called by launchd (com.nanobot.dailybrief) every day at 08:00.
# 1. Loads .env for API keys
# 2. Ensures Ollama is running (chat LLM + embeddings)
# 3. Runs the Python pipeline
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

PROJECT_DIR="/Users/yangyang/nanobot_project"
LOG_FILE="${PROJECT_DIR}/logs/cron_run.log"

# Timestamp helper
ts() { date "+%Y-%m-%d %H:%M:%S"; }

echo "[$(ts)] === Daily brief run started ===" >> "$LOG_FILE"

# ── 1. Load environment variables ────────────────────────────────────────────
export PATH="/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin:/usr/sbin:/sbin:$PATH"
set -a
source "${PROJECT_DIR}/.env"
set +a

# ── 2. Ensure Ollama is running (chat LLM + embeddings) ─────────────────────
OLLAMA_BIN="/usr/local/bin/ollama"

if ! pgrep -xq "ollama"; then
    echo "[$(ts)] Ollama not running, starting..." >> "$LOG_FILE"

    if [ -d "/Applications/Ollama.app" ]; then
        open -a "Ollama" 2>/dev/null
    elif [ -x "$OLLAMA_BIN" ]; then
        "$OLLAMA_BIN" serve &>/dev/null &
    else
        echo "[$(ts)] WARNING: Cannot find Ollama binary, embedding fallback will be used" >> "$LOG_FILE"
    fi

    # Wait up to 30s for Ollama to become healthy
    for i in $(seq 1 15); do
        if curl -s --max-time 2 http://localhost:11434/ >/dev/null 2>&1; then
            echo "[$(ts)] Ollama is ready (waited ${i}x2s)" >> "$LOG_FILE"
            break
        fi
        sleep 2
    done

    if ! curl -s --max-time 2 http://localhost:11434/ >/dev/null 2>&1; then
        echo "[$(ts)] WARNING: Ollama did not start within 30s" >> "$LOG_FILE"
    fi
else
    echo "[$(ts)] Ollama already running." >> "$LOG_FILE"
fi

# ── 3. Run the pipeline ─────────────────────────────────────────────────────
cd "${PROJECT_DIR}/app" || exit 1
source "${PROJECT_DIR}/.venv/bin/activate"

# 3a. Feed: push curated prediction-market + news + community items into the
#     #prediction-markets Discord channel. The daily_job's community pipeline
#     reads this channel, so the feed must land first.
echo "[$(ts)] Running prediction_markets_feed..." >> "$LOG_FILE"
python prediction_markets_feed.py real >> "$LOG_FILE" 2>&1 || {
    echo "[$(ts)] WARNING: prediction_markets_feed failed, continuing anyway" >> "$LOG_FILE"
}

# 3b. Daily brief
python daily_job.py >> "$LOG_FILE" 2>&1
EXIT_CODE=$?

echo "[$(ts)] === Daily brief run finished (exit=$EXIT_CODE) ===" >> "$LOG_FILE"
exit $EXIT_CODE
