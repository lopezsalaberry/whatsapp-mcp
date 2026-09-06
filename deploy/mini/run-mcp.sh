#!/bin/bash
# Arranca la capa MCP (Python) sobre el bridge. Lo llama launchd.
set -euo pipefail
HOME_DIR="$HOME/.whatsapp-mcp"
REPO="$HOME/base/code/whatsapp-mcp/whatsapp-mcp-server"
[ -f "$HOME_DIR/env" ] && set -a && . "$HOME_DIR/env" && set +a
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"
export WHATSAPP_API_URL="${WHATSAPP_API_URL:-http://127.0.0.1:${WHATSAPP_BRIDGE_PORT:-8814}/api}"
export WHATSAPP_DB_PATH="${WHATSAPP_DB_PATH:-$HOME_DIR/store/messages.db}"
export WHATSMEOW_DB_PATH="${WHATSMEOW_DB_PATH:-$HOME_DIR/store/whatsapp.db}"
export WA_MCP_PORT="${WA_MCP_PORT:-8804}"
export WA_STATE_DB="${WA_STATE_DB:-$HOME_DIR/state.db}"
export WA_OUTBOX="${WA_OUTBOX:-$HOME_DIR/outbox}"
cd "$REPO"
exec uv run --project "$REPO" --extra base python base_server.py
