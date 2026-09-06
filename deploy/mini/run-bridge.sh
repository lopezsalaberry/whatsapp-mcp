#!/bin/bash
# Arranca el bridge Go (whatsmeow) en modo servicio. Lo llama launchd.
# Config no secreta en ~/.whatsapp-mcp/env (600). Todo el estado en ~/.whatsapp-mcp/store.
set -euo pipefail
HOME_DIR="$HOME/.whatsapp-mcp"
[ -f "$HOME_DIR/env" ] && set -a && . "$HOME_DIR/env" && set +a
cd "$HOME_DIR"                          # el bridge escribe store/ relativo al cwd
export WHATSAPP_SERVICE_MODE=1          # nunca pide QR/código por sí solo
export WHATSAPP_BRIDGE_PORT="${WHATSAPP_BRIDGE_PORT:-8814}"
export WHATSAPP_DEVICE_NAME="${WHATSAPP_DEVICE_NAME:-base-mcp}"
export WHATSAPP_MEDIA_ROOTS="${WHATSAPP_MEDIA_ROOTS:-$HOME_DIR/outbox:$HOME_DIR/store}"
export WEBHOOK_ENABLED="${WEBHOOK_ENABLED:-false}"   # sin hub: no forwardear a nadie
exec "$HOME_DIR/bin/wa-bridge" --service
