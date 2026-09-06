#!/bin/bash
# Instala whatsapp-mcp (bridge Go + capa MCP Python) como dos daemons launchd
# en el Mac mini. Idempotente. Correr EN EL MINI:
#   bash ~/base/code/whatsapp-mcp/deploy/mini/install.sh
# NO empareja: el pairing es un paso manual con Juan (ver README del deploy).
set -euo pipefail

REPO="$HOME/base/code/whatsapp-mcp"
HOME_DIR="$HOME/.whatsapp-mcp"
DEPLOY="$REPO/deploy/mini"
UID_="$(id -u)"
export PATH="$HOME/.local/bin:/opt/homebrew/bin:/usr/local/bin:/usr/bin:/bin"

echo "== 1/7 repo al día =="
[ -d "$REPO" ] || { echo "FALTA el repo: corré ~/base/bootstrap.sh"; exit 1; }
cd "$REPO" && git pull --ff-only -q || echo "WARN: pull falló (auth no interactiva) — verificar que el repo esté al día"

echo "== 2/7 directorios y env =="
mkdir -p "$HOME_DIR"/{store,outbox,logs,bin}
chmod 700 "$HOME_DIR"
if [ ! -f "$HOME_DIR/env" ]; then
  umask 077
  cat > "$HOME_DIR/env" <<ENV
# whatsapp-mcp — config local del mini (600). Los tokens son locales y regenerables.
WHATSAPP_BRIDGE_PORT=8814
WA_MCP_PORT=8804
WA_MCP_TOKEN=$(openssl rand -hex 32)
WA_SEND_MAX_PER_HOUR=30
WA_STT_MODEL=large-v3
WA_STT_LANGUAGE=es
ENV
  umask 022
  echo "env creado con WA_MCP_TOKEN nuevo — copiarlo a ~/.base-mcp/env como WA_MCP_TOKEN"
fi
chmod 600 "$HOME_DIR/env"

echo "== 3/7 build del bridge (Go) =="
command -v go >/dev/null || brew install -q go
(cd "$REPO/whatsapp-bridge" && go build -o "$HOME_DIR/bin/wa-bridge" .)
"$HOME_DIR/bin/wa-bridge" -h 2>&1 | grep -q -- "-service" || { echo "el binario no tiene --service: build viejo"; exit 1; }

echo "== 4/7 deps Python + tests =="
(cd "$REPO/whatsapp-mcp-server" && uv sync -q --extra dev --extra base && uv run pytest -q 2>&1 | tail -1)

echo "== 5/7 launchd =="
for label in com.whatsapp-bridge com.whatsapp-mcp; do
  launchctl bootout "gui/$UID_/$label" 2>/dev/null || true
  cp "$DEPLOY/$label.plist" "$HOME/Library/LaunchAgents/$label.plist"
  launchctl bootstrap "gui/$UID_" "$HOME/Library/LaunchAgents/$label.plist"
done
sleep 4

echo "== 6/7 health =="
set -a; . "$HOME_DIR/env"; set +a
TOK="$(cat "$HOME_DIR/store/.bridge-token" 2>/dev/null || true)"
curl -sf -H "Authorization: Bearer $TOK" "http://127.0.0.1:${WHATSAPP_BRIDGE_PORT}/api/health" && echo " ← bridge /api/health" \
  || echo "bridge /api/health FALLÓ — ver $HOME_DIR/logs/bridge.err.log"
curl -s -o /dev/null -w "mcp /health → %{http_code} (503 = sin pairing todavía)\n" "http://127.0.0.1:${WA_MCP_PORT}/health"

echo "== 7/7 recordatorios =="
echo "· Pairing (una vez, con Juan): launchctl bootout gui/$UID_/com.whatsapp-bridge &&"
echo "    cd $HOME_DIR && WHATSAPP_DEVICE_NAME=base-mcp WHATSAPP_BRIDGE_PORT=8814 WHATSAPP_MEDIA_ROOTS=$HOME_DIR/outbox:$HOME_DIR/store \\"
echo "      $HOME_DIR/bin/wa-bridge --full-history-pair --pair-code <telefono sin +>"
echo "  Esperar el primer HistorySync en la salida, Ctrl-C, y launchctl bootstrap del plist del bridge."
echo "· Watchdog: dos entradas en ~/.mcp-watchdog/watchdog.sh (ver README)."
echo "· Gateway: WA_MCP_TOKEN en ~/.base-mcp/env + upstream 'wa' (http_bearer) + perfiles."
