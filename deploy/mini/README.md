# whatsapp-mcp en el Mac mini

El WhatsApp personal de Juan como servicio de `base-mcp` (namespace `wa`).
Dos daemons launchd, ambos loopback, ambos con bearer:

| Proceso | Label | Puerto | Qué es |
|---|---|---|---|
| `wa-bridge` (Go, whatsmeow) | `com.whatsapp-bridge` | 127.0.0.1:8814 | Dispositivo vinculado "base-mcp". Guarda sesión e historial en `~/.whatsapp-mcp/store/`. Token en `store/.bridge-token`. |
| `base_server.py` (Python, MCP) | `com.whatsapp-mcp` | 127.0.0.1:8804 | Tools MCP sobre el bridge y su SQLite: tope de envío, dedup, transcripción, outbox, `/health`. Token `WA_MCP_TOKEN` en `~/.whatsapp-mcp/env`. |

El gateway (`~/base/code/base-mcp`, upstream `wa`, `http_bearer`) consume
`http://127.0.0.1:8804/mcp` con `WA_MCP_TOKEN` desde `~/.base-mcp/env`.

## Instalar / actualizar

```bash
bash ~/base/code/whatsapp-mcp/deploy/mini/install.sh
```

Idempotente: pull, build del bridge, `uv sync` + tests, plists, health. La
primera vez crea `~/.whatsapp-mcp/env` con un `WA_MCP_TOKEN` nuevo; hay que
copiarlo a `~/.base-mcp/env` y reiniciar el gateway.

## Pairing (una vez, con Juan en el teléfono)

El servicio **nunca** empareja solo (`--service`): sin sesión sirve
`/api/health` con `status=not_paired` y espera. Emparejar es manual:

```bash
launchctl bootout gui/$(id -u)/com.whatsapp-bridge
cd ~/.whatsapp-mcp && set -a && . env && set +a
WHATSAPP_DEVICE_NAME=base-mcp WHATSAPP_MEDIA_ROOTS=$HOME/.whatsapp-mcp/outbox:$HOME/.whatsapp-mcp/store \
  ~/.whatsapp-mcp/bin/wa-bridge --full-history-pair --pair-code 34XXXXXXXXX
```

Imprime un código de 8 caracteres. En el teléfono: WhatsApp → Dispositivos
vinculados → Vincular un dispositivo → "Vincular con el número de teléfono" →
escribir el código. `--full-history-pair` pide el historial completo (3650
días); **sólo cuenta en el pairing**: si se olvida, hay que desvincular y
repetir. Dejar el proceso en primer plano hasta ver el primer `HistorySync`
en la salida (minutos), `Ctrl-C`, y volver a cargar el daemon:

```bash
launchctl bootstrap gui/$(id -u) ~/Library/LaunchAgents/com.whatsapp-bridge.plist
```

QR en vez de código: mismo comando sin `--pair-code` (sólo útil con una pantalla delante).

Gotchas del pairing por código (2026-09-06): el código vive ~3 min y WhatsApp cierra el stream
a los 3,5; el proceso reemite uno nuevo solo cuando eso pasa (ventana total 15 min). Si el
teléfono dice "no se pudo vincular" y el log muestra `stage="refresh_code"`, el código tipeado
no coincidía (pantalla de vincular vieja): cerrarla y tipear el último código impreso.

## Revincular (sesión caída o `LoggedOut`)

`curl -s 127.0.0.1:8804/health` devuelve 503 y `logged_in:false`. No hay
backup de sesión a propósito (restaurar claves viejas rompe el cifrado):

1. `launchctl bootout gui/$(id -u)/com.whatsapp-bridge`
2. `rm ~/.whatsapp-mcp/store/whatsapp.db*` (el historial en `messages.db` queda)
3. Pairing como arriba.

Si el teléfono pasa 14 días sin conexión, WhatsApp desvincula solo.

## Watchdog

Entradas en `~/.mcp-watchdog/watchdog.sh` (array `SERVICIOS`):

```
"com.whatsapp-bridge|http://127.0.0.1:8814/api/health|401|2|"
"com.whatsapp-mcp|http://127.0.0.1:8804/health|200|2| Si el JSON dice logged_in=false: revincular (README de deploy/mini)."
```

El bridge responde 401 sin token, lo que alcanza para saber que está vivo (y
no pedir QR en loop: sin sesión igual da 200 al que sí lleva token). El MCP
devuelve 503 cuando el bridge no está emparejado o conectado: ahí llega el
aviso por Telegram. Logs a rotar (array `LOGS`): `~/.whatsapp-mcp/logs/*.log`.

## Operación diaria

- Tope de envíos: `WA_SEND_MAX_PER_HOUR` en `~/.whatsapp-mcp/env`
  (`launchctl kickstart -k gui/$(id -u)/com.whatsapp-mcp` para aplicar).
- Registro de envíos: `sqlite3 ~/.whatsapp-mcp/state.db 'select datetime(ts,"unixepoch"),recipient,kind,ok from sends order by ts desc limit 20'`.
- Transcripciones cacheadas: tabla `transcripts` del mismo `state.db`.
- Archivos para mandar: `~/.whatsapp-mcp/outbox/` (tool `put_outbox` o copiar a mano).
- Leer por MCP nunca marca como leído: la tool `mark_messages_read` del
  upstream no se expone.
- Traer la deriva del upstream: `git fetch upstream && git merge upstream/main`
  en `~/base/code/whatsapp-mcp`; el bridge se rebuildea con `install.sh`.
  Subir whatsmeow: `cd whatsapp-bridge && go get go.mau.fi/whatsmeow@latest && go test ./...`.
