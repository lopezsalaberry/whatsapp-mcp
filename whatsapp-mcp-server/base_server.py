"""Base-layer entrypoint for the WhatsApp MCP (fork-local; upstream ``main.py`` stays untouched).

Run with ``uv run python base_server.py``. It builds a fresh FastMCP server that
re-exposes the upstream read tools, wraps the three send tools behind a
two-step confirm gate (preview + 10-minute single-use token, same pattern as
mail-mcp) and a rate-limit + dedup guard, hides ``mark_messages_read``
(reading through MCP never marks chats read), and adds voice-note
transcription, an outbox for files to send, a send-quota probe and a
``/health`` endpoint for the watchdog.

Environment variables:

    WA_MCP_HOST           Bind address (default 127.0.0.1).
    WA_MCP_PORT           Bind port (default 8804).
    WA_MCP_TOKEN          Bearer token required on every request except /health.
                          Required; the server refuses to start without it.
    WA_MCP_ALLOW_NO_AUTH  Set to "1" to run without a token (tests/dev ONLY).
    WA_STATE_DB           SQLite file for the send guard + transcript cache
                          (default ~/.whatsapp-mcp/state.db).
    WA_OUTBOX             Directory where put_outbox stores files
                          (default ~/.whatsapp-mcp/outbox).
    WA_SEND_REQUIRE_CONFIRM
                          "1" (default): every send is two-step -- the first call
                          returns a preview + confirm_token, the second call with
                          that token sends. "0": sends go straight through the
                          rate-limit/dedup guard (no token needed).
    WA_SEND_MAX_PER_HOUR  Max send attempts per trailing hour (default 30).
    WA_DEDUP_SECONDS      Window for suppressing identical sends (default 300).
    WA_STT_MODEL          faster-whisper model name (default large-v3).
    WA_STT_COMPUTE        CTranslate2 compute type (default int8).
    WA_STT_THREADS        CPU threads for inference (default 4).
    WA_STT_SYNC_MAX_S     Audio up to this many seconds is transcribed inline;
                          longer notes go to a background thread (default 45).
    WA_STT_IDLE_UNLOAD_S  Unload the model after this idle time (default 600).
    WA_STT_LANGUAGE       Default language hint for the model (default es).

Upstream env vars still apply: WHATSAPP_API_URL, WHATSAPP_DB_PATH,
WHATSAPP_BRIDGE_TOKEN (see whatsapp.py).
"""

from __future__ import annotations

import base64
import binascii
import hmac
import logging
import os
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import requests
from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.concurrency import run_in_threadpool
from starlette.datastructures import Headers
from starlette.requests import Request
from starlette.responses import JSONResponse, Response
from starlette.types import ASGIApp, Receive, Scope, Send

import main
import whatsapp
from send_guard import TOKEN_TTL_SECONDS, Decision, SendGuard, content_hash
from transcribe import Transcriber

logger = logging.getLogger("whatsapp.base")

HIDDEN_UPSTREAM_TOOLS = frozenset({"mark_messages_read"})
GUARDED_UPSTREAM_TOOLS = frozenset({"send_message", "send_file", "send_audio_message"})
LOCAL_HOSTS = frozenset({"localhost", "127.0.0.1", "::1", "[::1]"})
OPEN_PATHS = frozenset({"/health"})
OUTBOX_MAX_BYTES = 50 * 1024 * 1024
BRIDGE_HEALTH_TIMEOUT_S = 5
OUTBOX_FETCH_TIMEOUT_S = 30
TWO_STEP_DOC = (
    "Two-step send AS JUAN from his personal WhatsApp. Call WITHOUT confirm_token -> returns a preview and a "
    "10-minute single-use token. Show the preview to Juan and call again WITH the token ONLY after he explicitly "
    "approved this exact draft in the conversation. Never confirm on his behalf.\n\n"
)
GATE_ARGS_DOC = (
    "\n        confirm_token: Token returned by the previous call for this exact draft. Omit to get the\n"
    "                       preview + token; pass it (once) to send. Rejected with TOKEN_INVALID,\n"
    "                       TOKEN_EXPIRED, TOKEN_USED or PAYLOAD_MISMATCH -- none of those send anything.\n"
    "        idempotency_key: Optional caller-chosen key. A key already used in the last 24 h makes\n"
    "                         the call a no-op that returns the previous result with deduplicated=true.\n"
)


# --------------------------------------------------------------------------
# Settings
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class Settings:
    host: str = "127.0.0.1"
    port: int = 8804
    token: str | None = None
    allow_no_auth: bool = False
    state_db: str = os.path.expanduser("~/.whatsapp-mcp/state.db")
    outbox: str = os.path.expanduser("~/.whatsapp-mcp/outbox")
    send_require_confirm: bool = True
    send_max_per_hour: int = 30
    dedup_seconds: float = 300.0
    stt_model: str = "large-v3"
    stt_compute: str = "int8"
    stt_threads: int = 4
    stt_sync_max_s: float = 45.0
    stt_idle_unload_s: float = 600.0
    stt_language: str = "es"


def _env_int(name: str, default: int) -> int:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"Invalid {name}: {raw!r} is not an integer") from None


def _env_float(name: str, default: float) -> float:
    raw = os.getenv(name, "").strip()
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"Invalid {name}: {raw!r} is not a number") from None


def load_settings() -> Settings:
    """Read every WA_* env var into a :class:`Settings`. Raises ValueError on bad values."""
    token = os.getenv("WA_MCP_TOKEN", "").strip() or None
    allow_no_auth = os.getenv("WA_MCP_ALLOW_NO_AUTH", "").strip() == "1"
    if token is None and not allow_no_auth:
        raise ValueError("WA_MCP_TOKEN is required (set WA_MCP_ALLOW_NO_AUTH=1 only for tests/dev)")
    port = _env_int("WA_MCP_PORT", 8804)
    if not 1 <= port <= 65535:
        raise ValueError(f"Invalid WA_MCP_PORT: {port}")
    return Settings(
        host=os.getenv("WA_MCP_HOST", "").strip() or "127.0.0.1",
        port=port,
        token=token,
        allow_no_auth=allow_no_auth,
        state_db=os.path.expanduser(os.getenv("WA_STATE_DB", "").strip() or "~/.whatsapp-mcp/state.db"),
        outbox=os.path.expanduser(os.getenv("WA_OUTBOX", "").strip() or "~/.whatsapp-mcp/outbox"),
        send_require_confirm=(os.getenv("WA_SEND_REQUIRE_CONFIRM", "").strip() or "1") != "0",
        send_max_per_hour=_env_int("WA_SEND_MAX_PER_HOUR", 30),
        dedup_seconds=_env_float("WA_DEDUP_SECONDS", 300.0),
        stt_model=os.getenv("WA_STT_MODEL", "").strip() or "large-v3",
        stt_compute=os.getenv("WA_STT_COMPUTE", "").strip() or "int8",
        stt_threads=_env_int("WA_STT_THREADS", 4),
        stt_sync_max_s=_env_float("WA_STT_SYNC_MAX_S", 45.0),
        stt_idle_unload_s=_env_float("WA_STT_IDLE_UNLOAD_S", 600.0),
        stt_language=os.getenv("WA_STT_LANGUAGE", "").strip() or "es",
    )


def _ensure_private_dir(path: str) -> None:
    if not os.path.isdir(path):
        os.makedirs(path, mode=0o700, exist_ok=True)
        os.chmod(path, 0o700)


# --------------------------------------------------------------------------
# Server assembly
# --------------------------------------------------------------------------


@dataclass
class BaseServer:
    settings: Settings
    mcp: FastMCP
    guard: SendGuard
    transcriber: Transcriber

    def tool(self, name: str) -> Callable[..., Any]:
        """Return the registered function behind a tool name (handy for tests)."""
        tool = self.mcp._tool_manager.get_tool(name)
        if tool is None:
            raise KeyError(name)
        return tool.fn


def _quota(guard: SendGuard) -> dict[str, Any]:
    return {"sends_last_hour": guard.sends_last_hour(), "limit": guard.max_per_hour}


def _normalize_jid(recipient: str) -> str:
    """Digits-only phone -> ``<digits>@s.whatsapp.net``; anything with '@' passes through."""
    value = (recipient or "").strip()
    if "@" in value:
        return value
    digits = "".join(ch for ch in value if ch.isdigit())
    return f"{digits}@s.whatsapp.net" if digits else value


def _chat_name(jid: str) -> str | None:
    """Best-effort chat name from messages.db (read-only); None when unknown or unreadable."""
    path = whatsapp.MESSAGES_DB_PATH
    if not jid or not os.path.exists(path):
        return None
    try:
        uri = Path(os.path.abspath(path)).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            row = conn.execute("SELECT name FROM chats WHERE jid = ?", (jid,)).fetchone()
        finally:
            conn.close()
    except (sqlite3.Error, OSError):
        return None
    name = row[0] if row else None
    return name if name and name != jid else None


def _preview(kind: str, recipient: str, **fields: Any) -> dict[str, Any]:
    jid = _normalize_jid(recipient)
    preview: dict[str, Any] = {
        "kind": kind,
        "recipient": recipient,
        "recipient_jid": jid,
        "chat_name": _chat_name(jid),
        "is_group": jid.endswith("@g.us"),
    }
    preview.update(fields)
    return preview


def _file_size(path: str) -> int | None:
    try:
        return os.path.getsize(path)
    except OSError:
        return None


def _guarded_send(
    guard: SendGuard,
    kind: str,
    recipient: str,
    digest: str,
    idempotency_key: str,
    size: int,
    do_send: Callable[[], dict[str, Any]],
    confirm_token: str = "",
    preview: Callable[[], dict[str, Any]] | None = None,
    require_confirm: bool = False,
) -> dict[str, Any]:
    if not recipient:
        return {"success": False, "message": "Recipient must be provided", "guard": _quota(guard)}

    if require_confirm:
        if not confirm_token:
            token = guard.issue_token(digest)
            logger.info("send: preview issued kind=%s recipient=%s size=%d", kind, recipient, size)
            return {
                "status": "confirm_required",
                "preview": preview() if preview else {"kind": kind, "recipient": recipient},
                "confirm_token": token,
                "expires_in_seconds": TOKEN_TTL_SECONDS,
                "guard": _quota(guard),
            }
        failure = guard.consume_token(confirm_token, digest)
        if failure:
            logger.warning("send: confirm rejected kind=%s recipient=%s error=%s", kind, recipient, failure)
            return {
                "success": False,
                "status": "confirm_rejected",
                "error": failure,
                "message": f"confirmation token rejected: {failure}; nothing was sent",
                "guard": _quota(guard),
            }

    decision: Decision = guard.check(recipient, kind, digest, idempotency_key)
    if decision.kind == "duplicate":
        previous = decision.previous or {}
        return {
            "success": True,
            "message": f"duplicate suppressed: {previous.get('message', 'same content sent recently')}",
            "deduplicated": True,
            "previous": previous,
            "guard": {"sends_last_hour": decision.sends_last_hour, "limit": guard.max_per_hour},
        }
    if decision.kind == "rate_limited":
        return {
            "success": False,
            "message": (
                f"rate limited: {decision.sends_last_hour}/{guard.max_per_hour} sends in the last hour, "
                f"resets in {int(decision.resets_in_s)}s"
            ),
            "rate_limited": True,
            "guard": {
                "sends_last_hour": decision.sends_last_hour,
                "limit": guard.max_per_hour,
                "resets_in_s": round(decision.resets_in_s, 1),
            },
        }

    logger.info("send: kind=%s recipient=%s size=%d", kind, recipient, size)
    try:
        result = do_send()
    except Exception as exc:
        logger.exception("send: upstream raised kind=%s recipient=%s", kind, recipient)
        result = {"success": False, "message": f"send failed: {type(exc).__name__}: {exc}"}
    if not isinstance(result, dict):
        result = {"success": bool(result), "message": str(result)}
    ok = bool(result.get("success"))
    guard.record(recipient, kind, digest, idempotency_key, ok, result, message_id=result.get("message_id"))
    logger.info("send: kind=%s recipient=%s ok=%s", kind, recipient, ok)
    return {**result, "guard": _quota(guard)}


def _guarded_description(upstream_fn: Callable[..., Any]) -> str:
    doc = (upstream_fn.__doc__ or "").strip()
    if "\n    Returns:" in doc:
        head, _, tail = doc.partition("\n    Returns:")
        doc = head.rstrip("\n") + GATE_ARGS_DOC + "\n    Returns:" + tail
    else:
        doc = doc + "\n" + GATE_ARGS_DOC
    return TWO_STEP_DOC + doc


def _validate_outbox_filename(filename: str) -> str:
    name = (filename or "").strip()
    if not name or name in {".", ".."}:
        raise ValueError("filename must be a bare file name")
    if "\x00" in name or "/" in name or "\\" in name or os.path.basename(name) != name:
        raise ValueError("filename must not contain path separators")
    if name.startswith("."):
        raise ValueError("filename must not start with '.'")
    return name


def _open_unique(outbox: str, name: str) -> tuple[str, int]:
    """Create ``name`` (or ``name-1``, ``name-2``...) exclusively inside the outbox."""
    stem, ext = os.path.splitext(name)
    for attempt in range(0, 10_000):
        candidate = name if attempt == 0 else f"{stem}-{attempt}{ext}"
        path = os.path.join(outbox, candidate)
        try:
            fd = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
        except FileExistsError:
            continue
        return path, fd
    raise ValueError("could not find a free file name in the outbox")


def _fetch_url_to(fd: int, url: str) -> int:
    parts = urlsplit(url)
    if parts.scheme not in {"http", "https"} or not parts.netloc:
        raise ValueError("url must be http:// or https://")
    total = 0
    with os.fdopen(fd, "wb") as out, requests.get(url, stream=True, timeout=OUTBOX_FETCH_TIMEOUT_S) as resp:
        resp.raise_for_status()
        declared = resp.headers.get("Content-Length")
        if declared and declared.isdigit() and int(declared) > OUTBOX_MAX_BYTES:
            raise ValueError(f"remote file is larger than {OUTBOX_MAX_BYTES} bytes")
        for chunk in resp.iter_content(chunk_size=1 << 16):
            if not chunk:
                continue
            total += len(chunk)
            if total > OUTBOX_MAX_BYTES:
                raise ValueError(f"remote file is larger than {OUTBOX_MAX_BYTES} bytes")
            out.write(chunk)
    return total


def _messages_db_info(path: str) -> dict[str, Any]:
    info: dict[str, Any] = {"path": path, "exists": os.path.exists(path)}
    if not info["exists"]:
        return info
    try:
        info["size_bytes"] = os.path.getsize(path)
        uri = Path(os.path.abspath(path)).as_uri() + "?mode=ro"
        conn = sqlite3.connect(uri, uri=True, timeout=2)
        try:
            row = conn.execute("SELECT MAX(timestamp) FROM messages").fetchone()
        finally:
            conn.close()
        info["last_message_at"] = row[0] if row else None
    except (sqlite3.Error, OSError) as exc:
        info["error"] = f"{type(exc).__name__}: {exc}"
    return info


def _bridge_health() -> dict[str, Any]:
    url = f"{whatsapp.WHATSAPP_API_BASE_URL}/health"
    try:
        resp = requests.get(url, headers=whatsapp._bridge_headers(), timeout=BRIDGE_HEALTH_TIMEOUT_S)
    except requests.RequestException as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    if resp.status_code != 200:
        return {"error": f"HTTP {resp.status_code}", "http_status": resp.status_code}
    try:
        body = resp.json()
    except ValueError:
        return {"error": "bridge returned non-JSON body"}
    if not isinstance(body, dict):
        return {"error": "bridge returned unexpected JSON"}
    return body


def _bridge_ok(bridge: dict[str, Any]) -> bool:
    if "error" in bridge:
        return False
    if bridge.get("status") != "ok":
        return False
    logged_in = bridge.get("logged_in")
    if logged_in is None:
        logged_in = bridge.get("connected")
    return bool(logged_in)


def build_server(settings: Settings | None = None) -> BaseServer:
    """Assemble the FastMCP server with guard, transcriber and all tools registered."""
    settings = settings or load_settings()
    if settings.token is None and not settings.allow_no_auth:
        raise ValueError("WA_MCP_TOKEN is required (set WA_MCP_ALLOW_NO_AUTH=1 only for tests/dev)")

    _ensure_private_dir(os.path.dirname(os.path.abspath(settings.state_db)))
    _ensure_private_dir(settings.outbox)

    guard = SendGuard(settings.state_db, settings.send_max_per_hour, settings.dedup_seconds)
    transcriber = Transcriber(
        settings.state_db,
        model_name=settings.stt_model,
        compute_type=settings.stt_compute,
        cpu_threads=settings.stt_threads,
        sync_max_s=settings.stt_sync_max_s,
        idle_unload_s=settings.stt_idle_unload_s,
        default_language=settings.stt_language,
    )

    mcp = FastMCP("whatsapp", host=settings.host, port=settings.port)

    # -- upstream tools, minus the hidden and the guarded ones ---------------
    for tool in main.mcp._tool_manager.list_tools():
        if tool.name in HIDDEN_UPSTREAM_TOOLS or tool.name in GUARDED_UPSTREAM_TOOLS:
            continue
        mcp.add_tool(
            tool.fn, name=tool.name, title=tool.title, description=tool.description, annotations=tool.annotations
        )

    # -- guarded send wrappers (same names + trailing confirm_token, idempotency_key)
    require_confirm = settings.send_require_confirm

    def send_message(
        recipient: str,
        message: str,
        quoted_message_id: str = "",
        quoted_sender_jid: str = "",
        quoted_content: str = "",
        mentions: list[str] | None = None,
        confirm_token: str = "",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Two-step send AS JUAN from his personal WhatsApp (see tool description)."""
        return _guarded_send(
            guard,
            "message",
            recipient,
            content_hash("message", recipient, message),
            idempotency_key,
            len(message or ""),
            lambda: main.send_message(
                recipient, message, quoted_message_id, quoted_sender_jid, quoted_content, mentions
            ),
            confirm_token=confirm_token,
            preview=lambda: _preview(
                "text",
                recipient,
                message=message,
                quoted_message_id=quoted_message_id or None,
                mentions=mentions or None,
            ),
            require_confirm=require_confirm,
        )

    def send_file(
        recipient: str, media_path: str, caption: str = "", confirm_token: str = "", idempotency_key: str = ""
    ) -> dict[str, Any]:
        """Two-step send AS JUAN from his personal WhatsApp (see tool description)."""
        return _guarded_send(
            guard,
            "file",
            recipient,
            content_hash("file", recipient, media_path, caption),
            idempotency_key,
            len(caption or ""),
            lambda: main.send_file(recipient, media_path, caption),
            confirm_token=confirm_token,
            preview=lambda: _preview(
                "file", recipient, caption=caption, media_path=media_path, media_bytes=_file_size(media_path)
            ),
            require_confirm=require_confirm,
        )

    def send_audio_message(
        recipient: str, media_path: str, confirm_token: str = "", idempotency_key: str = ""
    ) -> dict[str, Any]:
        """Two-step send AS JUAN from his personal WhatsApp (see tool description)."""
        return _guarded_send(
            guard,
            "audio",
            recipient,
            content_hash("audio", recipient, media_path),
            idempotency_key,
            0,
            lambda: main.send_audio_message(recipient, media_path),
            confirm_token=confirm_token,
            preview=lambda: _preview("audio", recipient, media_path=media_path, media_bytes=_file_size(media_path)),
            require_confirm=require_confirm,
        )

    mcp.add_tool(send_message, name="send_message", description=_guarded_description(main.send_message))
    mcp.add_tool(send_file, name="send_file", description=_guarded_description(main.send_file))
    mcp.add_tool(
        send_audio_message, name="send_audio_message", description=_guarded_description(main.send_audio_message)
    )

    # -- new tools ---------------------------------------------------------
    def transcribe_audio(message_id: str, chat_jid: str, language: str = "") -> dict[str, Any]:
        """Transcribe a WhatsApp voice note or audio message with a local Whisper model.

        Downloads the media through the bridge, converts it with ffmpeg and runs faster-whisper
        on CPU. Short notes are transcribed inline; long ones run in the background and the call
        returns status "pending" — call again with the same ids to collect the text. Results are
        cached per (chat_jid, message_id).

        Args:
            message_id: The ID of the message containing the audio
            chat_jid: The JID of the chat containing the message
            language: Optional ISO language hint (e.g. "es", "en"); defaults to WA_STT_LANGUAGE

        Returns:
            {"status": "done"|"pending"|"error", "text", "language", "duration_s", "cached", "message"}
        """
        return transcriber.transcribe(message_id, chat_jid, language)

    def put_outbox(filename: str, content_base64: str = "", url: str = "") -> dict[str, Any]:
        """Store a file in the outbox so it can be sent later with send_file or send_audio_message.

        Provide exactly one of content_base64 or url. The filename must be a bare name (no
        directories, no leading dot); if it already exists a numeric suffix (-1, -2, ...) is added,
        never overwriting. Files larger than 50 MB are rejected.

        Args:
            filename: Bare file name, e.g. "invoice.pdf"
            content_base64: File bytes encoded in base64
            url: http(s) URL to download the file from (30 s timeout)

        Returns:
            {"success": bool, "path": str|None, "bytes": int, "message": str}
        """
        try:
            name = _validate_outbox_filename(filename)
        except ValueError as exc:
            return {"success": False, "path": None, "bytes": 0, "message": str(exc)}
        if bool(content_base64) == bool(url):
            return {
                "success": False,
                "path": None,
                "bytes": 0,
                "message": "provide exactly one of content_base64 or url",
            }

        path, fd = _open_unique(settings.outbox, name)
        try:
            if content_base64:
                try:
                    data = base64.b64decode(content_base64, validate=True)
                except (binascii.Error, ValueError):
                    raise ValueError("content_base64 is not valid base64") from None
                if len(data) > OUTBOX_MAX_BYTES:
                    raise ValueError(f"content is larger than {OUTBOX_MAX_BYTES} bytes")
                with os.fdopen(fd, "wb") as out:
                    out.write(data)
                size = len(data)
            else:
                size = _fetch_url_to(fd, url)
        except (ValueError, OSError, requests.RequestException) as exc:
            try:
                os.unlink(path)
            except OSError:
                pass
            logger.warning("outbox: rejected name=%s reason=%s", name, type(exc).__name__)
            return {"success": False, "path": None, "bytes": 0, "message": str(exc)}

        logger.info("outbox: stored name=%s bytes=%d", os.path.basename(path), size)
        return {"success": True, "path": path, "bytes": size, "message": f"stored {os.path.basename(path)}"}

    def send_quota() -> dict[str, Any]:
        """Report how many sends happened in the last hour against the configured limit.

        Returns:
            {"sends_last_hour": int, "limit": int, "resets_in_s": float}
        """
        return {
            "sends_last_hour": guard.sends_last_hour(),
            "limit": guard.max_per_hour,
            "resets_in_s": round(guard.resets_in_s(), 1),
        }

    mcp.add_tool(transcribe_audio, name="transcribe_audio")
    mcp.add_tool(put_outbox, name="put_outbox")
    mcp.add_tool(send_quota, name="send_quota")

    # -- /health ----------------------------------------------------------
    @mcp.custom_route("/health", methods=["GET"])
    async def health(_request: Request) -> Response:
        try:
            bridge = await run_in_threadpool(_bridge_health)
            messages_db = await run_in_threadpool(_messages_db_info, whatsapp.MESSAGES_DB_PATH)
            sends = await run_in_threadpool(guard.sends_last_hour)
            ok = _bridge_ok(bridge)
            payload = {
                "ok": ok,
                "bridge": bridge,
                "messages_db": messages_db,
                "sends_last_hour": sends,
                "stt": {"model": settings.stt_model, "loaded": transcriber.loaded},
            }
        except Exception as exc:  # /health must never raise
            logger.exception("health: unexpected failure")
            ok = False
            payload = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        return JSONResponse(payload, status_code=200 if ok else 503)

    return BaseServer(settings=settings, mcp=mcp, guard=guard, transcriber=transcriber)


# --------------------------------------------------------------------------
# HTTP layer
# --------------------------------------------------------------------------


def _host_without_port(raw: str) -> str:
    host = (raw or "").strip().lower()
    if host.startswith("["):
        end = host.find("]")
        return host[: end + 1] if end != -1 else host
    if host.count(":") == 1:
        return host.split(":", 1)[0]
    return host


class LocalBearerAuth:
    """Pure-ASGI middleware: local Host only, bearer token everywhere except OPEN_PATHS."""

    def __init__(self, app: ASGIApp, token: str | None, allow_no_auth: bool = False) -> None:
        self.app = app
        self.token = token
        self.allow_no_auth = allow_no_auth

    async def __call__(self, scope: Scope, receive: Receive, send: Send) -> None:
        if scope["type"] != "http":
            await self.app(scope, receive, send)
            return
        headers = Headers(scope=scope)
        if _host_without_port(headers.get("host", "")) not in LOCAL_HOSTS:
            response = JSONResponse({"error": "misdirected request: local Host header required"}, status_code=421)
            await response(scope, receive, send)
            return
        if scope["path"] not in OPEN_PATHS and not self._authorized(headers.get("authorization", "")):
            response = JSONResponse({"error": "unauthorized"}, status_code=401, headers={"WWW-Authenticate": "Bearer"})
            await response(scope, receive, send)
            return
        await self.app(scope, receive, send)

    def _authorized(self, header: str) -> bool:
        if self.token is None:
            return self.allow_no_auth
        scheme, _, credential = header.strip().partition(" ")
        if scheme.lower() != "bearer" or not credential:
            return False
        return hmac.compare_digest(credential.strip().encode(), self.token.encode())


def build_app(settings: Settings | None = None) -> Starlette:
    """Build the ASGI app: streamable-HTTP MCP at /mcp, /health open, everything behind bearer auth."""
    server = build_server(settings)
    app = server.mcp.streamable_http_app()
    app.add_middleware(LocalBearerAuth, token=server.settings.token, allow_no_auth=server.settings.allow_no_auth)
    app.state.server = server
    return app


def _main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    try:
        settings = load_settings()
    except ValueError as exc:
        logger.error("%s", exc)
        return 2
    app = build_app(settings)
    if settings.token is None:
        logger.warning("WA_MCP_ALLOW_NO_AUTH=1: running WITHOUT bearer auth (dev only)")
    logger.info(
        "listening on http://%s:%d/mcp (health at /health, outbox=%s, state=%s)",
        settings.host,
        settings.port,
        settings.outbox,
        settings.state_db,
    )
    import uvicorn

    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())
