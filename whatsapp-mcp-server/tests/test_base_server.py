"""Tests for base_server: tool surface, guarded sends, outbox, /health and auth."""

import base64
import os
import sqlite3
import time

import pytest
import requests
from starlette.responses import PlainTextResponse
from starlette.testclient import TestClient

import base_server
import main
import send_guard
import whatsapp
from base_server import LocalBearerAuth, build_app

TOKEN = "testtoken"
AUTH = {"Authorization": f"Bearer {TOKEN}"}
TWO_STEP_PREFIX = (
    "Two-step send AS JUAN from his personal WhatsApp. Call WITHOUT confirm_token -> returns a preview and a "
    "10-minute single-use token. Show the preview to Juan and call again WITH the token ONLY after he explicitly "
    "approved this exact draft in the conversation. Never confirm on his behalf."
)
UPSTREAM_TOOLS = {t.name for t in main.mcp._tool_manager.list_tools()}


@pytest.fixture
def env(tmp_path, monkeypatch):
    monkeypatch.setenv("WA_MCP_TOKEN", TOKEN)
    monkeypatch.delenv("WA_MCP_ALLOW_NO_AUTH", raising=False)
    monkeypatch.setenv("WA_STATE_DB", str(tmp_path / "state" / "state.db"))
    monkeypatch.setenv("WA_OUTBOX", str(tmp_path / "outbox"))
    monkeypatch.setenv("WA_SEND_MAX_PER_HOUR", "2")
    monkeypatch.setenv("WA_DEDUP_SECONDS", "300")
    monkeypatch.delenv("WA_SEND_REQUIRE_CONFIRM", raising=False)  # default: confirm gate ON
    return tmp_path


@pytest.fixture
def no_confirm(env, monkeypatch):
    """Relax the two-step gate (WA_SEND_REQUIRE_CONFIRM=0) before build_app runs."""
    monkeypatch.setenv("WA_SEND_REQUIRE_CONFIRM", "0")
    return env


@pytest.fixture
def app(env):
    return build_app()


@pytest.fixture
def server(app):
    return app.state.server


@pytest.fixture
def client(app):
    # Default TestClient Host is "testserver", which the middleware rejects; use a local Host.
    return TestClient(app, base_url="http://localhost")


@pytest.fixture
def sent(monkeypatch):
    """Replace the upstream send functions with recorders; nothing touches the network."""
    calls: dict[str, list] = {"message": [], "file": [], "audio": []}

    def fake_send_message(
        recipient, message, quoted_message_id="", quoted_sender_jid="", quoted_content="", mentions=None
    ):
        calls["message"].append((recipient, message, quoted_message_id, quoted_sender_jid, quoted_content, mentions))
        return {"success": True, "message": "Message sent to " + recipient}

    def fake_send_file(recipient, media_path, caption=""):
        calls["file"].append((recipient, media_path, caption))
        return {"success": True, "message": "File sent"}

    def fake_send_audio(recipient, media_path):
        calls["audio"].append((recipient, media_path))
        return {"success": True, "message": "Audio sent"}

    monkeypatch.setattr(main, "send_message", fake_send_message)
    monkeypatch.setattr(main, "send_file", fake_send_file)
    monkeypatch.setattr(main, "send_audio_message", fake_send_audio)
    return calls


# --------------------------------------------------------------------------
# Tool surface
# --------------------------------------------------------------------------


class TestToolSurface:
    def test_registered_tool_names(self, server):
        names = {t.name for t in server.mcp._tool_manager.list_tools()}
        expected = (UPSTREAM_TOOLS - {"mark_messages_read"}) | {"transcribe_audio", "put_outbox", "send_quota"}
        assert names == expected
        assert "mark_messages_read" not in names

    def test_upstream_read_tools_keep_their_function_and_description(self, server):
        ours = server.mcp._tool_manager.get_tool("list_chats")
        theirs = main.mcp._tool_manager.get_tool("list_chats")
        assert ours.fn is theirs.fn
        assert ours.description == theirs.description

    @pytest.mark.parametrize("name", ["send_message", "send_file", "send_audio_message"])
    def test_send_wrappers_signature_and_docs(self, server, name):
        ours = server.mcp._tool_manager.get_tool(name)
        theirs = main.mcp._tool_manager.get_tool(name)
        assert ours.fn is not theirs.fn
        assert ours.description.startswith(TWO_STEP_PREFIX)
        assert theirs.description.strip().splitlines()[0] in ours.description
        assert "confirm_token" in ours.description and "idempotency_key" in ours.description
        ours_params = list(ours.parameters["properties"])
        theirs_params = list(theirs.parameters["properties"])
        assert ours_params == theirs_params + ["confirm_token", "idempotency_key"]

    def test_creates_state_and_outbox_dirs_private(self, env, server):
        outbox = env / "outbox"
        state_dir = env / "state"
        assert outbox.is_dir() and state_dir.is_dir()
        assert oct(outbox.stat().st_mode & 0o777) == "0o700"
        assert oct(state_dir.stat().st_mode & 0o777) == "0o700"

    def test_token_required(self, env, monkeypatch):
        monkeypatch.delenv("WA_MCP_TOKEN")
        with pytest.raises(ValueError, match="WA_MCP_TOKEN"):
            build_app()
        monkeypatch.setenv("WA_MCP_ALLOW_NO_AUTH", "1")
        assert build_app().state.server.settings.token is None


# --------------------------------------------------------------------------
# Guarded sends
# --------------------------------------------------------------------------


class TestGuardedSends:
    """Rate limit + dedup with the confirm gate relaxed (WA_SEND_REQUIRE_CONFIRM=0)."""

    @pytest.fixture(autouse=True)
    def _direct(self, no_confirm):
        return no_confirm

    def test_direct_send_works_without_token(self, server, sent):
        assert server.settings.send_require_confirm is False
        result = server.tool("send_message")("111@s.whatsapp.net", "hola")
        assert result["success"] is True and "confirm_token" not in result
        assert len(sent["message"]) == 1

    def test_send_message_delegates_and_reports_quota(self, server, sent):
        send = server.tool("send_message")
        result = send("111@s.whatsapp.net", "hola", mentions=["222"])
        assert result["success"] is True
        assert result["message"] == "Message sent to 111@s.whatsapp.net"
        assert result["guard"] == {"sends_last_hour": 1, "limit": 2}
        assert sent["message"] == [("111@s.whatsapp.net", "hola", "", "", "", ["222"])]

    def test_duplicate_content_is_suppressed(self, server, sent):
        send = server.tool("send_message")
        send("111@s.whatsapp.net", "hola")
        dup = send("111@s.whatsapp.net", "hola")
        assert dup["success"] is True
        assert dup["deduplicated"] is True
        assert dup["message"].startswith("duplicate suppressed: ")
        assert dup["previous"]["message"] == "Message sent to 111@s.whatsapp.net"
        assert len(sent["message"]) == 1  # the second call never reached upstream

    def test_idempotency_key_suppresses_different_content(self, server, sent):
        send = server.tool("send_message")
        send("111@s.whatsapp.net", "hola", idempotency_key="abc")
        dup = send("111@s.whatsapp.net", "otro texto", idempotency_key="abc")
        assert dup["deduplicated"] is True
        assert len(sent["message"]) == 1

    def test_rate_limited_returns_success_false(self, server, sent):
        send = server.tool("send_message")
        send("111@s.whatsapp.net", "uno")
        send("111@s.whatsapp.net", "dos")
        blocked = send("111@s.whatsapp.net", "tres")
        assert blocked["success"] is False
        assert blocked["rate_limited"] is True
        assert "rate limited" in blocked["message"]
        assert blocked["guard"]["sends_last_hour"] == 2 and blocked["guard"]["limit"] == 2
        assert len(sent["message"]) == 2

    def test_failed_upstream_send_counts_but_is_retryable(self, server, sent, monkeypatch):
        monkeypatch.setattr(main, "send_message", lambda *a, **k: {"success": False, "message": "bridge down"})
        send = server.tool("send_message")
        first = send("111@s.whatsapp.net", "hola")
        assert first["success"] is False and "deduplicated" not in first
        second = send("111@s.whatsapp.net", "hola")
        assert second["success"] is False and "deduplicated" not in second
        assert server.guard.sends_last_hour() == 2

    def test_upstream_exception_is_contained(self, server, monkeypatch):
        def boom(*a, **k):
            raise RuntimeError("kaput")

        monkeypatch.setattr(main, "send_message", boom)
        result = server.tool("send_message")("111@s.whatsapp.net", "hola")
        assert result["success"] is False
        assert "kaput" in result["message"]

    def test_empty_recipient_rejected_without_recording(self, server, sent):
        result = server.tool("send_message")("", "hola")
        assert result["success"] is False
        assert server.guard.sends_last_hour() == 0

    def test_send_file_and_audio_wrappers(self, server, sent):
        file_result = server.tool("send_file")("111@s.whatsapp.net", "/tmp/a.pdf", "caption")
        assert file_result["success"] is True and file_result["guard"]["sends_last_hour"] == 1
        assert sent["file"] == [("111@s.whatsapp.net", "/tmp/a.pdf", "caption")]

        dup = server.tool("send_file")("111@s.whatsapp.net", "/tmp/a.pdf", "caption")
        assert dup["deduplicated"] is True

        audio_result = server.tool("send_audio_message")("111@s.whatsapp.net", "/tmp/a.ogg")
        assert audio_result["success"] is True
        assert sent["audio"] == [("111@s.whatsapp.net", "/tmp/a.ogg")]

    def test_send_quota(self, server, sent):
        quota = server.tool("send_quota")()
        assert quota == {"sends_last_hour": 0, "limit": 2, "resets_in_s": 0.0}
        server.tool("send_message")("111@s.whatsapp.net", "hola")
        quota = server.tool("send_quota")()
        assert quota["sends_last_hour"] == 1
        assert 0 < quota["resets_in_s"] <= 3600


# --------------------------------------------------------------------------
# Two-step confirm gate (default)
# --------------------------------------------------------------------------


def _fake_chats_db(path, rows):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE chats (jid TEXT PRIMARY KEY, name TEXT)")
    conn.executemany("INSERT INTO chats VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


class TestConfirmGate:
    def test_gate_is_on_by_default(self, server):
        assert server.settings.send_require_confirm is True

    def test_without_token_returns_preview_and_does_not_send(self, server, sent, env, monkeypatch):
        db = env / "messages.db"
        _fake_chats_db(str(db), [("34600111222@s.whatsapp.net", "Mattea"), ("123-456@g.us", "Familia")])
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(db))

        result = server.tool("send_message")("34600111222", "hola", quoted_message_id="Q1")
        assert result["status"] == "confirm_required"
        assert result["expires_in_seconds"] == 600
        assert isinstance(result["confirm_token"], str) and len(result["confirm_token"]) >= 24
        assert result["preview"] == {
            "kind": "text",
            "recipient": "34600111222",
            "recipient_jid": "34600111222@s.whatsapp.net",
            "chat_name": "Mattea",
            "is_group": False,
            "message": "hola",
            "quoted_message_id": "Q1",
            "mentions": None,
        }
        assert sent["message"] == []
        assert server.guard.sends_last_hour() == 0

        group = server.tool("send_message")("123-456@g.us", "hola grupo")
        assert group["preview"]["recipient_jid"] == "123-456@g.us"
        assert group["preview"]["is_group"] is True
        assert group["preview"]["chat_name"] == "Familia"

    def test_preview_without_messages_db(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        result = server.tool("send_message")("+34 600 111 222", "hola")
        assert result["preview"]["recipient_jid"] == "34600111222@s.whatsapp.net"
        assert result["preview"]["chat_name"] is None

    def test_file_and_audio_previews(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        media = env / "doc.pdf"
        media.write_bytes(b"%PDF-1.4 hello")
        file_result = server.tool("send_file")("111@s.whatsapp.net", str(media), "mirá esto")
        assert file_result["status"] == "confirm_required"
        assert file_result["preview"]["kind"] == "file"
        assert file_result["preview"]["caption"] == "mirá esto"
        assert file_result["preview"]["media_path"] == str(media)
        assert file_result["preview"]["media_bytes"] == 14

        audio_result = server.tool("send_audio_message")("111@s.whatsapp.net", str(env / "nope.ogg"))
        assert audio_result["status"] == "confirm_required"
        assert audio_result["preview"]["kind"] == "audio"
        assert audio_result["preview"]["media_bytes"] is None
        assert sent["file"] == [] and sent["audio"] == []

    def test_token_sends_once_and_second_use_is_rejected(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        send = server.tool("send_message")
        token = send("111@s.whatsapp.net", "hola")["confirm_token"]

        sent_result = send("111@s.whatsapp.net", "hola", confirm_token=token)
        assert sent_result["success"] is True
        assert sent_result["message"] == "Message sent to 111@s.whatsapp.net"
        assert sent_result["guard"]["sends_last_hour"] == 1
        assert len(sent["message"]) == 1

        retry = send("111@s.whatsapp.net", "hola", confirm_token=token)
        assert retry["success"] is False
        assert retry["status"] == "confirm_rejected"
        assert retry["error"] == "TOKEN_USED"
        assert len(sent["message"]) == 1  # never a second message

    def test_token_bound_to_payload(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        send = server.tool("send_message")
        token = send("111@s.whatsapp.net", "hola")["confirm_token"]

        other_text = send("111@s.whatsapp.net", "chau", confirm_token=token)
        assert other_text["status"] == "confirm_rejected" and other_text["error"] == "PAYLOAD_MISMATCH"
        other_recipient = send("222@s.whatsapp.net", "hola", confirm_token=token)
        assert other_recipient["error"] == "PAYLOAD_MISMATCH"
        other_kind = server.tool("send_file")("111@s.whatsapp.net", "hola", confirm_token=token)
        assert other_kind["error"] == "PAYLOAD_MISMATCH"
        assert sent["message"] == [] and sent["file"] == []

        # The token is still unspent after the mismatches: the original draft can still go.
        assert send("111@s.whatsapp.net", "hola", confirm_token=token)["success"] is True
        assert len(sent["message"]) == 1

    def test_unknown_token_is_invalid(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        result = server.tool("send_message")("111@s.whatsapp.net", "hola", confirm_token="garbage")
        assert result["success"] is False
        assert result["status"] == "confirm_rejected" and result["error"] == "TOKEN_INVALID"
        assert sent["message"] == []

    def test_expired_token(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        send = server.tool("send_message")
        real_now = time.time()
        token = send("111@s.whatsapp.net", "hola")["confirm_token"]

        monkeypatch.setattr(send_guard.time, "time", lambda: real_now + send_guard.TOKEN_TTL_SECONDS + 1)
        result = send("111@s.whatsapp.net", "hola", confirm_token=token)
        assert result["success"] is False
        assert result["status"] == "confirm_rejected" and result["error"] == "TOKEN_EXPIRED"
        assert sent["message"] == []

    def test_confirmed_send_still_goes_through_guard(self, server, sent, env, monkeypatch):
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        send = server.tool("send_message")
        for text in ("uno", "dos"):
            token = send("111@s.whatsapp.net", text)["confirm_token"]
            assert send("111@s.whatsapp.net", text, confirm_token=token)["success"] is True
        token = send("111@s.whatsapp.net", "tres")["confirm_token"]
        blocked = send("111@s.whatsapp.net", "tres", confirm_token=token)
        assert blocked["success"] is False and blocked["rate_limited"] is True
        assert len(sent["message"]) == 2

    def test_health_unaffected_by_gate(self, client, monkeypatch, env):
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        resp = client.get("/health")
        assert resp.status_code == 503
        assert set(resp.json()) == {"ok", "bridge", "messages_db", "sends_last_hour", "stt"}


# --------------------------------------------------------------------------
# Outbox
# --------------------------------------------------------------------------


class TestPutOutbox:
    @pytest.mark.parametrize(
        "bad", ["../x", "..", ".", "", "   ", ".hidden", "dir/file.txt", "dir\\file.txt", "a\x00b"]
    )
    def test_rejects_bad_filenames(self, server, env, bad):
        result = server.tool("put_outbox")(bad, content_base64=base64.b64encode(b"x").decode())
        assert result["success"] is False
        assert result["path"] is None
        assert list((env / "outbox").iterdir()) == []

    def test_requires_exactly_one_source(self, server):
        put = server.tool("put_outbox")
        neither = put("a.txt")
        assert neither["success"] is False and "exactly one" in neither["message"]
        both = put("a.txt", content_base64="eA==", url="https://example.com/a.txt")
        assert both["success"] is False and "exactly one" in both["message"]

    def test_stores_base64_and_never_overwrites(self, server, env):
        put = server.tool("put_outbox")
        first = put("nota.txt", content_base64=base64.b64encode(b"hola").decode())
        assert first["success"] is True
        assert first["bytes"] == 4
        assert first["path"] == str(env / "outbox" / "nota.txt")
        assert open(first["path"], "rb").read() == b"hola"

        second = put("nota.txt", content_base64=base64.b64encode(b"otra").decode())
        third = put("nota.txt", content_base64=base64.b64encode(b"otra").decode())
        assert second["path"] == str(env / "outbox" / "nota-1.txt")
        assert third["path"] == str(env / "outbox" / "nota-2.txt")
        assert open(first["path"], "rb").read() == b"hola"

    def test_rejects_invalid_base64_and_oversize(self, server, env, monkeypatch):
        put = server.tool("put_outbox")
        assert put("a.bin", content_base64="not base64!!")["success"] is False
        monkeypatch.setattr(base_server, "OUTBOX_MAX_BYTES", 3)
        assert put("a.bin", content_base64=base64.b64encode(b"1234").decode())["success"] is False
        assert list((env / "outbox").iterdir()) == []

    def test_fetches_url_streaming(self, server, env, monkeypatch):
        seen = {}

        class FakeResponse:
            headers = {"Content-Length": "6"}

            def __enter__(self):
                return self

            def __exit__(self, *exc):
                return False

            def raise_for_status(self):
                pass

            def iter_content(self, chunk_size):
                yield b"abc"
                yield b"def"

        def fake_get(url, stream=False, timeout=None, **kw):
            seen.update(url=url, stream=stream, timeout=timeout)
            return FakeResponse()

        monkeypatch.setattr(requests, "get", fake_get)
        result = server.tool("put_outbox")("doc.pdf", url="https://example.com/doc.pdf")
        assert result["success"] is True and result["bytes"] == 6
        assert open(result["path"], "rb").read() == b"abcdef"
        assert seen == {"url": "https://example.com/doc.pdf", "stream": True, "timeout": 30}

    def test_rejects_non_http_url_and_http_errors(self, server, env, monkeypatch):
        put = server.tool("put_outbox")
        assert put("a.txt", url="file:///etc/passwd")["success"] is False
        assert put("a.txt", url="ftp://example.com/a")["success"] is False

        def fake_get(url, **kw):
            raise requests.ConnectionError("nope")

        monkeypatch.setattr(requests, "get", fake_get)
        assert put("a.txt", url="https://example.com/a.txt")["success"] is False
        assert list((env / "outbox").iterdir()) == []


# --------------------------------------------------------------------------
# /health
# --------------------------------------------------------------------------


def _fake_messages_db(path, ts="2026-09-06T10:00:00"):
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE messages (id TEXT, timestamp TEXT)")
    conn.execute("INSERT INTO messages VALUES ('m1', ?)", (ts,))
    conn.commit()
    conn.close()


class TestHealth:
    def test_503_when_bridge_unreachable(self, client, monkeypatch, env):
        def fake_get(url, **kw):
            raise requests.ConnectionError("refused")

        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        resp = client.get("/health")
        assert resp.status_code == 503
        body = resp.json()
        assert body["ok"] is False
        assert "ConnectionError" in body["bridge"]["error"]
        assert body["messages_db"] == {"path": str(env / "missing.db"), "exists": False}
        assert body["sends_last_hour"] == 0
        assert body["stt"] == {"model": "large-v3", "loaded": False}

    def test_200_when_bridge_ok(self, client, monkeypatch, env):
        seen = {}

        class FakeResponse:
            status_code = 200

            def json(self):
                return {"status": "ok", "connected": True, "logged_in": True}

        def fake_get(url, headers=None, timeout=None, **kw):
            seen.update(url=url, headers=headers, timeout=timeout)
            return FakeResponse()

        monkeypatch.setattr(requests, "get", fake_get)
        monkeypatch.setattr(whatsapp, "WHATSAPP_API_BASE_URL", "http://bridge.test/api")
        monkeypatch.setattr(whatsapp, "_bridge_headers", lambda: {"Authorization": "Bearer bridge"})
        db = env / "messages.db"
        _fake_messages_db(str(db))
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(db))

        resp = client.get("/health")
        assert resp.status_code == 200
        body = resp.json()
        assert body["ok"] is True
        assert body["bridge"] == {"status": "ok", "connected": True, "logged_in": True}
        assert body["messages_db"]["exists"] is True
        assert body["messages_db"]["size_bytes"] > 0
        assert body["messages_db"]["last_message_at"] == "2026-09-06T10:00:00"
        assert seen == {
            "url": "http://bridge.test/api/health",
            "headers": {"Authorization": "Bearer bridge"},
            "timeout": 5,
        }

    @pytest.mark.parametrize(
        ("payload", "expected"),
        [
            ({"status": "ok", "connected": True, "logged_in": False}, 503),
            ({"status": "disconnected", "connected": False}, 503),
            ({"status": "not_paired", "connected": True}, 503),
            ({"status": "ok", "connected": True}, 200),  # missing logged_in falls back to connected
            ({"status": "ok", "connected": False}, 503),
        ],
    )
    def test_status_mapping(self, client, monkeypatch, env, payload, expected):
        class FakeResponse:
            status_code = 200

            def json(self):
                return payload

        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        assert client.get("/health").status_code == expected

    def test_bridge_http_error_is_503(self, client, monkeypatch, env):
        class FakeResponse:
            status_code = 401
            text = "unauthorized"

        monkeypatch.setattr(requests, "get", lambda *a, **k: FakeResponse())
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        resp = client.get("/health")
        assert resp.status_code == 503
        assert resp.json()["bridge"]["http_status"] == 401

    def test_health_does_not_write_messages_db(self, client, monkeypatch, env):
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
        db = env / "messages.db"
        _fake_messages_db(str(db))
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(db))
        before = db.stat().st_mtime_ns
        client.get("/health")
        assert db.stat().st_mtime_ns == before
        assert not os.path.exists(str(db) + "-wal")


# --------------------------------------------------------------------------
# Auth middleware
# --------------------------------------------------------------------------


class TestAuth:
    def test_mcp_without_bearer_is_401(self, client):
        resp = client.post("/mcp", json={})
        assert resp.status_code == 401
        assert resp.headers["WWW-Authenticate"] == "Bearer"
        assert client.get("/mcp").status_code == 401

    def test_wrong_bearer_is_401(self, client):
        assert client.post("/mcp", json={}, headers={"Authorization": "Bearer nope"}).status_code == 401
        assert client.post("/mcp", json={}, headers={"Authorization": "Basic dGVzdHRva2Vu"}).status_code == 401

    def test_health_stays_open(self, client, monkeypatch, env):
        monkeypatch.setattr(requests, "get", lambda *a, **k: (_ for _ in ()).throw(requests.ConnectionError("x")))
        monkeypatch.setattr(whatsapp, "MESSAGES_DB_PATH", str(env / "missing.db"))
        assert client.get("/health").status_code == 503  # reached the handler, not 401

    def test_wrong_host_is_421(self, app, monkeypatch):
        foreign = TestClient(app)  # Host: testserver
        assert foreign.get("/health").status_code == 421
        assert foreign.post("/mcp", json={}, headers=AUTH).status_code == 421
        evil = TestClient(app, base_url="http://evil.example.com")
        assert evil.get("/health", headers=AUTH).status_code == 421

    @pytest.mark.parametrize("host", ["localhost", "127.0.0.1", "127.0.0.1:8804", "localhost:8804", "[::1]:8804"])
    def test_local_hosts_pass(self, app, host):
        client = TestClient(app, base_url="http://localhost")
        # Explicit Host header: TestClient cannot parse an IPv6 base_url.
        assert client.get("/mcp", headers={"Host": host}).status_code == 401  # past the Host check, stopped at auth

    def test_authorized_mcp_initialize_round_trip(self, app):
        init = {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "initialize",
            "params": {
                "protocolVersion": "2025-03-26",
                "capabilities": {},
                "clientInfo": {"name": "t", "version": "0"},
            },
        }
        headers = {**AUTH, "Accept": "application/json, text/event-stream", "Content-Type": "application/json"}
        with TestClient(app, base_url="http://127.0.0.1:8804") as live:
            resp = live.post("/mcp", json=init, headers=headers)
            assert resp.status_code == 200, resp.text
            assert "whatsapp" in resp.text

    def test_no_auth_mode_skips_bearer(self):
        async def inner(scope, receive, send):
            await PlainTextResponse("in")(scope, receive, send)

        client = TestClient(LocalBearerAuth(inner, token=None, allow_no_auth=True), base_url="http://localhost")
        assert client.get("/mcp").status_code == 200
        strict = TestClient(LocalBearerAuth(inner, token=None, allow_no_auth=False), base_url="http://localhost")
        assert strict.get("/mcp").status_code == 401
