"""SQLite-backed guard for outbound WhatsApp sends.

Three rules, evaluated in order by :meth:`SendGuard.check`:

1. A non-empty ``idempotency_key`` already recorded in the last 24 h -> duplicate.
2. The same ``(recipient, kind, content_hash)`` sent successfully within
   ``dedup_seconds`` -> duplicate.
3. ``max_per_hour`` or more allowed attempts (successful or not) in the last
   hour -> rate limited.

Only sends that passed the guard are recorded (see :meth:`SendGuard.record`),
so the ``sends`` table is the audit trail of everything that was attempted.
Message content is never stored, only its hash and the bridge's result.

The same DB also holds the two-step confirm tokens (``confirm_tokens``),
mirroring mail-mcp: :meth:`SendGuard.issue_token` binds a fresh token to a
payload hash for ``TOKEN_TTL_SECONDS``; :meth:`SendGuard.consume_token` spends
it atomically and single-use, so a retried call after a send in flight gets
``TOKEN_USED`` and never a second message.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import secrets
import sqlite3
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any, Literal

logger = logging.getLogger("whatsapp.send_guard")

HOUR_S = 3600.0
IDEMPOTENCY_WINDOW_S = 24 * HOUR_S
TOKEN_TTL_SECONDS = 600  # 10 minutes, same as mail-mcp

DecisionKind = Literal["allow", "duplicate", "rate_limited"]


@dataclass(frozen=True)
class Decision:
    """Outcome of :meth:`SendGuard.check`."""

    kind: DecisionKind
    sends_last_hour: int = 0
    resets_in_s: float = 0.0
    previous: dict[str, Any] | None = field(default=None)

    @property
    def allowed(self) -> bool:
        return self.kind == "allow"


def content_hash(*parts: str | None) -> str:
    """SHA-256 of the canonical payload (NUL-joined parts, empty for None)."""
    canonical = "\x00".join("" if part is None else str(part) for part in parts)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class SendGuard:
    """Rate limit + dedup for outbound sends, persisted in ``db_path``."""

    def __init__(
        self,
        db_path: str,
        max_per_hour: int = 30,
        dedup_seconds: float = 300.0,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self.db_path = db_path
        self.max_per_hour = int(max_per_hour)
        self.dedup_seconds = float(dedup_seconds)
        # Resolved lazily so monkeypatching ``time.time`` in tests takes effect.
        self._clock: Callable[[], float] = clock or (lambda: time.time())
        self._lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(db_path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self._conn = sqlite3.connect(db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS sends (
                    id INTEGER PRIMARY KEY,
                    ts REAL NOT NULL,
                    recipient TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    idempotency_key TEXT,
                    message_id TEXT,
                    ok INTEGER NOT NULL,
                    result TEXT
                )
                """
            )
            self._conn.execute("CREATE INDEX IF NOT EXISTS sends_ts ON sends(ts)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS sends_key ON sends(idempotency_key, ts)")
            self._conn.execute("CREATE INDEX IF NOT EXISTS sends_content ON sends(recipient, kind, content_hash, ts)")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS confirm_tokens (
                    token TEXT PRIMARY KEY,
                    payload_sha256 TEXT NOT NULL,
                    created_at REAL NOT NULL,
                    used_at REAL
                )
                """
            )
            self._conn.commit()

    # -- queries -----------------------------------------------------------

    def _count_last_hour(self, now: float) -> int:
        row = self._conn.execute("SELECT COUNT(*) AS n FROM sends WHERE ts > ?", (now - HOUR_S,)).fetchone()
        return int(row["n"]) if row else 0

    def _resets_in(self, now: float) -> float:
        row = self._conn.execute("SELECT MIN(ts) AS oldest FROM sends WHERE ts > ?", (now - HOUR_S,)).fetchone()
        if not row or row["oldest"] is None:
            return 0.0
        return max(0.0, float(row["oldest"]) + HOUR_S - now)

    def sends_last_hour(self) -> int:
        """Number of allowed attempts (any outcome) in the trailing hour."""
        with self._lock:
            return self._count_last_hour(self._clock())

    def resets_in_s(self) -> float:
        """Seconds until the oldest attempt in the window ages out (0 when empty)."""
        with self._lock:
            return self._resets_in(self._clock())

    # -- decisions ---------------------------------------------------------

    def check(self, recipient: str, kind: str, content_hash: str, idempotency_key: str = "") -> Decision:
        """Decide whether a send may proceed. Does not record anything."""
        with self._lock:
            now = self._clock()
            count = self._count_last_hour(now)

            if idempotency_key:
                row = self._conn.execute(
                    "SELECT result FROM sends WHERE idempotency_key = ? AND ts > ? ORDER BY ts DESC LIMIT 1",
                    (idempotency_key, now - IDEMPOTENCY_WINDOW_S),
                ).fetchone()
                if row is not None:
                    logger.info("guard: duplicate idempotency_key kind=%s recipient=%s", kind, recipient)
                    return Decision("duplicate", count, self._resets_in(now), _load_result(row["result"]))

            if self.dedup_seconds > 0:
                row = self._conn.execute(
                    "SELECT result FROM sends WHERE recipient = ? AND kind = ? AND content_hash = ? "
                    "AND ok = 1 AND ts > ? ORDER BY ts DESC LIMIT 1",
                    (recipient, kind, content_hash, now - self.dedup_seconds),
                ).fetchone()
                if row is not None:
                    logger.info("guard: duplicate content kind=%s recipient=%s", kind, recipient)
                    return Decision("duplicate", count, self._resets_in(now), _load_result(row["result"]))

            if count >= self.max_per_hour:
                resets = self._resets_in(now)
                logger.warning(
                    "guard: rate limited kind=%s recipient=%s sends_last_hour=%d limit=%d",
                    kind,
                    recipient,
                    count,
                    self.max_per_hour,
                )
                return Decision("rate_limited", count, resets)

            return Decision("allow", count, self._resets_in(now))

    def record(
        self,
        recipient: str,
        kind: str,
        content_hash: str,
        idempotency_key: str,
        ok: bool,
        result: dict[str, Any] | None,
        message_id: str | None = None,
    ) -> None:
        """Persist an attempt that passed :meth:`check` (whatever the bridge answered)."""
        payload = json.dumps(result or {}, default=str, ensure_ascii=False)
        with self._lock:
            self._conn.execute(
                "INSERT INTO sends (ts, recipient, kind, content_hash, idempotency_key, message_id, ok, result) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    self._clock(),
                    recipient,
                    kind,
                    content_hash,
                    idempotency_key or None,
                    message_id,
                    1 if ok else 0,
                    payload,
                ),
            )
            self._conn.commit()

    # -- two-step confirm tokens -------------------------------------------

    def issue_token(self, payload_sha256: str) -> str:
        """Mint a single-use token bound to ``payload_sha256`` (valid TOKEN_TTL_SECONDS)."""
        token = secrets.token_urlsafe(24)
        with self._lock:
            self._conn.execute(
                "INSERT INTO confirm_tokens (token, payload_sha256, created_at) VALUES (?, ?, ?)",
                (token, payload_sha256, self._clock()),
            )
            self._conn.commit()
        return token

    def consume_token(self, token: str, payload_sha256: str) -> str | None:
        """Spend a token atomically. Returns None on success or an error code.

        Codes: TOKEN_INVALID (unknown), PAYLOAD_MISMATCH (token bound to other content),
        TOKEN_USED (already spent -- a retried call never sends twice), TOKEN_EXPIRED.
        """
        with self._lock:
            now = self._clock()
            cur = self._conn.execute(
                "UPDATE confirm_tokens SET used_at = ? WHERE token = ? AND payload_sha256 = ? "
                "AND used_at IS NULL AND created_at > ?",
                (now, token, payload_sha256, now - TOKEN_TTL_SECONDS),
            )
            self._conn.commit()
            if cur.rowcount == 1:
                return None
            row = self._conn.execute(
                "SELECT used_at, created_at, payload_sha256 FROM confirm_tokens WHERE token = ?", (token,)
            ).fetchone()
        if row is None:
            return "TOKEN_INVALID"
        if row["payload_sha256"] != payload_sha256:
            return "PAYLOAD_MISMATCH"
        if row["used_at"] is not None:
            return "TOKEN_USED"
        return "TOKEN_EXPIRED"

    def purge_expired_tokens(self, older_than_s: float = 24 * HOUR_S) -> int:
        """Housekeeping: drop tokens older than ``older_than_s`` (used or not)."""
        with self._lock:
            cur = self._conn.execute("DELETE FROM confirm_tokens WHERE created_at < ?", (self._clock() - older_than_s,))
            self._conn.commit()
            return int(cur.rowcount)

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _load_result(raw: str | None) -> dict[str, Any]:
    if not raw:
        return {}
    try:
        value = json.loads(raw)
    except (TypeError, ValueError):
        return {}
    return value if isinstance(value, dict) else {}
