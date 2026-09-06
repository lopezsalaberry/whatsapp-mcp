"""Tests for the SQLite-backed send guard (rate limit + dedup)."""

import pytest

from send_guard import SendGuard, content_hash


class FakeClock:
    def __init__(self, start: float = 1_000_000.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def guard(tmp_path, clock) -> SendGuard:
    return SendGuard(str(tmp_path / "state.db"), max_per_hour=3, dedup_seconds=300, clock=clock)


def _send(guard: SendGuard, recipient: str, text: str, key: str = "", ok: bool = True) -> None:
    digest = content_hash("message", recipient, text)
    decision = guard.check(recipient, "message", digest, key)
    assert decision.kind == "allow", decision
    guard.record(recipient, "message", digest, key, ok, {"success": ok, "message": "sent" if ok else "failed"})


class TestRateLimit:
    def test_allows_up_to_limit_then_blocks(self, guard, clock):
        for i in range(3):
            _send(guard, "111@s.whatsapp.net", f"msg {i}")
        assert guard.sends_last_hour() == 3

        decision = guard.check("222@s.whatsapp.net", "message", content_hash("message", "222", "x"))
        assert decision.kind == "rate_limited"
        assert decision.sends_last_hour == 3
        assert decision.resets_in_s == pytest.approx(3600.0)

    def test_failed_attempts_count_toward_limit(self, guard):
        for i in range(3):
            _send(guard, "111@s.whatsapp.net", f"msg {i}", ok=False)
        decision = guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "new"))
        assert decision.kind == "rate_limited"

    def test_resets_after_window(self, guard, clock):
        _send(guard, "111@s.whatsapp.net", "a")
        clock.advance(1000)
        _send(guard, "111@s.whatsapp.net", "b")
        _send(guard, "111@s.whatsapp.net", "c")
        assert guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "d")).kind == "rate_limited"
        assert guard.resets_in_s() == pytest.approx(2600.0)

        clock.advance(2601)  # first send ages out of the window
        assert guard.sends_last_hour() == 2
        assert guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "d")).kind == "allow"

    def test_resets_in_zero_when_empty(self, guard):
        assert guard.resets_in_s() == 0.0
        assert guard.sends_last_hour() == 0


class TestIdempotencyKey:
    def test_same_key_is_duplicate_with_previous_result(self, guard):
        _send(guard, "111@s.whatsapp.net", "hello", key="k1")
        decision = guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "other"), "k1")
        assert decision.kind == "duplicate"
        assert decision.previous == {"success": True, "message": "sent"}

    def test_key_expires_after_24h(self, guard, clock):
        _send(guard, "111@s.whatsapp.net", "hello", key="k1")
        clock.advance(24 * 3600 + 1)
        decision = guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "other"), "k1")
        assert decision.kind == "allow"

    def test_empty_key_never_matches(self, guard):
        _send(guard, "111@s.whatsapp.net", "hello", key="")
        decision = guard.check("111@s.whatsapp.net", "message", content_hash("message", "111", "other"), "")
        assert decision.kind == "allow"


class TestContentDedup:
    def test_same_content_within_window_is_duplicate(self, guard, clock):
        _send(guard, "111@s.whatsapp.net", "hello")
        clock.advance(299)
        digest = content_hash("message", "111@s.whatsapp.net", "hello")
        assert guard.check("111@s.whatsapp.net", "message", digest).kind == "duplicate"

    def test_same_content_after_window_is_allowed(self, guard, clock):
        _send(guard, "111@s.whatsapp.net", "hello")
        clock.advance(301)
        digest = content_hash("message", "111@s.whatsapp.net", "hello")
        assert guard.check("111@s.whatsapp.net", "message", digest).kind == "allow"

    def test_different_recipient_or_kind_is_not_duplicate(self, guard):
        _send(guard, "111@s.whatsapp.net", "hello")
        digest = content_hash("message", "111@s.whatsapp.net", "hello")
        assert guard.check("222@s.whatsapp.net", "message", digest).kind == "allow"
        assert guard.check("111@s.whatsapp.net", "file", digest).kind == "allow"

    def test_failed_send_is_not_deduplicated(self, guard):
        _send(guard, "111@s.whatsapp.net", "hello", ok=False)
        digest = content_hash("message", "111@s.whatsapp.net", "hello")
        assert guard.check("111@s.whatsapp.net", "message", digest).kind == "allow"

    def test_dedup_takes_precedence_over_rate_limit(self, guard):
        for i in range(3):
            _send(guard, "111@s.whatsapp.net", f"msg {i}")
        digest = content_hash("message", "111@s.whatsapp.net", "msg 0")
        assert guard.check("111@s.whatsapp.net", "message", digest).kind == "duplicate"


class TestPersistence:
    def test_state_survives_new_instance(self, tmp_path, clock):
        path = str(tmp_path / "state.db")
        first = SendGuard(path, max_per_hour=2, dedup_seconds=300, clock=clock)
        _send(first, "111@s.whatsapp.net", "hello", key="k1")
        _send(first, "111@s.whatsapp.net", "world")
        first.close()

        second = SendGuard(path, max_per_hour=2, dedup_seconds=300, clock=clock)
        assert second.sends_last_hour() == 2
        digest = content_hash("message", "111@s.whatsapp.net", "hello")
        assert second.check("111@s.whatsapp.net", "message", digest).kind == "duplicate"
        assert second.check("111@s.whatsapp.net", "message", "zzz", "k1").kind == "duplicate"
        assert second.check("111@s.whatsapp.net", "message", "zzz").kind == "rate_limited"

    def test_creates_parent_directory(self, tmp_path, clock):
        path = tmp_path / "nested" / "dir" / "state.db"
        SendGuard(str(path), clock=clock)
        assert path.exists()


def test_content_hash_is_stable_and_distinct():
    assert content_hash("message", "a", "b") == content_hash("message", "a", "b")
    assert content_hash("message", "a", "b") != content_hash("message", "ab", "")
    assert content_hash("file", "a", None) == content_hash("file", "a", "")


class TestConfirmTokens:
    def test_issue_and_consume_once(self, guard):
        token = guard.issue_token("hash-a")
        assert len(token) >= 24
        assert guard.consume_token(token, "hash-a") is None
        assert guard.consume_token(token, "hash-a") == "TOKEN_USED"

    def test_error_codes(self, guard, clock):
        assert guard.consume_token("nope", "hash-a") == "TOKEN_INVALID"
        token = guard.issue_token("hash-a")
        assert guard.consume_token(token, "hash-b") == "PAYLOAD_MISMATCH"
        assert guard.consume_token(token, "hash-a") is None  # mismatch did not burn it
        expired = guard.issue_token("hash-c")
        clock.advance(601)
        assert guard.consume_token(expired, "hash-c") == "TOKEN_EXPIRED"

    def test_tokens_are_unique_and_persist(self, tmp_path, clock):
        path = str(tmp_path / "state.db")
        first = SendGuard(path, clock=clock)
        tokens = {first.issue_token("h") for _ in range(20)}
        assert len(tokens) == 20
        first.close()
        second = SendGuard(path, clock=clock)
        assert second.consume_token(tokens.pop(), "h") is None

    def test_purge_expired(self, guard, clock):
        guard.issue_token("h")
        clock.advance(25 * 3600)
        assert guard.purge_expired_tokens() == 1
        assert guard.purge_expired_tokens() == 0
