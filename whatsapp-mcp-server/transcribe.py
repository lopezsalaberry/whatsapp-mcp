"""Voice-note transcription with faster-whisper, cached in the state DB.

Flow for :meth:`Transcriber.transcribe`:

1. Cache hit (``done``) -> returned immediately with ``cached=True``.
2. Already ``pending`` -> returned as pending (a worker thread owns it).
3. Otherwise download the media, measure it with ``ffprobe``, convert it to
   16 kHz mono WAV with ``ffmpeg`` and run the model: inline when the audio is
   at most ``sync_max_s`` seconds, in a daemon thread (returning ``pending``)
   when longer.

The model is loaded lazily on first use (``faster_whisper`` is imported inside
the method, so the module itself has no hard dependency) and unloaded by a
timer after ``idle_unload_s`` seconds without work. Errors are recorded with
status ``error`` and returned, never raised.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import subprocess
import tempfile
import threading
import time
from collections.abc import Callable
from typing import Any

logger = logging.getLogger("whatsapp.transcribe")

DownloadFn = Callable[[str, str], str | None]

_FFPROBE_TIMEOUT_S = 60
_FFMPEG_TIMEOUT_S = 300


def _default_download(message_id: str, chat_jid: str) -> str | None:
    import whatsapp

    return whatsapp.download_media(message_id, chat_jid)


class Transcriber:
    """Cached, lazily-loaded faster-whisper transcriber."""

    def __init__(
        self,
        state_db_path: str,
        model_name: str = "large-v3",
        compute_type: str = "int8",
        cpu_threads: int = 4,
        sync_max_s: float = 45.0,
        idle_unload_s: float = 600.0,
        default_language: str = "es",
    ) -> None:
        self.db_path = state_db_path
        self.model_name = model_name
        self.compute_type = compute_type
        self.cpu_threads = int(cpu_threads)
        self.sync_max_s = float(sync_max_s)
        self.idle_unload_s = float(idle_unload_s)
        self.default_language = default_language

        # Test hook: replace with any callable returning an object with ``.transcribe``.
        self._model_factory: Callable[[], Any] = self._load_model
        self._model: Any = None
        self._model_lock = threading.Lock()
        self._idle_timer: threading.Timer | None = None

        self._db_lock = threading.Lock()
        directory = os.path.dirname(os.path.abspath(state_db_path))
        os.makedirs(directory, mode=0o700, exist_ok=True)
        self._conn = sqlite3.connect(state_db_path, check_same_thread=False, timeout=10)
        self._conn.row_factory = sqlite3.Row
        with self._db_lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA busy_timeout=10000")
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS transcripts (
                    chat_jid TEXT NOT NULL,
                    message_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    text TEXT,
                    language TEXT,
                    duration_s REAL,
                    error TEXT,
                    created REAL NOT NULL,
                    updated REAL NOT NULL,
                    PRIMARY KEY (chat_jid, message_id)
                )
                """
            )
            self._conn.commit()

    # -- model lifecycle ---------------------------------------------------

    @property
    def loaded(self) -> bool:
        return self._model is not None

    def _load_model(self) -> Any:
        from faster_whisper import WhisperModel  # imported lazily: optional dependency

        logger.info("stt: loading model=%s compute=%s threads=%d", self.model_name, self.compute_type, self.cpu_threads)
        started = time.monotonic()
        model = WhisperModel(
            self.model_name, device="cpu", compute_type=self.compute_type, cpu_threads=self.cpu_threads
        )
        logger.info("stt: model loaded in %.1fs", time.monotonic() - started)
        return model

    def _get_model(self) -> Any:
        """Return the model, loading it if needed. Caller must hold ``_model_lock``."""
        if self._model is None:
            self._model = self._model_factory()
        return self._model

    def _touch_idle_timer(self) -> None:
        """(Re)arm the unload timer. Caller must hold ``_model_lock``."""
        if self._idle_timer is not None:
            self._idle_timer.cancel()
            self._idle_timer = None
        if self.idle_unload_s > 0 and self._model is not None:
            timer = threading.Timer(self.idle_unload_s, self.unload)
            timer.daemon = True
            timer.start()
            self._idle_timer = timer

    def unload(self) -> None:
        """Drop the model to free memory. Safe to call at any time."""
        with self._model_lock:
            if self._idle_timer is not None:
                self._idle_timer.cancel()
                self._idle_timer = None
            if self._model is not None:
                logger.info("stt: unloading model after idle")
                self._model = None

    # -- persistence -------------------------------------------------------

    def _get_row(self, chat_jid: str, message_id: str) -> sqlite3.Row | None:
        with self._db_lock:
            return self._conn.execute(
                "SELECT * FROM transcripts WHERE chat_jid = ? AND message_id = ?", (chat_jid, message_id)
            ).fetchone()

    def _upsert(
        self,
        chat_jid: str,
        message_id: str,
        status: str,
        text: str | None = None,
        language: str | None = None,
        duration_s: float | None = None,
        error: str | None = None,
    ) -> None:
        now = time.time()
        with self._db_lock:
            self._conn.execute(
                """
                INSERT INTO transcripts (chat_jid, message_id, status, text, language, duration_s, error, created, updated)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(chat_jid, message_id) DO UPDATE SET
                    status = excluded.status,
                    text = excluded.text,
                    language = excluded.language,
                    duration_s = excluded.duration_s,
                    error = excluded.error,
                    updated = excluded.updated
                """,
                (chat_jid, message_id, status, text, language, duration_s, error, now, now),
            )
            self._conn.commit()

    @staticmethod
    def _row_to_result(row: sqlite3.Row, cached: bool, message: str) -> dict[str, Any]:
        return {
            "status": row["status"],
            "text": row["text"],
            "language": row["language"],
            "duration_s": row["duration_s"],
            "cached": cached,
            "message": message,
        }

    # -- media helpers -----------------------------------------------------

    @staticmethod
    def _probe_duration(path: str) -> float:
        proc = subprocess.run(
            ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFPROBE_TIMEOUT_S,
        )
        if proc.returncode != 0:
            raise RuntimeError(f"ffprobe failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
        raw = (proc.stdout or "").strip().splitlines()
        if not raw:
            raise RuntimeError("ffprobe returned no duration")
        return float(raw[-1].strip())

    @staticmethod
    def _to_wav(src: str) -> str:
        fd, wav_path = tempfile.mkstemp(prefix="wa-stt-", suffix=".wav")
        os.close(fd)
        proc = subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", src, "-ac", "1", "-ar", "16000", "-f", "wav", wav_path],
            capture_output=True,
            text=True,
            check=False,
            timeout=_FFMPEG_TIMEOUT_S,
        )
        if proc.returncode != 0:
            _unlink_quietly(wav_path)
            raise RuntimeError(f"ffmpeg failed (exit {proc.returncode}): {proc.stderr.strip()[:200]}")
        return wav_path

    # -- main entry --------------------------------------------------------

    def transcribe(
        self,
        message_id: str,
        chat_jid: str,
        language: str = "",
        download_fn: DownloadFn | None = None,
    ) -> dict[str, Any]:
        """Transcribe a voice note. Never raises; errors come back as ``status="error"``."""
        try:
            return self._transcribe(message_id, chat_jid, language, download_fn or _default_download)
        except Exception as exc:  # defensive: the tool must never blow up the MCP call
            logger.exception("stt: unexpected failure chat=%s message=%s", chat_jid, message_id)
            error = f"{type(exc).__name__}: {exc}"
            try:
                self._upsert(chat_jid, message_id, "error", error=error)
            except Exception:
                logger.exception("stt: could not persist error state")
            return _error_result(error)

    def _transcribe(self, message_id: str, chat_jid: str, language: str, download_fn: DownloadFn) -> dict[str, Any]:
        row = self._get_row(chat_jid, message_id)
        if row is not None:
            if row["status"] == "done":
                return self._row_to_result(row, cached=True, message="transcript from cache")
            if row["status"] == "pending":
                return self._row_to_result(row, cached=False, message="transcription in progress; call again later")
            # status == "error": retry below

        media_path = download_fn(message_id, chat_jid)
        if not media_path or not os.path.exists(media_path):
            error = "download failed or file missing"
            self._upsert(chat_jid, message_id, "error", error=error)
            return _error_result(error)

        try:
            duration = self._probe_duration(media_path)
        except (RuntimeError, ValueError, OSError, subprocess.SubprocessError) as exc:
            error = f"ffprobe: {exc}"
            self._upsert(chat_jid, message_id, "error", error=error)
            return _error_result(error)

        try:
            wav_path = self._to_wav(media_path)
        except (RuntimeError, OSError, subprocess.SubprocessError) as exc:
            error = f"ffmpeg: {exc}"
            self._upsert(chat_jid, message_id, "error", error=error, duration_s=duration)
            return _error_result(error, duration)

        lang = language or self.default_language
        self._upsert(chat_jid, message_id, "pending", duration_s=duration, language=lang or None)

        if duration <= self.sync_max_s:
            return self._run(chat_jid, message_id, wav_path, lang, duration)

        worker = threading.Thread(
            target=self._run,
            args=(chat_jid, message_id, wav_path, lang, duration),
            name=f"stt-{message_id[:12]}",
            daemon=True,
        )
        worker.start()
        logger.info("stt: queued chat=%s message=%s duration=%.1fs", chat_jid, message_id, duration)
        return {
            "status": "pending",
            "text": None,
            "language": lang or None,
            "duration_s": duration,
            "cached": False,
            "message": f"audio is {duration:.0f}s (> {self.sync_max_s:.0f}s); transcribing in background, call again later",
        }

    def _run(self, chat_jid: str, message_id: str, wav_path: str, language: str, duration: float) -> dict[str, Any]:
        started = time.monotonic()
        try:
            with self._model_lock:
                model = self._get_model()
                segments, info = model.transcribe(wav_path, language=language or None, vad_filter=True, beam_size=5)
                text = " ".join(seg.text.strip() for seg in segments if getattr(seg, "text", "")).strip()
                self._touch_idle_timer()
            detected = getattr(info, "language", None) or language or None
            measured = getattr(info, "duration", None)
            duration_s = float(measured) if measured else duration
            self._upsert(chat_jid, message_id, "done", text=text, language=detected, duration_s=duration_s)
            logger.info(
                "stt: done chat=%s message=%s duration=%.1fs took=%.1fs chars=%d",
                chat_jid,
                message_id,
                duration_s,
                time.monotonic() - started,
                len(text),
            )
            return {
                "status": "done",
                "text": text,
                "language": detected,
                "duration_s": duration_s,
                "cached": False,
                "message": "transcribed",
            }
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            logger.exception("stt: failed chat=%s message=%s", chat_jid, message_id)
            self._upsert(chat_jid, message_id, "error", error=error, duration_s=duration)
            return _error_result(error, duration)
        finally:
            _unlink_quietly(wav_path)


def _error_result(error: str, duration: float | None = None) -> dict[str, Any]:
    return {
        "status": "error",
        "text": None,
        "language": None,
        "duration_s": duration,
        "cached": False,
        "message": error,
    }


def _unlink_quietly(path: str) -> None:
    try:
        os.unlink(path)
    except OSError:
        pass
