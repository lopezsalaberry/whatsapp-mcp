"""Tests for the faster-whisper transcriber (model, ffmpeg and download all faked)."""

import subprocess
import threading
from dataclasses import dataclass
from types import SimpleNamespace

import pytest

import transcribe
from transcribe import Transcriber


@dataclass
class FakeSegment:
    text: str


class FakeModel:
    def __init__(self, text: str = "hola  que tal", language: str = "es", block: threading.Event | None = None):
        self.calls: list[dict] = []
        self.text = text
        self.language = language
        self.block = block

    def transcribe(self, wav, language=None, vad_filter=True, beam_size=5):
        self.calls.append({"wav": wav, "language": language, "vad_filter": vad_filter, "beam_size": beam_size})
        if self.block is not None:
            self.block.wait(timeout=5)
        segments = [FakeSegment(" hola "), FakeSegment("que tal ")]
        info = SimpleNamespace(language=self.language, duration=3.5)
        return iter(segments), info


class FakeRun:
    """Stand-in for subprocess.run that answers ffprobe/ffmpeg."""

    def __init__(self, duration: float = 3.5, probe_rc: int = 0, ffmpeg_rc: int = 0):
        self.duration = duration
        self.probe_rc = probe_rc
        self.ffmpeg_rc = ffmpeg_rc
        self.commands: list[list[str]] = []

    def __call__(self, cmd, **kwargs):
        self.commands.append(list(cmd))
        if cmd[0] == "ffprobe":
            return subprocess.CompletedProcess(cmd, self.probe_rc, stdout=f"{self.duration}\n", stderr="boom")
        if cmd[0] == "ffmpeg":
            if self.ffmpeg_rc == 0:
                with open(cmd[-1], "wb") as fh:
                    fh.write(b"RIFF")
            return subprocess.CompletedProcess(cmd, self.ffmpeg_rc, stdout="", stderr="ffmpeg boom")
        raise AssertionError(f"unexpected command {cmd}")


@pytest.fixture
def media(tmp_path):
    path = tmp_path / "voice.ogg"
    path.write_bytes(b"OggS")
    return str(path)


@pytest.fixture
def download(media):
    calls: list[tuple[str, str]] = []

    def _download(message_id: str, chat_jid: str):
        calls.append((message_id, chat_jid))
        return media

    _download.calls = calls  # type: ignore[attr-defined]
    return _download


@pytest.fixture
def transcriber(tmp_path, monkeypatch):
    fake_run = FakeRun()
    monkeypatch.setattr(transcribe.subprocess, "run", fake_run)
    t = Transcriber(str(tmp_path / "state.db"), sync_max_s=45, idle_unload_s=0, default_language="es")
    t._model_factory = lambda: FakeModel()
    t.fake_run = fake_run  # type: ignore[attr-defined]
    return t


class TestSyncPath:
    def test_transcribes_inline_and_caches(self, transcriber, download):
        result = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert result["status"] == "done"
        assert result["text"] == "hola que tal"
        assert result["language"] == "es"
        assert result["duration_s"] == pytest.approx(3.5)
        assert result["cached"] is False
        assert transcriber.loaded is True
        assert download.calls == [("MSG1", "111@s.whatsapp.net")]

        probe, convert = transcriber.fake_run.commands
        assert probe[:2] == ["ffprobe", "-v"] and probe[-1].endswith("voice.ogg")
        assert convert[0] == "ffmpeg" and "-ar" in convert and convert[-1].endswith(".wav")

        again = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert again["status"] == "done"
        assert again["cached"] is True
        assert again["text"] == "hola que tal"
        assert len(download.calls) == 1  # no second download

    def test_language_hint_and_default(self, transcriber, download):
        transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        transcriber.transcribe("MSG2", "111@s.whatsapp.net", language="en", download_fn=download)
        calls = transcriber._model.calls
        assert calls[0]["language"] == "es"
        assert calls[1]["language"] == "en"
        assert calls[0]["vad_filter"] is True and calls[0]["beam_size"] == 5

    def test_model_loaded_once_and_wav_removed(self, transcriber, download):
        factory_calls = []

        def factory():
            factory_calls.append(1)
            return FakeModel()

        transcriber._model_factory = factory
        transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        transcriber.transcribe("MSG2", "111@s.whatsapp.net", download_fn=download)
        assert len(factory_calls) == 1
        import os

        wav = transcriber.fake_run.commands[1][-1]
        assert not os.path.exists(wav)

    def test_cache_persists_across_instances(self, tmp_path, transcriber, download):
        transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        other = Transcriber(str(tmp_path / "state.db"))
        other._model_factory = lambda: pytest.fail("model must not load on cache hit")
        result = other.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert result["cached"] is True and result["text"] == "hola que tal"
        assert other.loaded is False


class TestPendingPath:
    def test_long_audio_goes_to_background(self, tmp_path, monkeypatch, download):
        fake_run = FakeRun(duration=120.0)
        monkeypatch.setattr(transcribe.subprocess, "run", fake_run)
        gate = threading.Event()
        model = FakeModel(block=gate)
        t = Transcriber(str(tmp_path / "state.db"), sync_max_s=10, idle_unload_s=0)
        t._model_factory = lambda: model

        first = t.transcribe("LONG", "111@s.whatsapp.net", download_fn=download)
        assert first["status"] == "pending"
        assert first["text"] is None
        assert first["duration_s"] == pytest.approx(120.0)

        # Still pending while the worker is blocked; no second download or model call.
        second = t.transcribe("LONG", "111@s.whatsapp.net", download_fn=download)
        assert second["status"] == "pending"
        assert len(download.calls) == 1

        gate.set()
        deadline = threading.Event()
        for _ in range(100):
            done = t.transcribe("LONG", "111@s.whatsapp.net", download_fn=download)
            if done["status"] == "done":
                break
            deadline.wait(0.05)
        assert done["status"] == "done"
        assert done["cached"] is True
        assert done["text"] == "hola que tal"


class TestErrorPath:
    def test_download_failure(self, transcriber):
        result = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=lambda m, c: None)
        assert result["status"] == "error"
        assert "download" in result["message"]
        assert transcriber.loaded is False

    def test_ffprobe_failure(self, tmp_path, monkeypatch, download):
        monkeypatch.setattr(transcribe.subprocess, "run", FakeRun(probe_rc=1))
        t = Transcriber(str(tmp_path / "state.db"))
        t._model_factory = lambda: FakeModel()
        result = t.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert result["status"] == "error"
        assert result["message"].startswith("ffprobe")

    def test_ffmpeg_failure(self, tmp_path, monkeypatch, download):
        monkeypatch.setattr(transcribe.subprocess, "run", FakeRun(ffmpeg_rc=1))
        t = Transcriber(str(tmp_path / "state.db"))
        t._model_factory = lambda: FakeModel()
        result = t.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert result["status"] == "error"
        assert result["message"].startswith("ffmpeg")
        assert result["duration_s"] == pytest.approx(3.5)

    def test_model_failure_is_recorded_and_retried(self, transcriber, download):
        class Broken:
            def transcribe(self, *a, **k):
                raise RuntimeError("no weights")

        transcriber._model_factory = lambda: Broken()
        result = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert result["status"] == "error"
        assert "no weights" in result["message"]

        # An error row is retried on the next call, not served from cache.
        transcriber._model = None
        transcriber._model_factory = lambda: FakeModel()
        retry = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
        assert retry["status"] == "done" and retry["cached"] is False

    def test_download_fn_raising_never_propagates(self, transcriber):
        def boom(m, c):
            raise OSError("bridge down")

        result = transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=boom)
        assert result["status"] == "error"
        assert "bridge down" in result["message"]


def test_unload_drops_model(transcriber, download):
    transcriber.transcribe("MSG1", "111@s.whatsapp.net", download_fn=download)
    assert transcriber.loaded
    transcriber.unload()
    assert not transcriber.loaded
