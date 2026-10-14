"""Голосовой слой без загрузки моделей: числительные, подготовка текста к озвучке, WAV, эндпоинты."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.container import build_container
from app.main import create_app
from app.voice.audio import AudioError, read_wav, to_wav
from app.voice.numerals import cardinal, ordinal
from app.voice.speakable import speakable
from tests.conftest import make_settings


@pytest.mark.parametrize("n, gender, case, expected", [
    (1, "femn", "nomn", "одна"),
    (2, "femn", "nomn", "две"),
    (21, "femn", "nomn", "двадцать одна"),
    (73, "masc", "nomn", "семьдесят три"),
    (1001, "masc", "nomn", "тысяча один"),
    (2027, "masc", "nomn", "две тысячи двадцать семь"),
    (15, "masc", "gent", "пятнадцати"),
])
def test_cardinal(n, gender, case, expected):
    assert cardinal(n, gender, case) == expected


@pytest.mark.parametrize("n, case, gender, expected", [
    (14, "gent", "neut", "четырнадцатого"),
    (3, "gent", "neut", "третьего"),
    (31, "gent", "neut", "тридцать первого"),
    (2027, "gent", "masc", "две тысячи двадцать седьмого"),
    (2000, "gent", "masc", "двухтысячного"),
])
def test_ordinal(n, case, gender, expected):
    assert ordinal(n, case, gender) == expected


@pytest.mark.parametrize("text, expected", [
    ("напомню завтра, 14 октября, в 10:00: позвонить Ивану.",
     "напомню завтра, четырнадцатого октября, в десять ноль-ноль: позвонить Ивану."),
    ("Нашёл 73 записи", "Нашёл семьдесят три записи"),
    ("1 встреча, 2 задачи, 1 напоминание, 21 запись",
     "одна встреча, две задачи, одно напоминание, двадцать одна запись"),
    ("«Созвон» (15:00–16:00).", "Созвон, с пятнадцати ноль-ноль до шестнадцати ноль-ноль."),
    ("9 января 2027 в 09:05", "девятого января две тысячи двадцать седьмого года в девять ноль пять"),
    ("про CRM", "про си-ар-эм"),
])
def test_speakable(text, expected):
    assert speakable(text) == expected


def test_wav_roundtrip_and_validation():
    samples = np.sin(np.linspace(0, 200, 16000)).astype(np.float32) * 0.5
    restored, rate = read_wav(to_wav(samples, 16000))
    assert rate == 16000 and np.allclose(restored, samples, atol=1e-3)
    with pytest.raises(AudioError):
        read_wav(b"not a wav")


class FakeRecognizer:
    name = "fake-stt"

    def transcribe(self, wav: bytes) -> str:
        read_wav(wav)
        return "Что у меня сегодня?"


class FakeSynthesizer:
    name = "fake-tts"

    def synthesize(self, text: str) -> bytes:
        return to_wav(np.zeros(2400, dtype=np.float32), 24000)


@pytest.fixture
def voice_client():
    container = build_container(make_settings(seed=False), stt=FakeRecognizer(), tts=FakeSynthesizer())
    with TestClient(create_app(container=container)) as client:
        yield client


def test_voice_endpoints(voice_client):
    health = voice_client.get("/api/health").json()
    assert health["server_stt"] and health["server_tts"] and health["models"]["stt"] == "fake-stt"
    wav = to_wav(np.zeros(16000, dtype=np.float32), 16000)
    assert voice_client.post("/api/stt", content=wav, headers={"Content-Type": "audio/wav"}).json() == {"text": "Что у меня сегодня?"}
    response = voice_client.post("/api/tts", json={"text": "Привет"})
    assert response.headers["content-type"] == "audio/wav" and response.content[:4] == b"RIFF"


def test_voice_endpoint_errors(voice_client):
    assert voice_client.post("/api/stt", content=b"garbage").status_code == 422
    assert voice_client.post("/api/stt", content=b"").status_code == 422
    assert voice_client.post("/api/tts", json={"text": ""}).status_code == 422


def test_voice_disabled_without_models():
    container = build_container(make_settings(seed=False))
    with TestClient(create_app(container=container)) as client:
        assert client.get("/api/health").json()["server_stt"] is False
        assert client.post("/api/stt", content=b"x").status_code == 503
        assert client.post("/api/tts", json={"text": "x"}).status_code == 503
