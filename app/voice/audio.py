"""WAV: чтение того, что прислал браузер, и упаковка синтезированной речи."""

from __future__ import annotations

import io
import wave

import numpy as np


class AudioError(ValueError):
    pass


def read_wav(data: bytes) -> tuple[np.ndarray, int]:
    """WAV (PCM 16 бит) → моно-сигнал float32 в диапазоне [-1, 1] и частота дискретизации."""
    try:
        with wave.open(io.BytesIO(data), "rb") as wav:
            channels, width, rate = wav.getnchannels(), wav.getsampwidth(), wav.getframerate()
            frames = wav.readframes(wav.getnframes())
    except (wave.Error, EOFError) as exc:
        raise AudioError("ожидается WAV-файл") from exc
    if width != 2:
        raise AudioError("поддерживается только PCM 16 бит")
    samples = np.frombuffer(frames, dtype="<i2").astype(np.float32) / 32768.0
    if channels > 1:
        samples = samples.reshape(-1, channels).mean(axis=1)
    return samples, rate


def to_wav(samples: np.ndarray, rate: int) -> bytes:
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return buffer.getvalue()
