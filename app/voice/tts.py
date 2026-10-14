"""Синтез речи на сервере: Silero TTS v5 (русский).

Самая быстрая из открытых русских моделей (секунды речи за сотые доли секунды
на CPU), сама расставляет ударения и букву «ё», в том числе в омографах.
Цифры модель не читает, поэтому текст сначала проходит через `speakable`.
Лицензия Silero разрешает бесплатное некоммерческое использование.
"""

from __future__ import annotations

import logging
import threading

from app.voice.audio import to_wav
from app.voice.speakable import speakable

log = logging.getLogger(__name__)

SAMPLE_RATE = 24000
MAX_CHARS = 900  # ограничение Silero на длину одного фрагмента ~1000 символов


class SpeechSynthesizer:
    def __init__(self, speaker: str = "xenia", model_id: str = "v5_ru"):
        self.speaker = speaker
        self.model_id = model_id
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"silero-{self.model_id}/{self.speaker}"

    def _load(self):
        if self._model is None:
            from silero import silero_tts

            self._model, _ = silero_tts(language="ru", speaker=self.model_id)
            log.info("Модель синтеза Silero %s загружена", self.model_id)
        return self._model

    def warm_up(self) -> None:
        with self._lock:
            self._load()

    def synthesize(self, text: str) -> bytes:
        prepared = speakable(text)[:MAX_CHARS]
        if not prepared:
            prepared = "Готово."
        with self._lock:
            audio = self._load().apply_tts(
                text=prepared, speaker=self.speaker, sample_rate=SAMPLE_RATE,
                put_accent=True, put_yo=True, put_stress_homo=True, put_yo_homo=True,
            )
        return to_wav(audio.numpy(), SAMPLE_RATE)
