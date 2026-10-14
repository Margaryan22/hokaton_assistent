"""Распознавание речи на сервере: GigaAM v3 (Сбер, MIT) через onnx-asr.

Вариант `e2e-rnnt` сразу выдаёт текст с пунктуацией и цифрами («завтра в 10:00»),
работает на CPU без PyTorch и распознаёт фразу за десятки миллисекунд.
Модель скачивается с Hugging Face (istupakov/gigaam-v3-onnx) при первом запросе;
версия int8 весит 225 МБ вместо 885 МБ и на наших фразах распознаёт так же.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

from app.voice.audio import AudioError, read_wav

log = logging.getLogger(__name__)

MAX_SECONDS = 25  # модели GigaAM рассчитаны на фразы до 25–30 секунд
MIN_SECONDS = 0.3


class SpeechRecognizer:
    def __init__(self, model_name: str = "gigaam-v3-e2e-rnnt", quantization: str | None = "int8"):
        self.model_name = model_name
        self.quantization = quantization
        self._model = None
        self._lock = threading.Lock()

    @property
    def name(self) -> str:
        return f"{self.model_name} ({self.quantization})" if self.quantization else self.model_name

    def _load(self):
        if self._model is None:
            import onnx_asr

            # CoreML-провайдер onnxruntime не справляется с этой моделью, CPU быстрее и надёжнее.
            self._model = onnx_asr.load_model(self.model_name, quantization=self.quantization,
                                              providers=["CPUExecutionProvider"])
            log.info("Модель распознавания %s загружена", self.model_name)
        return self._model

    def warm_up(self) -> None:
        with self._lock:
            self._load()

    def transcribe(self, wav_bytes: bytes) -> str:
        samples, rate = read_wav(wav_bytes)
        duration = len(samples) / rate
        if duration > MAX_SECONDS:
            raise AudioError(f"запись длиннее {MAX_SECONDS} секунд")
        if duration < MIN_SECONDS or float(np.abs(samples).max(initial=0)) < 1e-3:
            return ""
        with self._lock:
            text = self._load().recognize(samples, sample_rate=rate)
        return " ".join(str(text).split())
