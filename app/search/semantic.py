"""Смысловое сходство через эмбеддинги (sentence-transformers, модель с Hugging Face).

По умолчанию — deepvk/USER2-small: русскоязычный энкодер на 34M параметров
(Apache-2.0). Названия записей короткие и часто повторяются, поэтому их векторы
кэшируются по тексту: 1000 записей из CSV — это ~380 уникальных названий.
"""

from __future__ import annotations

import logging
import threading

import numpy as np

log = logging.getLogger(__name__)


class SemanticIndex:
    def __init__(self, model_name: str = "deepvk/USER2-small", query_prefix: str = "search_query: ",
                 document_prefix: str = "search_document: "):
        self.model_name = model_name
        self.query_prefix = query_prefix
        self.document_prefix = document_prefix
        self._model = None
        self._cache: dict[str, np.ndarray] = {}
        self._lock = threading.Lock()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer

            self._model = SentenceTransformer(self.model_name, device="cpu")
            log.info("Модель эмбеддингов %s загружена", self.model_name)
        return self._model

    def warm_up(self) -> None:
        with self._lock:
            self._load()

    def _encode(self, texts: list[str]) -> np.ndarray:
        return self._load().encode(texts, normalize_embeddings=True, batch_size=64, convert_to_numpy=True)

    def similarities(self, query: str, documents: list[str]) -> np.ndarray:
        """Косинусное сходство запроса с каждым документом (векторы нормированы)."""
        with self._lock:
            missing = [d for d in dict.fromkeys(documents) if d not in self._cache]
            if missing:
                for text, vector in zip(missing, self._encode([self.document_prefix + d for d in missing])):
                    self._cache[text] = vector
            query_vector = self._encode([self.query_prefix + query])[0]
            matrix = np.stack([self._cache[d] for d in documents]) if documents else np.zeros((0, len(query_vector)))
        return matrix @ query_vector
