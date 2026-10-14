"""Сборка зависимостей приложения в одном месте."""

from __future__ import annotations

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from app.config import Settings
from app.nlu.rules import RuleBasedParser
from app.nlu.service import NLU, LLMBackend
from app.search.engine import SearchEngine
from app.services.assistant import Assistant
from app.services.planner import Planner
from app.storage.db import Database
from app.storage.importer import import_csv
from app.storage.repository import DialogStateRepository, RecordRepository

log = logging.getLogger(__name__)


@dataclass
class Container:
    settings: Settings
    db: Database
    records: RecordRepository
    planner: Planner
    assistant: Assistant
    nlu: NLU
    stt: object | None = None
    tts: object | None = None
    models: dict[str, str] = field(default_factory=dict)

    def now(self, tz_name: str | None = None) -> datetime:
        """Текущее «настенное» время пользователя (часовой пояс берётся из браузера)."""
        zone = self.settings.zone
        if tz_name:
            try:
                zone = ZoneInfo(tz_name)
            except (ZoneInfoNotFoundError, ValueError):
                pass
        return datetime.now(zone).replace(tzinfo=None, second=0, microsecond=0)


def _build_llm(settings: Settings) -> LLMBackend | None:
    if settings.llm_backend == "local":
        from app.nlu.local_llm import LocalLLMParser

        return LocalLLMParser(settings.local_llm_model, model_path=settings.local_llm_path)
    if settings.llm_backend == "claude":
        import anthropic

        from app.nlu.claude import ClaudeParser

        return ClaudeParser(anthropic.Anthropic(), settings.claude_model, settings.claude_effort, settings.llm_timeout)
    return None


def build_container(settings: Settings, llm: LLMBackend | None = None, *, stt=None, tts=None, semantic=None) -> Container:
    db = Database(settings.db_path)
    records = RecordRepository(db)
    if semantic is None and settings.embeddings_model != "off":
        from app.search.semantic import SemanticIndex

        semantic = SemanticIndex(settings.embeddings_model)
    planner = Planner(records, SearchEngine(semantic=semantic))
    llm = llm if llm is not None else _build_llm(settings)
    nlu = NLU(RuleBasedParser(), llm, settings.timezone)
    if stt is None and settings.stt_model != "off":
        from app.voice.stt import SpeechRecognizer

        stt = SpeechRecognizer(settings.stt_model)
    if tts is None and settings.tts_speaker != "off":
        from app.voice.tts import SpeechSynthesizer

        tts = SpeechSynthesizer(settings.tts_speaker)
    models = {
        "nlu": f"{getattr(llm, 'model_name', getattr(llm, 'model', ''))} + правила" if llm else "правила",
        "stt": stt.name if stt else "браузер (Web Speech API)",
        "tts": tts.name if tts else "браузер (speechSynthesis)",
        "search": f"BM25 + {semantic.model_name}" if semantic else "BM25",
    }
    assistant = Assistant(nlu, planner, DialogStateRepository(db))
    container = Container(settings, db, records, planner, assistant, nlu, stt, tts, models)
    seed_if_empty(container)
    if settings.preload_models:
        threading.Thread(target=_warm_up, args=(container, semantic), daemon=True).start()
    return container


def _warm_up(container: Container, semantic) -> None:
    """Загружает модели в фоне, чтобы первый голосовой запрос не ждал скачивания и инициализации."""
    for name, component in (("stt", container.stt), ("tts", container.tts), ("nlu", container.nlu.llm), ("search", semantic)):
        warm = getattr(component, "warm_up", None)
        if warm is None:
            continue
        try:
            started = datetime.now()
            warm()
            log.info("Модель %s готова за %.1f с", name, (datetime.now() - started).total_seconds())
        except Exception:  # noqa: BLE001 — недоступная модель не должна ронять приложение
            log.exception("Не удалось загрузить модель %s; будет использован запасной вариант", name)


def seed_if_empty(container: Container) -> None:
    settings = container.settings
    if not settings.seed_csv or not Path(settings.seed_csv).is_file():
        return
    if container.records.count(settings.seed_user) > 0:
        return
    report = import_csv(Path(settings.seed_csv), container.records, settings.seed_user, container.now())
    log.info("Импортировано из %s: %d записей, ошибок: %d", settings.seed_csv, report.inserted, len(report.errors))
