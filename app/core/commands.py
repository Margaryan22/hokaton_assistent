"""Результат разбора реплики: намерение пользователя и извлечённые параметры.

Это контракт между слоем NLU (правила или LLM) и бизнес-логикой: диалоговый
менеджер не знает, кто именно разобрал фразу.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, time
from enum import StrEnum

from app.core.models import Period, Recurrence, RecordType


class Intent(StrEnum):
    CREATE_TASK = "create_task"
    CREATE_REMINDER = "create_reminder"
    CREATE_MEETING = "create_meeting"
    LIST = "list_events"
    SEARCH = "search"
    UPDATE = "update"
    DELETE = "delete"
    COMPLETE = "complete"
    CONFIRM = "confirm"
    DENY = "deny"
    # Ответ на уточняющий вопрос: «в 15:00», «первую», «с Иваном».
    ANSWER = "answer"
    HELP = "help"
    UNKNOWN = "unknown"


CREATE_INTENTS = {Intent.CREATE_TASK, Intent.CREATE_REMINDER, Intent.CREATE_MEETING}
TARGETED_INTENTS = {Intent.UPDATE, Intent.DELETE, Intent.COMPLETE}

INTENT_TO_TYPE = {
    Intent.CREATE_TASK: RecordType.TASK,
    Intent.CREATE_REMINDER: RecordType.REMINDER,
    Intent.CREATE_MEETING: RecordType.MEETING,
}


class Direction(StrEnum):
    PAST = "past"
    FUTURE = "future"
    ANY = "any"


@dataclass
class ParsedCommand:
    intent: Intent
    raw_text: str = ""
    # Создание: содержание записи. Изменение: новое название (переименование).
    title: str | None = None
    # Создание: когда. Изменение: новые дата/время.
    day: date | None = None
    at: time | None = None
    is_deadline: bool = False
    duration_min: int | None = None
    recurrence: Recurrence | None = None
    # Изменение: относительный сдвиг («на час позже»).
    shift_minutes: int | None = None
    # Просмотр и поиск: интервал дат.
    period: Period | None = None
    record_types: set[RecordType] = field(default_factory=set)
    # Поиск: текст запроса. Изменение/удаление: описание целевой записи.
    query: str | None = None
    # Семантически близкие слова (их подсказывает LLM) для расширения поиска.
    keywords: list[str] = field(default_factory=list)
    direction: Direction = Direction.ANY
    # «её», «эту встречу» — ссылка на запись из предыдущего ответа.
    refers_to_previous: bool = False
    # «вторую», «номер 3» — выбор из предложенного списка (с единицы).
    selection: int | None = None
    source: str = "rules"
    # Создание угадано без явного триггера («завтра Петров»): ассистент может переспросить.
    implicit: bool = False

    @property
    def has_when(self) -> bool:
        return self.day is not None or self.at is not None


def command_to_dict(cmd: ParsedCommand) -> dict:
    """Сериализация для хранения незавершённой команды в состоянии диалога."""
    return {
        "intent": cmd.intent.value,
        "raw_text": cmd.raw_text,
        "title": cmd.title,
        "day": cmd.day.isoformat() if cmd.day else None,
        "at": cmd.at.strftime("%H:%M") if cmd.at else None,
        "is_deadline": cmd.is_deadline,
        "duration_min": cmd.duration_min,
        "recurrence": cmd.recurrence.value if cmd.recurrence else None,
        "shift_minutes": cmd.shift_minutes,
        "period": [cmd.period.start.isoformat(), cmd.period.end.isoformat(), cmd.period.label] if cmd.period else None,
        "record_types": sorted(t.value for t in cmd.record_types),
        "query": cmd.query,
        "keywords": cmd.keywords,
        "direction": cmd.direction.value,
        "refers_to_previous": cmd.refers_to_previous,
        "selection": cmd.selection,
        "source": cmd.source,
        "implicit": cmd.implicit,
    }


def command_from_dict(data: dict) -> ParsedCommand:
    period = data.get("period")
    return ParsedCommand(
        intent=Intent(data["intent"]),
        raw_text=data.get("raw_text", ""),
        title=data.get("title"),
        day=date.fromisoformat(data["day"]) if data.get("day") else None,
        at=time.fromisoformat(data["at"]) if data.get("at") else None,
        is_deadline=bool(data.get("is_deadline")),
        duration_min=data.get("duration_min"),
        recurrence=Recurrence(data["recurrence"]) if data.get("recurrence") else None,
        shift_minutes=data.get("shift_minutes"),
        period=Period(date.fromisoformat(period[0]), date.fromisoformat(period[1]), period[2]) if period else None,
        record_types={RecordType(t) for t in data.get("record_types", [])},
        query=data.get("query"),
        keywords=list(data.get("keywords", [])),
        direction=Direction(data.get("direction", "any")),
        refers_to_previous=bool(data.get("refers_to_previous")),
        selection=data.get("selection"),
        source=data.get("source", "rules"),
        implicit=bool(data.get("implicit")),
    )
