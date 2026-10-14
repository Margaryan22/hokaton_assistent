"""Доменные сущности: записи (задачи, напоминания, встречи) и их атрибуты."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from enum import StrEnum


class RecordType(StrEnum):
    TASK = "task"
    REMINDER = "reminder"
    MEETING = "meeting"


class Status(StrEnum):
    PENDING = "pending"
    DONE = "done"


class Recurrence(StrEnum):
    DAILY = "daily"
    WEEKDAYS = "weekdays"
    WEEKLY = "weekly"
    MONTHLY = "monthly"


DEFAULT_MEETING_MINUTES = 60


@dataclass
class Record:
    """Запись пользователя.

    `day` и `at` хранятся раздельно: задача «в пятницу отправить отчёт» имеет дату
    без времени, а задача «купить молоко» может не иметь ни того, ни другого.
    Время — «настенное» в часовом поясе пользователя.
    """

    type: RecordType
    title: str
    user_id: str = "demo"
    day: date | None = None
    at: time | None = None
    is_deadline: bool = False
    duration_min: int | None = None
    status: Status = Status.PENDING
    recurrence: Recurrence | None = None
    source: str = "user"
    external_id: str | None = None
    original_text: str | None = None
    created_at: datetime | None = None
    updated_at: datetime | None = None
    id: int | None = None

    @property
    def starts_at(self) -> datetime | None:
        if self.day is None:
            return None
        return datetime.combine(self.day, self.at or time(0, 0))

    @property
    def ends_at(self) -> datetime | None:
        start = self.starts_at
        if start is None or self.at is None:
            return None
        minutes = self.duration_min or (DEFAULT_MEETING_MINUTES if self.type == RecordType.MEETING else 0)
        return start + timedelta(minutes=minutes)

    def occurrences(self, start: date, end: date) -> list[date]:
        """Дни, в которые запись приходится на интервал [start, end] (с учётом повторов)."""
        if self.day is None:
            return []
        if self.recurrence is None:
            return [self.day] if start <= self.day <= end else []
        days: list[date] = []
        current = max(start, self.day)
        while current <= end:
            if _matches_recurrence(self.day, current, self.recurrence):
                days.append(current)
            current += timedelta(days=1)
        return days


def _matches_recurrence(anchor: date, day: date, rule: Recurrence) -> bool:
    if day < anchor:
        return False
    if rule == Recurrence.DAILY:
        return True
    if rule == Recurrence.WEEKDAYS:
        return day.weekday() < 5
    if rule == Recurrence.WEEKLY:
        return day.weekday() == anchor.weekday()
    # Ежемесячно: тот же день месяца; если его нет (31-е), — последний день месяца.
    next_day = day + timedelta(days=1)
    is_last_day = next_day.month != day.month
    return day.day == anchor.day or (is_last_day and anchor.day > day.day)


@dataclass(frozen=True)
class Occurrence:
    """Конкретное появление записи в календаре (для повторяющихся — одно из многих)."""

    record: Record
    # None — у записи нет даты (задача «без срока»).
    day: date | None

    @property
    def starts_at(self) -> datetime | None:
        return datetime.combine(self.day, self.record.at or time(0, 0)) if self.day else None

    @property
    def sort_key(self) -> tuple:
        # Записи без времени (дедлайны на день) показываем после записей со временем.
        return (self.day, self.record.at is None, self.record.at or time(0, 0), self.record.id or 0)


@dataclass(frozen=True)
class Period:
    start: date
    end: date
    label: str = ""

    def contains(self, day: date) -> bool:
        return self.start <= day <= self.end


@dataclass
class SearchHit:
    record: Record
    score: float
    matched_terms: list[str] = field(default_factory=list)
