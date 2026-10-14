"""Бизнес-операции над записями: создание, повестка за период, конфликты, поиск, изменение."""

from __future__ import annotations

from datetime import date, datetime, time, timedelta

from app.core.commands import Direction
from app.core.models import (
    DEFAULT_MEETING_MINUTES, Occurrence, Period, Record, RecordType, SearchHit, Status,
)
from app.search.engine import SearchEngine
from app.storage.repository import RecordRepository


class Planner:
    def __init__(self, repo: RecordRepository, engine: SearchEngine):
        self.repo = repo
        self.engine = engine

    # --- запись -----------------------------------------------------------------

    def create(self, record: Record, now: datetime) -> Record:
        if record.type == RecordType.MEETING and record.at and not record.duration_min:
            record.duration_min = DEFAULT_MEETING_MINUTES
        return self.repo.add(record, now)

    def save(self, record: Record, now: datetime) -> Record:
        return self.repo.update(record, now)

    def delete(self, record: Record) -> bool:
        return self.repo.delete(record.user_id, record.id)

    def complete(self, record: Record, now: datetime) -> Record:
        record.status = Status.DONE
        return self.repo.update(record, now)

    def get_many(self, user_id: str, ids: list[int]) -> list[Record]:
        return self.repo.get_many(user_id, ids)

    # --- чтение -----------------------------------------------------------------

    def agenda(self, user_id: str, period: Period, types: set[RecordType] | None = None) -> list[Occurrence]:
        occurrences = [
            Occurrence(record, day)
            for record in self.repo.in_range(user_id, period.start, period.end)
            if not types or record.type in types
            for day in record.occurrences(period.start, period.end)
        ]
        return sorted(occurrences, key=lambda o: o.sort_key)

    def upcoming(self, user_id: str, now: datetime, types: set[RecordType] | None, horizon_days: int = 30,
                 limit: int = 10) -> list[Occurrence]:
        period = Period(now.date(), now.date() + timedelta(days=horizon_days))
        future = [o for o in self.agenda(user_id, period, types)
                  if (o.record.at is None and o.day >= now.date()) or o.starts_at >= now]
        return future[:limit]

    def undated(self, user_id: str) -> list[Record]:
        return self.repo.undated(user_id)

    def due_reminders(self, user_id: str, since: datetime, now: datetime) -> list[Occurrence]:
        period = Period(since.date(), now.date())
        return [o for o in self.agenda(user_id, period, {RecordType.REMINDER})
                if o.record.at and since < o.starts_at <= now and o.record.status == Status.PENDING]

    def conflicts(self, user_id: str, day: date, at: time, duration_min: int | None,
                  exclude_id: int | None = None) -> list[Occurrence]:
        """Встречи, пересекающиеся по времени с интервалом [day at, +duration)."""
        start = datetime.combine(day, at)
        end = start + timedelta(minutes=duration_min or DEFAULT_MEETING_MINUTES)
        result = []
        for occ in self.agenda(user_id, Period(day, day), {RecordType.MEETING}):
            record = occ.record
            if record.id == exclude_id or record.at is None:
                continue
            other_start = occ.starts_at
            other_end = other_start + timedelta(minutes=record.duration_min or DEFAULT_MEETING_MINUTES)
            if start < other_end and other_start < end:
                result.append(occ)
        return result

    def search(self, user_id: str, query: str, *, now: datetime, keywords: list[str] | None = None,
               types: set[RecordType] | None = None, direction: Direction = Direction.ANY,
               period: Period | None = None, limit: int | None = 10) -> list[SearchHit]:
        return self.engine.search(
            self.repo.all(user_id), query, now=now, keywords=keywords, types=types or None,
            direction=direction, period=period, limit=limit,
        )
