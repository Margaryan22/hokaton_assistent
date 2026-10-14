"""Формат запросов и ответов HTTP API."""

from __future__ import annotations

from datetime import date

from pydantic import BaseModel, Field

from app.core.models import Occurrence
from app.services import phrases as ph


class ChatRequest(BaseModel):
    text: str = Field(..., max_length=1000)
    session_id: str = Field(..., min_length=1, max_length=100, pattern=r"^[\w-]+$")
    user_id: str = Field("demo", min_length=1, max_length=50, pattern=r"^[\w.-]+$")
    timezone: str | None = Field(None, max_length=64)


class ItemOut(BaseModel):
    id: int
    type: str
    type_label: str
    title: str
    day: date | None
    time: str | None
    when: str
    status: str
    is_deadline: bool
    recurrence: str | None
    duration_min: int | None

    @classmethod
    def from_occurrence(cls, occ: Occurrence, today: date) -> "ItemOut":
        record = occ.record
        day = occ.day
        if record.recurrence:
            when = ph.when_text(day, record.at, today) + f" ({ph.recurrence_text(record)})"
        elif day is None:
            when = "без срока"
        elif record.is_deadline:
            when = "срок — " + ph.when_text(day, record.at, today)
        else:
            when = ph.when_text(day, record.at, today)
        return cls(
            id=record.id,
            type=record.type.value,
            type_label=ph.type_name(record.type),
            title=record.title,
            day=day,
            time=ph.time_text(record.at) if record.at else None,
            when=when,
            status=record.status.value,
            is_deadline=record.is_deadline,
            recurrence=record.recurrence.value if record.recurrence else None,
            duration_min=record.duration_min,
        )


class ChatResponse(BaseModel):
    text: str
    summary: str
    speech: str
    items: list[ItemOut]
    numbered: bool
    intent: str
    awaiting: str | None
    nlu: str


class AgendaResponse(BaseModel):
    start: date
    end: date
    items: list[ItemOut]


class HealthResponse(BaseModel):
    status: str
    nlu: str
    models: dict[str, str]
    server_stt: bool
    server_tts: bool
    timezone: str
    records: int


class SpeechOut(BaseModel):
    text: str


class TTSRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=2000)
