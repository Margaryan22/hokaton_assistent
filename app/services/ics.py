"""Экспорт записей в iCalendar (.ics): файл можно импортировать в Google/Apple/Outlook-календарь
или подписаться на него по ссылке."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone

from app.core.models import DEFAULT_MEETING_MINUTES, Record, RecordType, Recurrence, Status
from app.services.phrases import type_name

_RRULES = {
    Recurrence.DAILY: "FREQ=DAILY",
    Recurrence.WEEKDAYS: "FREQ=WEEKLY;BYDAY=MO,TU,WE,TH,FR",
    Recurrence.WEEKLY: "FREQ=WEEKLY",
    Recurrence.MONTHLY: "FREQ=MONTHLY",
}


def _escape(text: str) -> str:
    return text.replace("\\", "\\\\").replace(";", "\\;").replace(",", "\\,").replace("\n", "\\n")


def _fold(line: str) -> list[str]:
    """Строки iCalendar длиннее 75 байт переносятся с пробелом в начале продолжения."""
    out, current = [], ""
    for char in line:
        if len((current + char).encode("utf-8")) > 74:
            out.append(current)
            current = " " + char
        else:
            current += char
    out.append(current)
    return out


def _event(record: Record, tzid: str, stamp: str) -> list[str]:
    lines = [
        "BEGIN:VEVENT",
        f"UID:record-{record.id}@voice-planner",
        f"DTSTAMP:{stamp}",
        f"SUMMARY:{_escape(('✓ ' if record.status == Status.DONE else '') + record.title)}",
        f"CATEGORIES:{_escape(type_name(record.type).capitalize())}",
    ]
    if record.at:
        start = datetime.combine(record.day, record.at)
        minutes = record.duration_min or (DEFAULT_MEETING_MINUTES if record.type == RecordType.MEETING else 15)
        lines += [
            f"DTSTART;TZID={tzid}:{start:%Y%m%dT%H%M%S}",
            f"DTEND;TZID={tzid}:{start + timedelta(minutes=minutes):%Y%m%dT%H%M%S}",
        ]
    else:
        lines += [f"DTSTART;VALUE=DATE:{record.day:%Y%m%d}", f"DTEND;VALUE=DATE:{record.day + timedelta(days=1):%Y%m%d}"]
    if record.recurrence:
        lines.append(f"RRULE:{_RRULES[record.recurrence]}")
    if record.type == RecordType.REMINDER and record.at:
        lines += ["BEGIN:VALARM", "ACTION:DISPLAY", f"DESCRIPTION:{_escape(record.title)}", "TRIGGER:PT0M", "END:VALARM"]
    lines.append("END:VEVENT")
    return lines


def to_ics(records: list[Record], tzid: str) -> str:
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    lines = [
        "BEGIN:VCALENDAR",
        "VERSION:2.0",
        "PRODID:-//voice-planner//RU",
        "CALSCALE:GREGORIAN",
        "X-WR-CALNAME:Голосовой планировщик",
        f"X-WR-TIMEZONE:{tzid}",
    ]
    for record in records:
        if record.day is not None:
            lines += _event(record, tzid, stamp)
    lines.append("END:VCALENDAR")
    return "\r\n".join(folded for line in lines for folded in _fold(line)) + "\r\n"
