"""Формулировки ответов: даты, время, склонения — так, чтобы их было удобно и читать, и слушать."""

from __future__ import annotations

from datetime import date, time, timedelta

from app.core.models import Occurrence, Period, Record, RecordType, Recurrence, Status

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
              "сентября", "октября", "ноября", "декабря"]
WEEKDAYS = ["понедельник", "вторник", "среда", "четверг", "пятница", "суббота", "воскресенье"]
WEEKDAYS_ACC = ["понедельник", "вторник", "среду", "четверг", "пятницу", "субботу", "воскресенье"]
WEEKDAYS_DAT_PL = ["понедельникам", "вторникам", "средам", "четвергам", "пятницам", "субботам", "воскресеньям"]

TYPE_NAMES = {
    RecordType.TASK: ("задача", "задачу", "задачи", "задач"),
    RecordType.REMINDER: ("напоминание", "напоминание", "напоминания", "напоминаний"),
    RecordType.MEETING: ("встреча", "встречу", "встречи", "встреч"),
}


def plural(n: int, one: str, few: str, many: str) -> str:
    n_abs = abs(n) % 100
    if 11 <= n_abs <= 14:
        return many
    if n_abs % 10 == 1:
        return one
    if 2 <= n_abs % 10 <= 4:
        return few
    return many


def count_phrase(n: int, one: str, few: str, many: str) -> str:
    return f"{n} {plural(n, one, few, many)}"


def records_word(n: int, types: set[RecordType] | None = None) -> str:
    if types and len(types) == 1:
        (only,) = types
        names = TYPE_NAMES[only]
        return count_phrase(n, names[0], names[2], names[3])
    if types == {RecordType.TASK, RecordType.REMINDER}:
        return count_phrase(n, "задача", "задачи", "задач")
    return count_phrase(n, "запись", "записи", "записей")


def found_word(n: int, types: set[RecordType] | None = None) -> str:
    """Для «Нашёл …»: винительный падеж — «1 задачу», «1 встречу», «3 записи»."""
    if types and len(types) == 1:
        (only,) = types
        names = TYPE_NAMES[only]
        return count_phrase(n, names[1], names[2], names[3])
    if types == {RecordType.TASK, RecordType.REMINDER}:
        return count_phrase(n, "задачу", "задачи", "задач")
    return count_phrase(n, "запись", "записи", "записей")


def date_text(day: date, today: date, case: str = "nom") -> str:
    """«сегодня, 13 октября», «пятница, 16 октября», «3 марта 2025».

    case="acc" — для конструкций с «на»: «на пятницу, 16 октября».
    """
    base = f"{day.day} {MONTHS_GEN[day.month - 1]}"
    if day.year != today.year:
        base += f" {day.year}"
    relative = {0: "сегодня", 1: "завтра", 2: "послезавтра", -1: "вчера"}.get((day - today).days)
    if relative:
        return f"{relative}, {base}"
    if abs((day - today).days) <= 13:
        weekday = (WEEKDAYS_ACC if case == "acc" else WEEKDAYS)[day.weekday()]
        return f"{weekday}, {base}"
    return base


def time_text(at: time) -> str:
    return at.strftime("%H:%M")


def when_text(day: date | None, at: time | None, today: date, case: str = "nom") -> str:
    if day is None:
        return f"в {time_text(at)}" if at else "без срока"
    text = date_text(day, today, case)
    if at:
        text += f", в {time_text(at)}"
    return text


def recurrence_text(record: Record) -> str:
    rule = record.recurrence
    if rule == Recurrence.DAILY:
        return "каждый день"
    if rule == Recurrence.WEEKDAYS:
        return "по будням"
    if rule == Recurrence.WEEKLY and record.day:
        return f"по {WEEKDAYS_DAT_PL[record.day.weekday()]}"
    if rule == Recurrence.MONTHLY and record.day:
        return f"каждый месяц, {record.day.day}-го числа"
    return ""


def record_when(record: Record, today: date) -> str:
    """Когда запись: «завтра, 14 октября, в 15:00», «срок — пятница, 16 октября», «по понедельникам в 10:00»."""
    if record.recurrence:
        part = recurrence_text(record)
        if record.at:
            part += f" в {time_text(record.at)}"
        if record.day and record.day > today:
            part += f", с {record.day.day} {MONTHS_GEN[record.day.month - 1]}"
        return part
    if record.day is None:
        return "без срока"
    if record.is_deadline:
        return f"срок — {when_text(record.day, record.at, today)}"
    return when_text(record.day, record.at, today)


def type_name(record_type: RecordType, case: int = 0) -> str:
    return TYPE_NAMES[record_type][case]


def describe(record: Record, today: date) -> str:
    """«встречу «Встреча с командой» на завтра, 14 октября, в 15:00»."""
    head = f"{type_name(record.type, 1)} «{record.title}»"
    if record.day is None:
        return f"{head} без срока"
    if record.recurrence:
        return f"{head} — {record_when(record, today)}"
    if record.is_deadline:
        return f"{head}, {record_when(record, today)}"
    return f"{head} на {when_text(record.day, record.at, today, case='acc')}"


def period_text(period: Period, today: date) -> str:
    if period.label:
        return period.label
    if period.start == period.end:
        relative = {0: "сегодня", 1: "завтра", 2: "послезавтра", -1: "вчера"}.get((period.start - today).days)
        if relative:
            return f"на {relative}" if relative != "вчера" else "вчера"
        if abs((period.start - today).days) <= 6:
            prep = "во" if period.start.weekday() == 1 else "в"
            return f"{prep} {WEEKDAYS_ACC[period.start.weekday()]}, {period.start.day} {MONTHS_GEN[period.start.month - 1]}"
        return f"на {date_text(period.start, today)}"
    start, end = period.start, period.end
    if start.month == end.month and start.year == end.year:
        return f"с {start.day} по {end.day} {MONTHS_GEN[end.month - 1]}"
    return f"с {start.day} {MONTHS_GEN[start.month - 1]} по {end.day} {MONTHS_GEN[end.month - 1]}"


def occurrence_line(occ: Occurrence, today: date, with_date: bool) -> str:
    record = occ.record
    parts = []
    if with_date:
        parts.append(date_text(occ.day, today))
    if record.at:
        parts.append(time_text(record.at))
    head = ", ".join(parts)
    mark = " ✓" if record.status == Status.DONE else ""
    kind = type_name(record.type)
    return f"{head} — {record.title} ({kind}){mark}" if head else f"{record.title} ({kind}){mark}"


def occurrence_speech(occ: Occurrence, today: date, with_date: bool) -> str:
    record = occ.record
    parts = []
    if with_date:
        parts.append(date_text(occ.day, today).split(",")[0])
    if record.at:
        parts.append(f"в {time_text(record.at)}")
    head = " ".join(parts)
    return f"{head} — {record.title}" if head else record.title


def ordinal_list(lines: list[str]) -> str:
    return "\n".join(f"{i}. {line}" for i, line in enumerate(lines, start=1))


def next_days(today: date, n: int) -> Period:
    return Period(today, today + timedelta(days=n - 1), "на ближайшую неделю" if n == 7 else "")
