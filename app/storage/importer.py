"""Импорт истории событий из CSV (`id,date,time,text`).

Запуск: `python -m app.storage.importer data/voice_assistant_events.csv --user demo`
Импорт идемпотентен: повторный запуск не создаёт дублей.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass, field
from datetime import date, datetime, time
from pathlib import Path

from app.core.models import DEFAULT_MEETING_MINUTES, Record, RecordType
from app.storage.cleaning import fix_grammar, infer_type
from app.storage.repository import RecordRepository

REQUIRED_COLUMNS = {"id", "date", "time", "text"}


@dataclass
class ImportReport:
    total: int = 0
    inserted: int = 0
    duplicates: int = 0
    errors: list[str] = field(default_factory=list)


def parse_rows(path: Path, user_id: str) -> tuple[list[Record], list[str]]:
    records: list[Record] = []
    errors: list[str] = []
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"В файле нет колонок: {', '.join(sorted(missing))}")
        for line_no, row in enumerate(reader, start=2):
            try:
                records.append(_row_to_record(row, user_id))
            except ValueError as exc:
                errors.append(f"строка {line_no}: {exc}")
    return records, errors


def _row_to_record(row: dict[str, str], user_id: str) -> Record:
    external_id = (row.get("id") or "").strip()
    text = " ".join((row.get("text") or "").split())
    if not external_id:
        raise ValueError("пустой id")
    if not text:
        raise ValueError("пустой текст")
    try:
        day = date.fromisoformat((row.get("date") or "").strip())
    except ValueError:
        raise ValueError(f"некорректная дата {row.get('date')!r}") from None
    raw_time = (row.get("time") or "").strip()
    try:
        at = time.fromisoformat(raw_time) if raw_time else None
    except ValueError:
        raise ValueError(f"некорректное время {raw_time!r}") from None
    record_type = infer_type(text)
    title = fix_grammar(text)
    return Record(
        user_id=user_id,
        type=record_type,
        title=title,
        day=day,
        at=at,
        duration_min=DEFAULT_MEETING_MINUTES if record_type == RecordType.MEETING else None,
        source="import",
        external_id=external_id,
        original_text=text if title != text else None,
    )


def import_csv(path: Path, repo: RecordRepository, user_id: str, now: datetime) -> ImportReport:
    records, errors = parse_rows(path, user_id)
    inserted = repo.add_many_ignoring_duplicates(records, now)
    return ImportReport(total=len(records) + len(errors), inserted=inserted, duplicates=len(records) - inserted, errors=errors)


def main() -> None:
    from app.config import get_settings
    from app.storage.db import Database

    parser = argparse.ArgumentParser(description="Импорт событий из CSV")
    parser.add_argument("path", type=Path)
    parser.add_argument("--user", default="demo")
    args = parser.parse_args()
    settings = get_settings()
    repo = RecordRepository(Database(settings.db_path))
    report = import_csv(args.path, repo, args.user, datetime.now(settings.zone).replace(tzinfo=None))
    print(f"Строк: {report.total}, добавлено: {report.inserted}, уже было: {report.duplicates}, ошибок: {len(report.errors)}")
    for error in report.errors[:20]:
        print("  ", error)


if __name__ == "__main__":
    main()
