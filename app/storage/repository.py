"""Доступ к данным: записи и состояние диалога."""

from __future__ import annotations

import json
import sqlite3
from datetime import date, datetime, time
from typing import Any, Iterable

from app.core.models import Recurrence, Record, RecordType, Status
from app.storage.db import Database

_COLUMNS = (
    "user_id", "type", "title", "day", "at", "is_deadline", "duration_min", "status",
    "recurrence", "source", "external_id", "original_text", "created_at", "updated_at",
)


def _to_row(record: Record) -> tuple:
    return (
        record.user_id,
        record.type.value,
        record.title.strip(),
        record.day.isoformat() if record.day else None,
        record.at.strftime("%H:%M") if record.at else None,
        int(record.is_deadline),
        record.duration_min,
        record.status.value,
        record.recurrence.value if record.recurrence else None,
        record.source,
        record.external_id,
        record.original_text,
        record.created_at.isoformat(timespec="seconds") if record.created_at else None,
        record.updated_at.isoformat(timespec="seconds") if record.updated_at else None,
    )


def _from_row(row: sqlite3.Row) -> Record:
    return Record(
        id=row["id"],
        user_id=row["user_id"],
        type=RecordType(row["type"]),
        title=row["title"],
        day=date.fromisoformat(row["day"]) if row["day"] else None,
        at=time.fromisoformat(row["at"]) if row["at"] else None,
        is_deadline=bool(row["is_deadline"]),
        duration_min=row["duration_min"],
        status=Status(row["status"]),
        recurrence=Recurrence(row["recurrence"]) if row["recurrence"] else None,
        source=row["source"],
        external_id=row["external_id"],
        original_text=row["original_text"],
        created_at=datetime.fromisoformat(row["created_at"]),
        updated_at=datetime.fromisoformat(row["updated_at"]),
    )


class RecordRepository:
    def __init__(self, db: Database):
        self.db = db

    def add(self, record: Record, now: datetime) -> Record:
        record.created_at = record.created_at or now
        record.updated_at = now
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self.db.transaction() as conn:
            cur = conn.execute(f"INSERT INTO records ({', '.join(_COLUMNS)}) VALUES ({placeholders})", _to_row(record))
            record.id = cur.lastrowid
        return record

    def add_many_ignoring_duplicates(self, records: Iterable[Record], now: datetime) -> int:
        """Импорт: повторная загрузка того же файла не создаёт дублей (уникальный external_id)."""
        rows = []
        for record in records:
            record.created_at = record.created_at or now
            record.updated_at = now
            rows.append(_to_row(record))
        placeholders = ", ".join("?" for _ in _COLUMNS)
        with self.db.transaction() as conn:
            before = conn.total_changes
            conn.executemany(f"INSERT OR IGNORE INTO records ({', '.join(_COLUMNS)}) VALUES ({placeholders})", rows)
            return conn.total_changes - before

    def update(self, record: Record, now: datetime) -> Record:
        record.updated_at = now
        assignments = ", ".join(f"{c} = ?" for c in _COLUMNS)
        with self.db.transaction() as conn:
            conn.execute(f"UPDATE records SET {assignments} WHERE id = ? AND user_id = ?", (*_to_row(record), record.id, record.user_id))
        return record

    def delete(self, user_id: str, record_id: int) -> bool:
        with self.db.transaction() as conn:
            return conn.execute("DELETE FROM records WHERE id = ? AND user_id = ?", (record_id, user_id)).rowcount > 0

    def get(self, user_id: str, record_id: int) -> Record | None:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT * FROM records WHERE id = ? AND user_id = ?", (record_id, user_id)).fetchone()
        return _from_row(row) if row else None

    def get_many(self, user_id: str, ids: list[int]) -> list[Record]:
        if not ids:
            return []
        marks = ", ".join("?" for _ in ids)
        with self.db.transaction() as conn:
            rows = conn.execute(f"SELECT * FROM records WHERE user_id = ? AND id IN ({marks})", (user_id, *ids)).fetchall()
        by_id = {r["id"]: _from_row(r) for r in rows}
        return [by_id[i] for i in ids if i in by_id]

    def all(self, user_id: str) -> list[Record]:
        with self.db.transaction() as conn:
            rows = conn.execute("SELECT * FROM records WHERE user_id = ? ORDER BY day, at, id", (user_id,)).fetchall()
        return [_from_row(r) for r in rows]

    def in_range(self, user_id: str, start: date, end: date) -> list[Record]:
        """Записи, которые могут попасть в интервал: разовые внутри него и все повторяющиеся, начатые до его конца."""
        with self.db.transaction() as conn:
            rows = conn.execute(
                """
                SELECT * FROM records
                WHERE user_id = ?
                  AND ((recurrence IS NULL AND day BETWEEN ? AND ?)
                       OR (recurrence IS NOT NULL AND day <= ?))
                ORDER BY day, at, id
                """,
                (user_id, start.isoformat(), end.isoformat(), end.isoformat()),
            ).fetchall()
        return [_from_row(r) for r in rows]

    def undated(self, user_id: str) -> list[Record]:
        with self.db.transaction() as conn:
            rows = conn.execute(
                "SELECT * FROM records WHERE user_id = ? AND day IS NULL AND status = 'pending' ORDER BY created_at",
                (user_id,),
            ).fetchall()
        return [_from_row(r) for r in rows]

    def count(self, user_id: str) -> int:
        with self.db.transaction() as conn:
            return conn.execute("SELECT COUNT(*) FROM records WHERE user_id = ?", (user_id,)).fetchone()[0]


class DialogStateRepository:
    """Состояние диалога (ожидаемый ответ, последние упомянутые записи) хранится в
    базе, поэтому переживает перезапуск сервера."""

    def __init__(self, db: Database):
        self.db = db

    def load(self, session_id: str) -> dict[str, Any]:
        with self.db.transaction() as conn:
            row = conn.execute("SELECT state FROM dialog_state WHERE session_id = ?", (session_id,)).fetchone()
        return json.loads(row["state"]) if row else {}

    def save(self, session_id: str, user_id: str, state: dict[str, Any], now: datetime) -> None:
        with self.db.transaction() as conn:
            conn.execute(
                """
                INSERT INTO dialog_state (session_id, user_id, state, updated_at) VALUES (?, ?, ?, ?)
                ON CONFLICT (session_id) DO UPDATE SET state = excluded.state, updated_at = excluded.updated_at
                """,
                (session_id, user_id, json.dumps(state, ensure_ascii=False), now.isoformat(timespec="seconds")),
            )
