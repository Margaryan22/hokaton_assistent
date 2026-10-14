"""SQLite: подключение и схема.

Одно соединение на процесс под блокировкой — для прототипа этого достаточно,
а заодно работает и с базой в памяти для тестов. Версия схемы хранится в
`PRAGMA user_version`, миграции применяются по порядку при старте.
"""

from __future__ import annotations

import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator

MIGRATIONS: list[str] = [
    """
    CREATE TABLE records (
        id            INTEGER PRIMARY KEY AUTOINCREMENT,
        user_id       TEXT    NOT NULL,
        type          TEXT    NOT NULL CHECK (type IN ('task', 'reminder', 'meeting')),
        title         TEXT    NOT NULL CHECK (length(trim(title)) > 0),
        day           TEXT,                       -- YYYY-MM-DD, локальная дата пользователя
        at            TEXT,                       -- HH:MM
        is_deadline   INTEGER NOT NULL DEFAULT 0, -- дата означает «сделать до»
        duration_min  INTEGER CHECK (duration_min IS NULL OR duration_min > 0),
        status        TEXT    NOT NULL DEFAULT 'pending' CHECK (status IN ('pending', 'done')),
        recurrence    TEXT    CHECK (recurrence IS NULL OR recurrence IN ('daily', 'weekdays', 'weekly', 'monthly')),
        source        TEXT    NOT NULL DEFAULT 'user',
        external_id   TEXT,
        original_text TEXT,
        created_at    TEXT    NOT NULL,
        updated_at    TEXT    NOT NULL,
        CHECK (at IS NULL OR day IS NOT NULL)
    );
    CREATE INDEX idx_records_user_day ON records (user_id, day);
    CREATE UNIQUE INDEX idx_records_import ON records (user_id, source, external_id) WHERE external_id IS NOT NULL;

    CREATE TABLE dialog_state (
        session_id TEXT PRIMARY KEY,
        user_id    TEXT NOT NULL,
        state      TEXT NOT NULL,
        updated_at TEXT NOT NULL
    );
    """,
]


class Database:
    def __init__(self, path: str | Path):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA foreign_keys = ON")
        if self.path != ":memory:":
            self._conn.execute("PRAGMA journal_mode = WAL")
        self._lock = threading.RLock()
        self.migrate()

    def migrate(self) -> None:
        with self._lock:
            version = self._conn.execute("PRAGMA user_version").fetchone()[0]
            for index, script in enumerate(MIGRATIONS[version:], start=version + 1):
                self._conn.executescript(f"BEGIN; {script}; PRAGMA user_version = {index}; COMMIT;")

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            try:
                yield self._conn
                self._conn.commit()
            except Exception:
                self._conn.rollback()
                raise

    def close(self) -> None:
        with self._lock:
            self._conn.close()
