from __future__ import annotations

from datetime import datetime
from pathlib import Path

import pytest

from app.config import Settings
from app.container import Container, build_container
from app.services.assistant import Reply

ROOT = Path(__file__).resolve().parent.parent
CSV = ROOT / "data" / "voice_assistant_events.csv"
# Вторник, 13 октября 2026, полдень.
NOW = datetime(2026, 10, 13, 12, 0)


def make_settings(seed: bool = True) -> Settings:
    # Модели выключены: тесты проверяют логику и не скачивают гигабайты с Hugging Face.
    return Settings(db_path=":memory:", timezone="Europe/Moscow", seed_csv=str(CSV) if seed else None, seed_user="demo")


@pytest.fixture
def container() -> Container:
    c = build_container(make_settings(seed=True))
    yield c
    c.db.close()


@pytest.fixture
def empty_container() -> Container:
    c = build_container(make_settings(seed=False))
    yield c
    c.db.close()


class Dialog:
    def __init__(self, container: Container, user: str = "demo", session: str = "s1", now: datetime = NOW):
        self.container, self.user, self.session, self.now = container, user, session, now

    def say(self, text: str) -> Reply:
        return self.container.assistant.handle(self.user, self.session, text, self.now)


@pytest.fixture
def dialog(container: Container) -> Dialog:
    return Dialog(container)


@pytest.fixture
def clean_dialog(empty_container: Container) -> Dialog:
    return Dialog(empty_container, user="tester")
