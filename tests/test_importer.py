import pytest

from app.core.models import RecordType
from app.storage.cleaning import fix_grammar, infer_type
from app.storage.db import Database
from app.storage.importer import import_csv
from app.storage.repository import RecordRepository
from tests.conftest import CSV, NOW


@pytest.mark.parametrize("text, expected", [
    ("Встреча с Иваном по CRM", RecordType.MEETING),
    ("Созвон с Партнёром по интеграции", RecordType.MEETING),
    ("Провести планёрку по бюджету", RecordType.MEETING),
    ("Обсудить договору с Ольгой", RecordType.MEETING),
    ("Напомнить Анной про CRM", RecordType.REMINDER),
    ("Подготовиться к встрече с Вектором", RecordType.TASK),
    ("Отправить итоги встречи Марией", RecordType.TASK),
])
def test_infer_type(text, expected):
    assert infer_type(text) == expected


@pytest.mark.parametrize("text, expected", [
    ("Позвонить Иваном", "Позвонить Ивану"),
    ("Подготовить вопросы для Дмитрием", "Подготовить вопросы для Дмитрия"),
    ("Напомнить Ольгой про сайту", "Напомнить Ольге про сайт"),
    ("Напомнить Дмитрием про мобильному приложению", "Напомнить Дмитрию про мобильное приложение"),
    ("Обсудить новому продукту с Еленой", "Обсудить новый продукт с Еленой"),
    ("Обсудить аналитике с Натальей", "Обсудить аналитику с Натальей"),
    ("Созвон с Стартом по CRM", "Созвон со Стартом по CRM"),
    ("Созвон с Партнёром по интеграции", "Созвон с Партнёром по интеграции"),
    ("Согласовать акт с Максимом", "Согласовать акт с Максимом"),
])
def test_fix_grammar(text, expected):
    assert fix_grammar(text) == expected


def test_import_is_idempotent_and_keeps_original_text():
    repo = RecordRepository(Database(":memory:"))
    first = import_csv(CSV, repo, "u", NOW)
    second = import_csv(CSV, repo, "u", NOW)
    assert (first.total, first.inserted, first.errors) == (1000, 1000, [])
    assert (second.inserted, second.duplicates) == (0, 1000)
    fixed = [r for r in repo.all("u") if r.original_text]
    assert fixed and all(r.title != r.original_text for r in fixed)


def test_import_reports_bad_rows(tmp_path):
    path = tmp_path / "bad.csv"
    path.write_text("id,date,time,text\n1,2026-01-01,10:00,Ок\n2,не дата,10:00,Плохо\n3,2026-01-02,,\n4,2026-01-03,,Без времени\n", encoding="utf-8")
    repo = RecordRepository(Database(":memory:"))
    report = import_csv(path, repo, "u", NOW)
    assert report.inserted == 2
    assert len(report.errors) == 2 and "строка 3" in report.errors[0]


def test_import_rejects_wrong_columns(tmp_path):
    path = tmp_path / "wrong.csv"
    path.write_text("a,b\n1,2\n", encoding="utf-8")
    with pytest.raises(ValueError):
        import_csv(path, RecordRepository(Database(":memory:")), "u", NOW)
