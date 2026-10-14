from datetime import date, datetime, time

import numpy as np

from app.core.commands import Direction
from app.core.models import Record, RecordType
from app.search.engine import SearchEngine
from app.search.morph import lemma
from tests.conftest import NOW

engine = SearchEngine()


def rec(title, day=None, type_=RecordType.TASK, rid=None, at=None):
    return Record(type=type_, title=title, day=day, at=at, id=rid)


RECORDS = [
    rec("Позвонить Ивану", date(2026, 10, 20), rid=1),
    rec("Подготовить презентацию для клиента", date(2026, 10, 15), rid=2),
    rec("Встреча с Иваном по бюджету", date(2026, 3, 5), RecordType.MEETING, 3, time(15, 0)),
    rec("Встреча с Иваном по CRM", date(2025, 4, 4), RecordType.MEETING, 4, time(18, 0)),
    rec("Согласовать контракт с партнёром", date(2026, 11, 1), rid=5),
    rec("Купить билеты в театр", date(2026, 10, 18), rid=6),
    rec("Отправить ТЗ дизайнеру", date(2026, 10, 30), rid=7),
]


def ids(hits):
    return [h.record.id for h in hits]


def test_morphology_matches_any_case():
    assert 1 in ids(engine.search(RECORDS, "Иваном", now=NOW))
    assert 2 in ids(engine.search(RECORDS, "про презентации", now=NOW))


def test_synonyms():
    # «договор» ~ «контракт», «созвон» ~ «позвонить»
    assert ids(engine.search(RECORDS, "договор", now=NOW)) == [5]
    assert 1 in ids(engine.search(RECORDS, "созвон с Иваном", now=NOW))


def test_abbreviation_and_typos():
    assert ids(engine.search(RECORDS, "техническое задание", now=NOW)) == [7]
    assert ids(engine.search(RECORDS, "презинтация", now=NOW)) == [2]


def test_llm_keywords_extend_query():
    assert ids(engine.search(RECORDS, "поход в театр", now=NOW, keywords=["билет"]))[0] == 6


def test_type_filter_and_direction():
    hits = engine.search(RECORDS, "Иван", now=NOW, types={RecordType.MEETING}, direction=Direction.PAST)
    assert ids(hits) == [3, 4]  # последние — первыми


def test_no_match_returns_empty():
    assert engine.search(RECORDS, "отпуск на море", now=NOW) == []


def test_last_meeting_with_ivan_on_real_data(container):
    """«Когда у меня была встреча с Иваном?» — самая поздняя прошедшая встреча с Иваном из CSV."""
    records = container.records.all("demo")
    expected = max(
        (r for r in records if r.type == RecordType.MEETING and "иван" in r.title.lower() and r.starts_at < NOW),
        key=lambda r: r.starts_at,
    )
    hits = engine.search(records, "встреча с Иваном", now=NOW, types={RecordType.MEETING}, direction=Direction.PAST)
    assert hits[0].record.id == expected.id


def test_search_is_fast_enough(container):
    records = container.records.all("demo")
    started = datetime.now()
    engine.search(records, "задача про презентацию", now=NOW)
    assert (datetime.now() - started).total_seconds() < 0.5


class FakeSemantic:
    """Детерминированная «модель»: «финансы» близки к смете и бюджету, остальное — фон."""

    model_name = "fake"

    def similarities(self, query, documents):
        related = ("смет", "бюджет") if "финанс" in query else ()
        return np.array([0.9 if any(r in d.lower() for r in related) else 0.65 for d in documents])


def test_semantic_signal_finds_related_records_without_shared_words():
    records = RECORDS + [rec("Проверить смету по сайту", date(2026, 10, 22), rid=8)]
    assert SearchEngine().search(records, "финансы", now=NOW) == []
    hits = SearchEngine(semantic=FakeSemantic()).search(records, "финансы", now=NOW)
    assert sorted(ids(hits)) == [3, 8]  # «Встреча с Иваном по бюджету» и «Проверить смету по сайту»


def test_semantic_signal_does_not_override_exact_matches():
    engine = SearchEngine(semantic=FakeSemantic())
    assert ids(engine.search(RECORDS, "договор", now=NOW)) == [5]


def test_demo_is_lemmatized_correctly():
    # pymorphy3 разбирает «демо» как причастие «дть» — без исключения не работал синоним «демо ≈ презентация».
    assert lemma("демо") == "демо"
    assert 2 in ids(engine.search(RECORDS, "демо для клиента", now=NOW))
