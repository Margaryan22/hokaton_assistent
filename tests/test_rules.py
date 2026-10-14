from datetime import date, time

import pytest

from app.core.commands import Direction, Intent
from app.core.models import RecordType
from app.nlu.rules import ParseContext, RuleBasedParser
from tests.conftest import NOW

parser = RuleBasedParser()


def parse(text, awaiting=None):
    return parser.parse(text, ParseContext(now=NOW, awaiting=awaiting))


def test_reminder_from_spec():
    cmd = parse("Напомни мне завтра в 10 утра позвонить Ивану")
    assert cmd.intent == Intent.CREATE_REMINDER
    assert (cmd.title, cmd.day, cmd.at) == ("Позвонить Ивану", date(2026, 10, 14), time(10, 0))


def test_task_with_weekday():
    cmd = parse("В пятницу нужно отправить отчет")
    assert cmd.intent == Intent.CREATE_TASK
    assert (cmd.title, cmd.day, cmd.at) == ("Отправить отчет", date(2026, 10, 16), None)


def test_task_with_deadline():
    cmd = parse("Поставь задачу купить билеты до конца недели")
    assert cmd.intent == Intent.CREATE_TASK
    assert (cmd.title, cmd.day, cmd.is_deadline) == ("Купить билеты", date(2026, 10, 18), True)


@pytest.mark.parametrize("text, title, day, at", [
    ("Добавь встречу с командой завтра в 15:00", "Встреча с командой", date(2026, 10, 14), time(15, 0)),
    ("В четверг в 11 у меня встреча с дизайнером", "Встреча с дизайнером", date(2026, 10, 15), time(11, 0)),
    ("Запланируй созвон с Петей в пятницу в 16", "Созвон с Петей", date(2026, 10, 16), time(16, 0)),
])
def test_meetings(text, title, day, at):
    cmd = parse(text)
    assert cmd.intent == Intent.CREATE_MEETING
    assert (cmd.title, cmd.day, cmd.at) == (title, day, at)


def test_meeting_without_time_keeps_day():
    cmd = parse("Поставь встречу с Иваном завтра")
    assert cmd.intent == Intent.CREATE_MEETING
    assert (cmd.day, cmd.at) == (date(2026, 10, 14), None)


@pytest.mark.parametrize("text", [
    "Что у меня сегодня?", "Какие планы на сегодня?", "Покажи мои встречи на сегодня", "Есть ли у меня встречи сегодня?",
    "Чем я занят сегодня?",
])
def test_paraphrases_are_one_list_action(text):
    cmd = parse(text)
    assert cmd.intent == Intent.LIST
    assert (cmd.period.start, cmd.period.end) == (NOW.date(), NOW.date())


def test_list_types_and_periods():
    cmd = parse("Какие задачи у меня на завтра?")
    assert cmd.intent == Intent.LIST and cmd.record_types == {RecordType.TASK, RecordType.REMINDER}
    cmd = parse("Что запланировано на эту неделю?")
    assert cmd.intent == Intent.LIST and (cmd.period.start, cmd.period.end) == (date(2026, 10, 12), date(2026, 10, 18))


@pytest.mark.parametrize("text, query, types, direction", [
    ("Найди задачу про презентацию", "про презентацию", {RecordType.TASK, RecordType.REMINDER}, Direction.ANY),
    ("Когда у меня была встреча с Иваном?", "с Иваном", {RecordType.MEETING}, Direction.PAST),
    ("Я когда-то ставил задачу по договору, найди её", "по договору", {RecordType.TASK, RecordType.REMINDER}, Direction.ANY),
])
def test_search(text, query, types, direction):
    cmd = parse(text)
    assert cmd.intent == Intent.SEARCH
    assert (cmd.query, cmd.record_types, cmd.direction) == (query, types, direction)


def test_query_with_topic_and_period_is_search():
    cmd = parse("Есть ли у меня встреча с Иваном завтра?")
    assert cmd.intent == Intent.SEARCH and cmd.period.start == date(2026, 10, 14)


def test_earliest_trigger_wins():
    assert parse("Поставь задачу удалить старые файлы").intent == Intent.CREATE_TASK
    assert parse("Удали задачу про отчёт").intent == Intent.DELETE


def test_update_with_pronoun_and_new_day():
    cmd = parse("А перенеси её на завтра")
    assert cmd.intent == Intent.UPDATE and cmd.refers_to_previous and cmd.day == date(2026, 10, 14)


def test_update_from_day_to_day():
    cmd = parse("Перенеси встречу с Иваном с пятницы на понедельник")
    assert cmd.intent == Intent.UPDATE
    assert cmd.query == "с Иваном" and cmd.period.start == date(2026, 10, 16) and cmd.day == date(2026, 10, 19)


def test_update_shift_and_rename():
    assert parse("Сдвинь созвон на час позже").shift_minutes == 60
    cmd = parse("Переименуй задачу про отчёт в отчёт за квартал")
    assert (cmd.query, cmd.title) == ("про отчёт", "Отчёт за квартал")


def test_complete_and_delete_targets():
    cmd = parse("Отметь задачу про договор выполненной")
    assert cmd.intent == Intent.COMPLETE and cmd.query == "про договор"
    cmd = parse("удали её")
    assert cmd.intent == Intent.DELETE and cmd.refers_to_previous


def test_reminder_for_another_person_keeps_verb():
    assert parse("Напомни Ивану про отчёт в пятницу в 12").title == "Напомнить Ивану про отчёт"


def test_answers_in_pending_dialog():
    assert parse("в 15:00", awaiting="slot").intent == Intent.ANSWER
    assert parse("позвонить маме", awaiting="slot").title == "Позвонить маме"
    assert parse("вторую", awaiting="choose").selection == 2
    assert parse("да").intent == Intent.CONFIRM
    assert parse("нет, не надо").intent == Intent.DENY


def test_help_and_unknown():
    assert parse("Что ты умеешь?").intent == Intent.HELP
    assert parse("бла бла").intent == Intent.UNKNOWN


@pytest.mark.parametrize("text", [
    "есть что-нибудь на сегодня", "слушай, а что там у меня сегодня", "ну и что у меня сегодня по встречам",
    "мои планы на сегодня", "расскажи, что у меня сегодня",
])
def test_fillers_do_not_turn_list_into_search(text):
    cmd = parse(text)
    assert cmd.intent == Intent.LIST and cmd.period.start == NOW.date()


def test_more_update_forms():
    cmd = parse("перенеси встречу на завтра на 11")
    assert (cmd.day, cmd.at) == (date(2026, 10, 14), time(11, 0))
    cmd = parse("поменяй время встречи с дизайнером на 12")
    assert (cmd.query, cmd.at) == ("с дизайнером", time(12, 0))
    assert parse("отложи задачу про отчёт на неделю").shift_minutes == 7 * 24 * 60


def test_complete_by_state_and_adjective_dates():
    cmd = parse("задача про билеты выполнена")
    assert cmd.intent == Intent.COMPLETE and cmd.query == "про билеты"
    cmd = parse("отмени завтрашнюю встречу с Иваном")
    assert cmd.intent == Intent.DELETE and cmd.period.start == date(2026, 10, 14)


def test_past_markers_in_search():
    cmd = parse("когда я последний раз созванивался с Партнёром")
    assert cmd.direction == Direction.PAST and cmd.query == "созванивался с Партнёром"
