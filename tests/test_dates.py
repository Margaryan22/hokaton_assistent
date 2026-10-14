from datetime import date, time

import pytest

from app.core.models import Recurrence
from app.nlu.dates import extract_when
from tests.conftest import NOW

# NOW — вторник, 13 октября 2026, 12:00


@pytest.mark.parametrize("text, day, at", [
    ("завтра в 10 утра", date(2026, 10, 14), time(10, 0)),
    ("сегодня в 15:00", date(2026, 10, 13), time(15, 0)),
    ("послезавтра", date(2026, 10, 15), None),
    ("в пятницу", date(2026, 10, 16), None),
    ("во вторник", date(2026, 10, 20), None),  # сегодня вторник → следующий
    ("в эту пятницу", date(2026, 10, 16), None),
    ("в следующую пятницу в 7 вечера", date(2026, 10, 23), time(19, 0)),
    ("через два часа", date(2026, 10, 13), time(14, 0)),
    ("через полчаса", date(2026, 10, 13), time(12, 30)),
    ("через 1 час 30 минут", date(2026, 10, 13), time(13, 30)),
    ("через неделю", date(2026, 10, 20), None),
    ("через 3 дня", date(2026, 10, 16), None),
    ("15 октября в 14:30", date(2026, 10, 15), time(14, 30)),
    ("1 марта", date(2027, 3, 1), None),  # дата прошла — следующий год
    ("двадцать первого ноября в двенадцать", date(2026, 11, 21), time(12, 0)),
    ("15.11 в 10.30", date(2026, 11, 15), time(10, 30)),
    ("на 15.10", date(2026, 10, 15), None),
    ("в четверг в 11", date(2026, 10, 15), time(11, 0)),
    ("в 3", None, time(15, 0)),  # час без уточнения — днём
    ("в 9", None, time(9, 0)),
    ("в пятнадцать тридцать", None, time(15, 30)),
    ("в половине третьего", None, time(14, 30)),
    ("в час дня", None, time(13, 0)),
    ("в 2 ночи", None, time(2, 0)),
    ("в 11 вечера", None, time(23, 0)),
    ("завтра утром", date(2026, 10, 14), time(9, 0)),
    ("сегодня вечером в 8", date(2026, 10, 13), time(20, 0)),
    ("в полдень", None, time(12, 0)),
])
def test_day_and_time(text, day, at):
    when = extract_when(text, NOW)
    assert (when.day, when.at) == (day, at)


@pytest.mark.parametrize("text, start, end", [
    ("на этой неделе", date(2026, 10, 12), date(2026, 10, 18)),
    ("на следующей неделе", date(2026, 10, 19), date(2026, 10, 25)),
    ("на прошлой неделе", date(2026, 10, 5), date(2026, 10, 11)),
    ("в этом месяце", date(2026, 10, 1), date(2026, 10, 31)),
    ("в прошлом месяце", date(2026, 9, 1), date(2026, 9, 30)),
    ("в 2025 году", date(2025, 1, 1), date(2025, 12, 31)),
    ("на выходных", date(2026, 10, 17), date(2026, 10, 18)),
    ("на ближайшие 3 дня", date(2026, 10, 13), date(2026, 10, 15)),
])
def test_periods(text, start, end):
    period = extract_when(text, NOW).period
    assert (period.start, period.end) == (start, end)


def test_until_end_of_week_is_deadline_on_sunday():
    when = extract_when("купить билеты до конца недели", NOW)
    assert when.day == date(2026, 10, 18) and when.is_deadline


def test_past_preference_for_search():
    when = extract_when("в понедельник", NOW, prefer_past=True)
    assert when.day == date(2026, 10, 12)


@pytest.mark.parametrize("text, rule, day, at", [
    ("каждый понедельник в 10", Recurrence.WEEKLY, date(2026, 10, 19), time(10, 0)),
    ("по средам", Recurrence.WEEKLY, date(2026, 10, 14), None),
    ("по будням в 9", Recurrence.WEEKDAYS, date(2026, 10, 14), time(9, 0)),  # 9:00 сегодня уже прошло
    ("каждый день в 20:00", Recurrence.DAILY, date(2026, 10, 13), time(20, 0)),
])
def test_recurrence(text, rule, day, at):
    when = extract_when(text, NOW)
    assert (when.recurrence, when.day, when.at) == (rule, day, at)


def test_duration_and_shift_are_not_times():
    assert extract_when("встреча на 2 часа", NOW).duration_min == 120
    assert extract_when("с 10 до 12", NOW).duration_min == 120
    assert extract_when("на час позже", NOW).shift_minutes == 60
    assert extract_when("на 30 минут раньше", NOW).shift_minutes == -30
    assert extract_when("на 2 часа дня", NOW).at == time(14, 0)
