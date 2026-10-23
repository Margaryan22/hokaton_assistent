"""Извлечение дат, времени, периодов и повторов из русской реплики.

Работает по нормализованному тексту (`TokenText`) и возвращает не только
значения, но и интервалы, которые они занимали: по ним разборщик потом
вырезает дату и время из содержания задачи.
"""

from __future__ import annotations

import re
from calendar import monthrange
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from typing import Callable

from app.core.models import Period, Recurrence
from app.nlu.text import TokenText

WEEKDAY_STEMS = ["понедельник", "вторник", "сред", "четверг", "пятниц", "суббот", "воскресень"]
_WEEKDAY_RE = r"(?P<wd>понедельник|вторник|сред[аеуы]|четверг|пятниц[аеуы]|суббот[аеуы]|воскресень[еяю])"
_WEEKDAY_PLURAL_RE = r"(?P<wd>понедельникам|вторникам|средам|четвергам|пятницам|субботам|воскресеньям)"

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
              "сентября", "октября", "ноября", "декабря"]
MONTHS_LOC = ["январе", "феврале", "марте", "апреле", "мае", "июне", "июле", "августе",
              "сентябре", "октябре", "ноябре", "декабре"]
MONTHS_NOM = ["январь", "февраль", "март", "апрель", "май", "июнь", "июль", "август",
              "сентябрь", "октябрь", "ноябрь", "декабрь"]

DAYPART_DEFAULTS = {"утро": time(9, 0), "день": time(13, 0), "вечер": time(19, 0), "ночь": time(23, 0)}

_NUM = r"(?P<n>\d+(?:\.\d+)?)"
_B = r"(?<![\w:.])"  # начало слова/числа
_E = r"(?![\w:])"    # конец слова/числа
_APPROX_WORDS = r"(?:часик(?:а|ов)?|где-то|примерно|около|ориентировочно)"
_APPROX_BEFORE_RE = re.compile(r"(?<![\w-])(?:" + _APPROX_WORDS + r" )+$")
_APPROX_AFTER_RE = re.compile(r"^ часик(?:а|ов)?(?![\w-])")


@dataclass
class When:
    day: date | None = None
    at: time | None = None
    period: Period | None = None
    is_deadline: bool = False
    recurrence: Recurrence | None = None
    duration_min: int | None = None
    shift_minutes: int | None = None
    daypart: str | None = None
    # День недели без уточнения «этот/следующий» — его можно отнести к названной неделе.
    weekday: int | None = None
    spans: list[tuple[int, int]] = field(default_factory=list)

    def snapshot(self) -> tuple:
        return (self.day, self.at, self.period, self.is_deadline, self.recurrence,
                self.duration_min, self.shift_minutes, self.daypart)

    @property
    def empty(self) -> bool:
        return not (self.day or self.at or self.period or self.recurrence or self.daypart or self.shift_minutes)


def week_bounds(day: date) -> tuple[date, date]:
    monday = day - timedelta(days=day.weekday())
    return monday, monday + timedelta(days=6)


def month_bounds(year: int, month: int) -> tuple[date, date]:
    return date(year, month, 1), date(year, month, monthrange(year, month)[1])


def _weekday_index(word: str) -> int:
    for i, stem in enumerate(WEEKDAY_STEMS):
        if word.startswith(stem):
            return i
    raise ValueError(word)


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


class WhenExtractor:
    """Последовательно применяет правила; найденный фрагмент «закрашивается»,
    чтобы следующие правила не разобрали его повторно."""

    def __init__(self, now: datetime, prefer_past: bool = False):
        self.now = now
        self.today = now.date()
        self.prefer_past = prefer_past
        self._rules: list[tuple[re.Pattern[str], Callable[[re.Match[str], When], None]]] = [
            (re.compile(_B + r"(?:каждый|каждую|каждое|по) (?:будним|будний|рабочим|рабочий) (?:дням|день)" + _E), self._weekdays_rule),
            (re.compile(_B + r"по будням" + _E), self._weekdays_rule),
            (re.compile(_B + r"(?:кажд(?:ый|ую|ое) )" + _WEEKDAY_RE + _E), self._weekly_on_rule),
            (re.compile(_B + r"по " + _WEEKDAY_PLURAL_RE + _E), self._weekly_on_rule),
            (re.compile(_B + r"(?:каждый день|ежедневно|каждое утро|каждый вечер|каждую ночь)" + _E), self._daily_rule),
            (re.compile(_B + r"(?:каждую неделю|еженедельно|раз в неделю)" + _E), self._simple_recurrence(Recurrence.WEEKLY)),
            (re.compile(_B + r"(?:каждый месяц|ежемесячно|раз в месяц)" + _E), self._simple_recurrence(Recurrence.MONTHLY)),
            # «через 2 часа», «через полчаса», «через 1 час 30 минут»
            (re.compile(_B + r"через (?:(?P<h>\d+(?:\.\d+)?) )?час(?:а|ов)?(?: (?P<m>\d+) минут[уы]?)?" + _E), self._in_hours_rule),
            (re.compile(_B + r"через (?:(?P<m>\d+) )?минут[уы]?" + _E), self._in_minutes_rule),
            (re.compile(_B + r"через (?:" + _NUM + r" )?(?P<unit>день|дня|дней|сутки|суток|неделю|недели|недель|месяц|месяца|месяцев|год|года|лет)" + _E), self._in_days_rule),
            (re.compile(_B + _NUM + r" (?P<unit>день|дня|дней|неделю|недели|недель|месяц|месяца|месяцев) назад" + _E), self._ago_rule),
            (re.compile(_B + r"(?P<unit>неделю|месяц|год) назад" + _E), self._ago_rule),
            # Интервал встречи: «с 10 до 12», «с 15:00 до 16:30»
            (re.compile(_B + r"с (?P<h1>\d{1,2})(?:[:.](?P<m1>\d{2}))? (?:до|по) (?P<h2>\d{1,2})(?:[:.](?P<m2>\d{2}))?(?: (?P<suffix>утра|дня|вечера))?" + _E), self._time_range_rule),
            (re.compile(_B + r"(?:(?P<prep>на|к|до|с|по|в) )?(?P<y>\d{4})-(?P<mo>\d{2})-(?P<d>\d{2})" + _E), self._iso_date_rule),
            (re.compile(_B + r"(?:(?P<prep>на|к|до|с|по|в) )?(?P<d>\d{1,2})\.(?P<mo>\d{1,2})(?:\.(?P<y>\d{2,4}))?" + _E), self._dotted_date_rule),
            (re.compile(_B + r"(?:(?P<prep>на|к|до|с|по|в) )?(?P<d>\d{1,2}) (?P<mon>" + "|".join(MONTHS_GEN) + r")(?: (?P<y>\d{4})(?: года| г)?)?" + _E), self._month_date_rule),
            (re.compile(_B + r"(?:(?P<prep>на|к|до) )?(?P<d>\d{1,2}) числа" + _E), self._day_of_month_rule),
            (re.compile(_B + r"(?:(?P<prep>на|к|до|с) )?(?P<w>сегодня|сегодняшний день|завтра|послезавтра|вчера|позавчера|сегодняшн\w+|завтрашн\w+|послезавтрашн\w+|вчерашн\w+)" + _E), self._relative_day_rule),
            (re.compile(_B + r"(?:(?P<prep>во|в|на|к|до|с) )?(?:(?P<mod>эт(?:от|у|о)|следующ(?:ий|ую|ее)|ближайш(?:ий|ую|ее)|прошл(?:ый|ую|ое)) )?" + _WEEKDAY_RE + _E), self._weekday_rule),
            (re.compile(_B + r"(?P<kind>до|к) конц[ау] (?P<what>(?:этой |текущей |следующей )?недели|(?:этого |текущего |следующего )?месяца|(?:этого |текущего )?года|дня)" + _E), self._end_of_rule),
            (re.compile(_B + r"(?:на|в течение|в течении|за|для)? ?(?P<mod>эт(?:ой|у|а)|текущ(?:ей|ую|ая)|следующ(?:ей|ую|ая)|прошл(?:ой|ую|ая)|предыдущ(?:ей|ую|ая)) недел(?:е|ю|и|я)" + _E), self._week_rule),
            # «что там на неделе» — эта неделя
            (re.compile(_B + r"(?:на|в течение|в течении) неделе" + _E), self._week_rule),
            (re.compile(_B + r"(?:на|в|за)? ?(?P<mod>эт(?:ом|от)|текущ(?:ем|ий)|следующ(?:ем|ий)|прошл(?:ом|ый)|предыдущ(?:ем|ий)) месяц[еау]?" + _E), self._month_rule),
            (re.compile(_B + r"(?:на|в|за)? ?(?P<mod>эт(?:ом|от)|текущ(?:ем|ий)|следующ(?:ем|ий)|прошл(?:ом|ый)) год[уа]?" + _E), self._year_rule),
            (re.compile(_B + r"(?:в|на|за) (?P<y>\d{4}) (?:году|год)" + _E), self._explicit_year_rule),
            (re.compile(_B + r"(?:на|в|за) (?P<mon>" + "|".join(MONTHS_LOC + MONTHS_NOM) + r")(?: (?P<y>\d{4})(?: года)?)?" + _E), self._month_name_rule),
            (re.compile(_B + r"(?:в|на) (?:эти |ближайшие |следующие |прошлые )?выходн(?:ые|ых)" + _E), self._weekend_rule),
            (re.compile(_B + r"(?:(?:на|в|за) )?(?:ближайш(?:ие|ую|ий)|следующ(?:ие|ий)) (?:(?P<n>\d+) )?(?P<unit>дня|дней|день|неделю|недели)" + _E), self._next_days_rule),
            # Сдвиг: «на час позже», «раньше на 30 минут», «на день вперёд»
            (re.compile(_B + r"на (?:(?P<n>\d+(?:\.\d+)?) )?(?P<unit>минуту|минуты|минут|час|часа|часов|день|дня|дней|неделю|недели|месяц|месяца) (?P<dir>позже|раньше|вперед|назад)" + _E), self._shift_rule),
            (re.compile(_B + r"(?P<dir>позже|раньше) на (?:(?P<n>\d+(?:\.\d+)?) )?(?P<unit>минуту|минуты|минут|час|часа|часов|день|дня|дней|неделю|недели)" + _E), self._shift_rule),
            # Длительность встречи: «на час», «на 2 часа», «на 30 минут» (но «на 2 часа дня» — это время)
            (re.compile(_B + r"на (?:(?P<n>\d+(?:\.\d+)?) )?(?P<unit>час|часа|часов|минут|минуты)(?! (?:дня|утра|вечера|ночи))" + _E), self._duration_rule),
            # Время
            (re.compile(_B + r"(?:(?P<prep>в|к|на|до|около|после|с) )?(?P<h>\d{1,2})[:.](?P<m>\d{2})(?: (?P<suffix>утра|дня|вечера|ночи))?" + _E), self._clock_rule),
            (re.compile(_B + r"(?P<prep>в|к|до|около|на) (?P<h>\d{1,2}) (?P<m>[0-5]\d)" + _E), self._clock_rule),
            (re.compile(_B + r"(?:(?P<prep>в|к|до|около|на) )?(?:половине|пол) (?P<h>\d{1,2})(?: (?P<suffix>утра|дня|вечера|ночи))?" + _E), self._half_past_rule),
            (re.compile(_B + r"(?:(?P<prep>в|к|до|около|на) )?(?P<h>\d{1,2}) час(?:а|ов)?(?: (?P<m>\d{1,2}) минут[уы]?)?(?: (?P<suffix>утра|дня|вечера|ночи))?" + _E), self._clock_rule),
            (re.compile(_B + r"(?P<prep>в|к|до|около) час(?: (?P<suffix>дня|ночи))?" + _E), self._one_oclock_rule),
            (re.compile(_B + r"(?P<prep>в|к|до|около|на) (?P<h>\d{1,2})(?: (?P<suffix>утра|дня|вечера|ночи))?" + _E), self._clock_rule),
            (re.compile(_B + r"(?P<dp>утром|с утра|днем|днём|в обед|вечером|ночью|под вечер)" + _E), self._daypart_rule),
        ]

    # --- публичный интерфейс -------------------------------------------------

    def extract(self, tt: TokenText) -> When:
        when = When()
        work = tt.text
        for pattern, handler in self._rules:
            for match in list(pattern.finditer(work)):
                if not match.group(0).strip():
                    continue
                before = when.snapshot()
                handler(match, when)
                if when.snapshot() != before:
                    start, end = match.span()
                    when.spans.append((start, end))
                    work = work[:start] + " " * (end - start) + work[end:]
        when.spans = [self._widen(tt.text, span) for span in when.spans]
        self._finalize(when)
        return when

    @staticmethod
    def _widen(text: str, span: tuple[int, int]) -> tuple[int, int]:
        """«часика в четыре», «где-то в 3»: разговорные слова при времени — часть времени, а не названия."""
        start, end = span
        before = _APPROX_BEFORE_RE.search(text[:start])
        after = _APPROX_AFTER_RE.match(text[end:])
        return (before.start() if before else start, end + after.end() if after else end)

    # --- правила --------------------------------------------------------------

    def _simple_recurrence(self, rule: Recurrence):
        def handler(match: re.Match[str], when: When) -> None:
            when.recurrence = rule
        return handler

    def _daily_rule(self, match: re.Match[str], when: When) -> None:
        when.recurrence = Recurrence.DAILY
        text = match.group(0)
        if "утро" in text:
            when.daypart = "утро"
        elif "вечер" in text:
            when.daypart = "вечер"
        elif "ночь" in text:
            when.daypart = "ночь"

    def _weekdays_rule(self, match: re.Match[str], when: When) -> None:
        when.recurrence = Recurrence.WEEKDAYS

    def _weekly_on_rule(self, match: re.Match[str], when: When) -> None:
        when.recurrence = Recurrence.WEEKLY
        target = _weekday_index(match.group("wd"))
        delta = (target - self.today.weekday()) % 7
        when.day = self.today + timedelta(days=delta)

    def _in_hours_rule(self, match: re.Match[str], when: When) -> None:
        hours = float(match.group("h") or 1)
        minutes = int(match.group("m") or 0)
        self._set_moment(when, self.now + timedelta(hours=hours, minutes=minutes))

    def _in_minutes_rule(self, match: re.Match[str], when: When) -> None:
        minutes = int(match.group("m") or 1)
        self._set_moment(when, self.now + timedelta(minutes=minutes))

    def _set_moment(self, when: When, moment: datetime) -> None:
        moment = moment.replace(second=0, microsecond=0)
        when.day = moment.date()
        when.at = moment.time()

    def _in_days_rule(self, match: re.Match[str], when: When) -> None:
        n = float(match.group("n") or 1)
        when.day = self._shift(self.today, match.group("unit"), n)

    def _ago_rule(self, match: re.Match[str], when: When) -> None:
        n = float(match.groupdict().get("n") or 1)
        when.day = self._shift(self.today, match.group("unit"), -n)

    @staticmethod
    def _shift(base: date, unit: str, n: float) -> date:
        if unit.startswith(("д", "сут")):
            return base + timedelta(days=round(n))
        if unit.startswith("нед"):
            return base + timedelta(weeks=n)
        months = round(n * 12) if unit.startswith(("год", "лет")) else round(n)
        month_index = base.month - 1 + months
        year, month = base.year + month_index // 12, month_index % 12 + 1
        return date(year, month, min(base.day, monthrange(year, month)[1]))

    def _time_range_rule(self, match: re.Match[str], when: When) -> None:
        start = self._make_time(int(match.group("h1")), int(match.group("m1") or 0), match.group("suffix"), bare=match.group("m1") is None)
        end = self._make_time(int(match.group("h2")), int(match.group("m2") or 0), match.group("suffix"), bare=match.group("m2") is None)
        if start is None or end is None:
            return
        start_dt, end_dt = datetime.combine(self.today, start), datetime.combine(self.today, end)
        if end_dt <= start_dt:
            end_dt += timedelta(hours=12) if end_dt + timedelta(hours=12) > start_dt else timedelta(days=1)
        when.at = start
        when.duration_min = int((end_dt - start_dt).total_seconds() // 60)

    def _apply_day(self, when: When, day: date | None, prep: str | None) -> None:
        if day is None:
            return
        when.day = day
        if prep in ("до", "к"):
            when.is_deadline = True

    def _infer_year(self, month: int, day: int) -> int:
        candidate = _safe_date(self.today.year, month, day)
        if candidate is None:
            return self.today.year
        if self.prefer_past:
            return self.today.year if candidate <= self.today else self.today.year - 1
        return self.today.year if candidate >= self.today else self.today.year + 1

    def _iso_date_rule(self, match: re.Match[str], when: When) -> None:
        self._apply_day(when, _safe_date(int(match.group("y")), int(match.group("mo")), int(match.group("d"))), match.group("prep"))

    def _dotted_date_rule(self, match: re.Match[str], when: When) -> None:
        d, mo = int(match.group("d")), int(match.group("mo"))
        prep = match.group("prep")
        # «в 15.30» — это время, а не дата.
        if match.group("y") is None and prep in ("в", "к", "до", "с") and d <= 23 and (mo > 12 or len(match.group("mo")) == 2):
            return
        if not 1 <= mo <= 12:
            return
        year_raw = match.group("y")
        if year_raw:
            year = int(year_raw) + (2000 if len(year_raw) == 2 else 0)
        else:
            year = self._infer_year(mo, d)
        self._apply_day(when, _safe_date(year, mo, d), prep)

    def _month_date_rule(self, match: re.Match[str], when: When) -> None:
        d, mo = int(match.group("d")), MONTHS_GEN.index(match.group("mon")) + 1
        year = int(match.group("y")) if match.group("y") else self._infer_year(mo, d)
        self._apply_day(when, _safe_date(year, mo, d), match.group("prep"))

    def _day_of_month_rule(self, match: re.Match[str], when: When) -> None:
        d = int(match.group("d"))
        candidate = _safe_date(self.today.year, self.today.month, d)
        if candidate is None:
            return
        if candidate < self.today and not self.prefer_past:
            nxt = self._shift(candidate.replace(day=1), "месяц", 1)
            candidate = _safe_date(nxt.year, nxt.month, d) or candidate
        self._apply_day(when, candidate, match.group("prep"))

    def _relative_day_rule(self, match: re.Match[str], when: When) -> None:
        word = match.group("w")
        prefixes = [("послезавтр", 2), ("позавчер", -2), ("сегодн", 0), ("завтр", 1), ("вчер", -1)]
        offset = next(value for prefix, value in prefixes if word.startswith(prefix))
        self._apply_day(when, self.today + timedelta(days=offset), match.group("prep"))

    def _weekday_rule(self, match: re.Match[str], when: When) -> None:
        target = _weekday_index(match.group("wd"))
        mod = match.group("mod") or ""
        monday, _ = week_bounds(self.today)
        if mod.startswith("эт"):
            day = monday + timedelta(days=target)
        elif mod.startswith("следующ"):
            day = monday + timedelta(days=7 + target)
        elif mod.startswith("прошл"):
            day = monday + timedelta(days=target - 7)
        elif self.prefer_past:
            delta = (self.today.weekday() - target) % 7
            day = self.today - timedelta(days=delta)
        else:
            # «в пятницу» — ближайшая будущая пятница; если сегодня пятница — через неделю
            # (о сегодняшнем дне обычно говорят «сегодня»).
            delta = (target - self.today.weekday()) % 7 or 7
            if mod.startswith("ближайш") and target == self.today.weekday():
                delta = 0
            day = self.today + timedelta(days=delta)
            if not mod:
                when.weekday = target
        self._apply_day(when, day, match.group("prep"))

    def _end_of_rule(self, match: re.Match[str], when: When) -> None:
        what = match.group("what")
        if "недел" in what:
            _, end = week_bounds(self.today)
            if "следующ" in what:
                end += timedelta(days=7)
            label = "до конца недели"
        elif "месяц" in what:
            base = self._shift(self.today.replace(day=1), "месяц", 1 if "следующ" in what else 0)
            _, end = month_bounds(base.year, base.month)
            label = "до конца месяца"
        elif "год" in what:
            end = date(self.today.year, 12, 31)
            label = "до конца года"
        else:
            end = self.today
            label = "до конца дня"
        when.day = end
        when.is_deadline = True
        when.period = Period(self.today, end, label)

    def _week_rule(self, match: re.Match[str], when: When) -> None:
        mod = match.groupdict().get("mod") or ""
        monday, sunday = week_bounds(self.today)
        if mod.startswith("следующ"):
            when.period = Period(monday + timedelta(days=7), sunday + timedelta(days=7), "на следующей неделе")
        elif mod.startswith(("прошл", "предыдущ")):
            when.period = Period(monday - timedelta(days=7), sunday - timedelta(days=7), "на прошлой неделе")
        else:
            when.period = Period(monday, sunday, "на этой неделе")

    def _month_rule(self, match: re.Match[str], when: When) -> None:
        mod = match.group("mod")
        offset = 1 if mod.startswith("следующ") else -1 if mod.startswith(("прошл", "предыдущ")) else 0
        base = self._shift(self.today.replace(day=1), "месяц", offset)
        start, end = month_bounds(base.year, base.month)
        label = {1: "в следующем месяце", -1: "в прошлом месяце", 0: "в этом месяце"}[offset]
        when.period = Period(start, end, label)

    def _year_rule(self, match: re.Match[str], when: When) -> None:
        mod = match.group("mod")
        year = self.today.year + (1 if mod.startswith("следующ") else -1 if mod.startswith("прошл") else 0)
        when.period = Period(date(year, 1, 1), date(year, 12, 31), f"в {year} году")

    def _explicit_year_rule(self, match: re.Match[str], when: When) -> None:
        year = int(match.group("y"))
        if 1990 <= year <= 2100:
            when.period = Period(date(year, 1, 1), date(year, 12, 31), f"в {year} году")

    def _month_name_rule(self, match: re.Match[str], when: When) -> None:
        name = match.group("mon")
        month = (MONTHS_LOC.index(name) if name in MONTHS_LOC else MONTHS_NOM.index(name)) + 1
        if match.group("y"):
            year = int(match.group("y"))
        elif self.prefer_past:
            year = self.today.year if month <= self.today.month else self.today.year - 1
        else:
            year = self.today.year if month >= self.today.month else self.today.year + 1
        start, end = month_bounds(year, month)
        when.period = Period(start, end, f"в {MONTHS_LOC[month - 1]}" + (f" {year}" if year != self.today.year else ""))

    def _weekend_rule(self, match: re.Match[str], when: When) -> None:
        monday, _ = week_bounds(self.today)
        saturday = monday + timedelta(days=5)
        phrase = match.group(0)
        if "прошл" in phrase:
            saturday -= timedelta(days=7)
        elif "следующ" in phrase:
            saturday += timedelta(days=7)
        start = saturday if "прошл" in phrase else max(saturday, self.today)
        when.period = Period(start, saturday + timedelta(days=1), "на выходных")

    def _next_days_rule(self, match: re.Match[str], when: When) -> None:
        unit = match.group("unit")
        n = int(match.group("n") or (1 if unit.startswith("недел") or unit == "день" else 3))
        days = n * 7 if unit.startswith("недел") else n
        when.period = Period(self.today, self.today + timedelta(days=days - 1), f"на ближайшие {days} дн.")

    def _make_time(self, hour: int, minute: int, suffix: str | None, bare: bool) -> time | None:
        if not (0 <= minute <= 59) or hour > 24:
            return None
        if suffix in ("дня", "вечера") and hour < 12:
            hour += 12
        elif suffix == "утра" and hour == 12:
            hour = 0
        elif suffix == "ночи":
            if hour == 12:
                hour = 0
            elif hour >= 6:
                hour = hour + 12 if hour < 12 else hour
        hour %= 24
        return time(hour, minute)

    def _clock_rule(self, match: re.Match[str], when: When) -> None:
        if when.at is not None:
            return
        groups = match.groupdict()
        hour = int(groups["h"])
        minute = int(groups.get("m") or 0)
        bare_hour = groups.get("m") is None and ":" not in match.group(0) and "." not in match.group(0)
        prep = groups.get("prep")
        # «на 5» без уточнений — скорее количество, чем время.
        if prep == "на" and bare_hour and not groups.get("suffix") and "час" not in match.group(0):
            return
        if prep is None and bare_hour and not groups.get("suffix") and "час" not in match.group(0):
            return
        value = self._make_time(hour, minute, groups.get("suffix"), bare_hour)
        if value is None:
            return
        when.at = value
        if bare_hour and not groups.get("suffix"):
            when.daypart = when.daypart or "?bare"
        if prep in ("к", "до"):
            when.is_deadline = True

    def _half_past_rule(self, match: re.Match[str], when: When) -> None:
        if when.at is not None:
            return
        hour = int(match.group("h")) - 1
        value = self._make_time(hour if hour >= 0 else 23, 30, match.group("suffix"), bare=False)
        if value is None:
            return
        when.at = value
        if not match.group("suffix"):
            when.daypart = when.daypart or "?bare"

    def _one_oclock_rule(self, match: re.Match[str], when: When) -> None:
        if when.at is None:
            when.at = time(1, 0) if match.group("suffix") == "ночи" else time(13, 0)

    def _shift_rule(self, match: re.Match[str], when: When) -> None:
        n = float(match.group("n") or 1)
        unit = match.group("unit")
        if unit.startswith("мин"):
            minutes = n
        elif unit.startswith("час"):
            minutes = n * 60
        elif unit.startswith(("д", "сут")):
            minutes = n * 24 * 60
        elif unit.startswith("месяц"):
            minutes = n * 30 * 24 * 60
        else:
            minutes = n * 7 * 24 * 60
        sign = -1 if match.group("dir") in ("раньше", "назад") else 1
        when.shift_minutes = int(sign * minutes)

    def _duration_rule(self, match: re.Match[str], when: When) -> None:
        n = float(match.group("n") or 1)
        minutes = n * 60 if match.group("unit").startswith("час") else n
        when.duration_min = int(minutes)

    def _daypart_rule(self, match: re.Match[str], when: When) -> None:
        dp = match.group("dp")
        if "утр" in dp:
            when.daypart = "утро" if when.daypart in (None, "?bare") else when.daypart
        elif "вечер" in dp:
            when.daypart = "вечер" if when.daypart in (None, "?bare") else when.daypart
        elif "ноч" in dp:
            when.daypart = "ночь" if when.daypart in (None, "?bare") else when.daypart
        else:
            when.daypart = "день" if when.daypart in (None, "?bare") else when.daypart

    # --- согласование ---------------------------------------------------------

    def _finalize(self, when: When) -> None:
        period = when.period
        if when.weekday is not None and period is not None and (period.end - period.start).days == 6 \
                and period.start.weekday() == 0:
            # «на следующей неделе в среду» — среда той недели, а не ближайшая.
            when.day = period.start + timedelta(days=when.weekday)
            when.period = Period(when.day, when.day)
        dp = when.daypart
        if when.at is not None and dp is not None:
            hour = when.at.hour
            if dp in ("день", "вечер") and hour < 12:
                when.at = when.at.replace(hour=hour + 12)
            elif dp == "?bare" and 1 <= hour <= 7:
                # «в 3» без уточнения — 15:00: встречи и звонки в 3 ночи редкость.
                when.at = when.at.replace(hour=hour + 12)
        elif when.at is None and dp in DAYPART_DEFAULTS:
            when.at = DAYPART_DEFAULTS[dp]
        if dp == "?bare":
            when.daypart = None
        if when.recurrence in (Recurrence.DAILY, Recurrence.WEEKDAYS, Recurrence.MONTHLY, Recurrence.WEEKLY) and when.day is None:
            day = self.today
            if when.at is not None and datetime.combine(day, when.at) <= self.now:
                day += timedelta(days=1)
            if when.recurrence == Recurrence.WEEKDAYS:
                while day.weekday() >= 5:
                    day += timedelta(days=1)
            when.day = day
        if when.period is None and when.day is not None and when.is_deadline and when.at is None:
            when.period = Period(self.today, when.day, "")


_SHIFT_UNIT_RE = re.compile(r"^(?:день|дня|дней|сутки|суток|неделю|недели|недель|месяц|месяца|месяцев)$")


def mark_shifts(tt: TokenText) -> TokenText:
    """В просьбе перенести «на неделю», «на 2 дня» — это сдвиг («на неделю позже»), а не дата."""
    from app.nlu.text import Token

    tokens = list(tt.tokens)
    for i, tok in enumerate(tokens):
        if tok.norm != "на":
            continue
        unit_index = i + 2 if i + 1 < len(tokens) and tokens[i + 1].norm.replace(".", "").isdigit() else i + 1
        if unit_index >= len(tokens) or not _SHIFT_UNIT_RE.match(tokens[unit_index].norm):
            continue
        following = tokens[unit_index + 1].norm if unit_index + 1 < len(tokens) else ""
        if following not in ("позже", "раньше", "вперед", "назад"):
            unit = tokens[unit_index]
            tokens[unit_index] = Token(unit.orig, unit.norm + " позже")
    return TokenText(tokens)


def extract_when(text: str | TokenText, now: datetime, prefer_past: bool = False) -> When:
    tt = text if isinstance(text, TokenText) else TokenText.from_text(text)
    return WhenExtractor(now, prefer_past).extract(tt)
