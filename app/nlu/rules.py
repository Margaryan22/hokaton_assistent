"""Разбор реплики правилами — работает без LLM и служит запасным путём.

Намерение определяется по «триггерам» (глаголам и вопросительным словам);
если их несколько, побеждает самый ранний: «Поставь задачу удалить файлы» —
это создание, а «Удали задачу про отчёт» — удаление. Дата и время извлекаются
`WhenExtractor`, а то, что осталось от фразы, становится содержанием записи
или поисковым запросом.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import datetime, time

from app.core.commands import Direction, Intent, ParsedCommand
from app.core.models import Period, RecordType
from app.nlu.dates import When, WhenExtractor, mark_shifts, nearest_when
from app.nlu.text import Token, TokenText
from app.search.engine import content_terms
from app.search.morph import analyzer, lemma


@dataclass
class ParseContext:
    now: datetime
    # Какой ответ ждёт ассистент: "slot" (уточнение), "confirm", "choose" или None.
    awaiting: str | None = None


def _rx(*words: str) -> re.Pattern[str]:
    return re.compile(r"(?<![\w-])(?:" + "|".join(words) + r")(?![\w-])")


REMINDER_RE = _rx("напомни", "напомнить", "напомните", "напоминание", "напоминалку")
CREATE_RE = _rx("добавь", "добавить", "добавьте", "поставь", "поставить", "поставьте", "создай", "создать",
                "запиши", "записать", "внеси", "внести", "запланируй", "запланировать", "назначь", "назначить",
                "организуй", "забронируй", "сохрани", "заведи", "нужно", "надо", "необходимо",
                r"не забыть", r"не забудь", "должен", "должна", "хочу", "планирую", "собираюсь",
                "закинь", "кинь", "впиши", "черкни", "зафиксируй", "забей", r"поставь-ка")
HELP_RE = re.compile(
    r"(?:что ты (?:умеешь|можешь)|что умеешь|помощь|справка|помоги|как (?:тобой|с тобой) пользоваться|"
    r"кто ты|^привет|^здравствуй|^добрый (?:день|вечер)|^доброе утро|^хай$|^hello|^hi$|^алло$|^ау$)"
)
THANKS_RE = re.compile(r"^(?:спасибо|благодарю|спс|отлично|супер|класс|круто)(?: большое| огромное)?(?: тебе)?$")
TRIGGERS: list[tuple[str, re.Pattern[str]]] = [
    ("delete", _rx("удали", "удалить", "удалите", "удаляй", "убери", "убрать", "уберите", "сотри", "стереть",
                   "вычеркни", "отмени", "отменить", "отмените")),
    ("complete", _rx("отметь", "отметить", "отмечай", "пометь", "пометить", "закрой", "закрыть", "выполнил",
                     "выполнила", "сделал", "сделала", "завершил", "завершила", "закончил", "закончила",
                     "выполнена", "выполнено", "сделана", "сделано", "завершена", "закрыта", "готова")),
    ("update", _rx("перенеси", "перенести", "перенесите", "передвинь", "подвинь", "сдвинь", "сдвинуть", "измени",
                   "изменить", "поменяй", "поменять", "переименуй", "переименовать", "отложи", "отложить",
                   "переставь", "перепланируй")),
    ("search", _rx("найди", "найти", "найдите", "поищи", "поиск", "ищи", "отыщи", "вспомни", "где", "когда",
                   "ставил", "ставила", "создавал", "создавала", "добавлял", "добавляла", "записывал",
                   "записывала", r"была? ли", r"были ли", r"было ли")),
    ("query", _rx("что", "чем", "какие", "какая", "какой", "каких", "покажи", "показать", "выведи", "расскажи",
                  "список", r"есть ли", r"есть что(?:-нибудь|-то)?", r"что-нибудь", "сколько", "свободен",
                  "свободна", "занят", "занята", "расписание", "планы",
                  # Разговорное: «чё по планам на завтра», «чо там на неделе», «какие дела».
                  r"ч[ео] (?:по|там|у меня|на)", r"какие дела")),
    ("reminder", REMINDER_RE),
    ("create", CREATE_RE),
]
MEETING_NOUN_RE = _rx(r"встреч[аеуи]", "созвон", "созвониться", "совещание", "планерк[аеу]", "планерка",
                      "митинг", "собрание", "переговоры", r"конференци[яюи]", "демо", r"звонок с", "стендап",
                      "интервью", "собеседование")
TASK_NOUN_RE = _rx(r"задач[аеуи]", "задачку", "todo", "туду")
CONFIRM_RE = re.compile(
    r"^(?:да|ага|угу|конечно|давай|давайте|хорошо|ок|окей|ладно|верно|точно|подтверждаю|согласен|согласна|"
    r"все равно|добавляй|сохраняй|удаляй|да,? (?:давай|конечно|добавь|удаляй|сохрани|все равно)|"
    r"(?:все равно|всё равно) (?:добавь|сохрани|ставь|поставь|оставь))(?: пожалуйста)?$"
)
DENY_RE = re.compile(
    r"^(?:нет|неа|не|не надо|не нужно|не стоит|отмена|отмени|отбой|стоп|забудь|не сохраняй|не добавляй|"
    r"не удаляй|оставь|оставь как есть|нет,? спасибо|нет,? не надо)(?: пожалуйста)?$"
)
PAST_RE = _rx("был", "была", "было", "были", "проходил", "проходила", "проходило", "состоялся", "состоялась",
              "звонил", "звонила", "делал", "делала", r"последний раз", r"в последний раз", r"созванивал\w*",
              r"встречал\w*", r"общал\w*", "прошлая", "прошлую", "прошлой")
FUTURE_RE = _rx("будет", "будут", "предстоит", r"ближайш\w*", r"следующ\w*", "скоро", r"предстоящ\w*", "когда")
ORDINALS = {
    "первый": 1, "первую": 1, "первая": 1, "первое": 1, "первого": 1,
    "второй": 2, "вторую": 2, "вторая": 2, "второе": 2,
    "третий": 3, "третью": 3, "третья": 3, "третье": 3,
    "четвертый": 4, "четвертую": 4, "четвертая": 4, "четвертое": 4,
    "пятый": 5, "пятую": 5, "пятая": 5, "пятое": 5,
    "последний": -1, "последнюю": -1, "последняя": -1, "последнее": -1,
}
PRONOUNS = {"ее", "его", "нее", "него", "эту", "этот", "это", "эти", "их", "она", "он", "оно", "ней", "нем"}
LEADING_FILLERS = {
    "а", "и", "ну", "так", "слушай", "пожалуйста", "мне", "меня", "у", "нужно", "надо", "необходимо", "не",
    "забыть", "забудь", "я", "должен", "должна", "хочу", "планирую", "собираюсь", "что", "чтобы", "бы", "еще",
    "новую", "новое", "новый", "новая", "будет", "есть", "мой", "мою", "моя", "мне", "нам", "нас", "плиз",
    "задачу", "задача", "задачку", "напоминание", "напоминалку", "дело", "запись", "событие", "на", "в", "к",
    "до", "с", "по", "о", "об", "про", "что-нибудь", "кстати", "кое-что", "это", "эту", "такую", "такую-то",
}
TRAILING_FILLERS = {"пожалуйста", "плиз", "на", "в", "к", "до", "с", "по", "у", "и", "а", "мне", "же", "ли", "бы"}
# Эти слова в вопросе не означают тему поиска: «Какие планы на сегодня?»
QUERY_STOPWORDS = {
    "план", "планы", "дело", "расписание", "запланировать", "запланированный", "занятый", "занять", "заняться",
    "свободный", "сколько", "ближайший", "следующий", "предстоящий", "будущий", "показать", "сказать",
    "подсказать", "рассказать", "список", "вывести", "напомнить", "планироваться", "событие", "иметься",
    "какой", "что", "чем", "весь", "вообще", "сегодняшний", "завтрашний", "предстоять", "назначить",
    "планировать", "скоро", "сейчас", "день", "неделя", "месяц", "мероприятие", "календарь", "намечаться",
    "намеченный", "ожидаться", "стоять", "делать", "важный", "срочный", "пора", "время", "наличие",
    "найти", "поискать", "сделать", "заплан", "мочь", "хотеть", "мой", "ставить", "создавать", "добавлять",
    "записывать", "вспомнить", "есть", "быть", "когда", "где", "ли", "либо", "нибудь", "поиск",
    "все", "последний", "раз", "ну", "слушать", "глянуть", "окей", "там", "тут", "вот", "что-нибудь",
    "что-то", "кое-что", "происходить", "ожидать", "интересный", "кто",
}
# Существительные, которые сами по себе — понятное дело: «в среду отчёт», «завтра звонок».
TASK_LIKE_NOUNS = {
    "отчет", "звонок", "встреча", "созвон", "планерка", "совещание", "митинг", "собрание", "переговоры",
    "конференция", "демо", "стендап", "интервью", "собеседование", "презентация", "доклад", "письмо", "рассылка",
    "покупка", "оплата", "платеж", "визит", "прием", "тренировка", "экзамен", "дедлайн", "задача", "напоминание",
    "уборка", "ремонт", "обед", "ужин", "завтрак", "поездка", "командировка", "отпуск", "релиз",
    "ревью", "код-ревью", "сдача", "защита", "дежурство", "занятие", "урок", "лекция", "семинар", "вебинар",
}
COMPLETE_WORDS = {
    "как", "выполненной", "выполненную", "выполненным", "выполненное", "выполнена", "выполнено", "выполнен",
    "сделанной", "сделанную", "сделана", "сделано", "сделан", "готовой", "готовую", "готово", "завершенной",
    "завершенную", "завершена", "закрытой", "закрытую", "выполненными", "сделанными",
}


def types_from_text(norm: str) -> set[RecordType]:
    types: set[RecordType] = set()
    if re.search(r"встреч|созвон|совещан|планерк|митинг|собрани|переговор|демо", norm):
        types.add(RecordType.MEETING)
    if re.search(r"(?<![\w])задач|туду|todo", norm):
        types.update({RecordType.TASK, RecordType.REMINDER})
    if re.search(r"напоминани|напоминалк", norm):
        types.add(RecordType.REMINDER)
    return types


def is_topic(token: Token) -> bool:
    """Слово задаёт тему поиска, а не служебное («покажи», «мои», «планы», «встречи»)."""
    terms = content_terms(token.norm)
    return any(t not in QUERY_STOPWORDS and t not in PRONOUNS and not t.isdigit() for t in terms)


def has_topic(text: str) -> bool:
    return any(is_topic(t) for t in TokenText.from_text(text).tokens)


def looks_like_action(title: str) -> bool:
    """Название похоже на дело: начинается с глагола («Забрать посылку», «Позвони маме»)
    или с существительного-дела («Отчёт», «Звонок бухгалтеру»), а не просто «Петров»."""
    for word in TokenText.from_text(title).tokens:
        if word.norm in LEADING_FILLERS or not re.search(r"[а-я]", word.norm):
            continue
        parses = analyzer().parse(word.norm)
        top = parses[0]
        if "INFN" in top.tag or "impr" in top.tag:
            return True
        return any(p.normal_form.replace("ё", "е") in TASK_LIKE_NOUNS for p in parses[:3])
    return False


def _parse_selection(norm: str) -> int | None:
    words = norm.split()
    for word in words:
        if word in ORDINALS:
            return ORDINALS[word]
    match = re.search(r"(?:номер|под номером|вариант|№)\s*(\d{1,2})", norm)
    if match:
        return int(match.group(1))
    if re.fullmatch(r"(?:давай |ну )?(\d{1,2})(?:-?(?:ю|я|й|ую|ая))?", norm.strip()):
        return int(re.search(r"\d{1,2}", norm).group(0))
    return None


def _clean_title(tokens: list[Token]) -> str:
    words = [t.orig for t in tokens]
    while words and words[0].lower().replace("ё", "е").strip(",.") in LEADING_FILLERS:
        words.pop(0)
    while words and words[-1].lower().replace("ё", "е").strip(",.") in TRAILING_FILLERS:
        words.pop()
    title = " ".join(words).strip(" ,.:;-—")
    return title[:1].upper() + title[1:] if title else ""


def _meeting_title(tokens: list[Token]) -> str:
    """«встречу с командой» → «Встреча с командой»: существительное ставим в именительный падеж."""
    title = _clean_title(tokens)
    if not title:
        return ""
    first, *rest = title.split(" ", 1)
    if MEETING_NOUN_RE.fullmatch(first.lower().replace("ё", "е")):
        nominative = analyzer().parse(first.lower())[0].normal_form
        return " ".join([nominative.capitalize(), *rest])
    if first.lower() in ("с", "со", "по"):
        return f"Встреча {title[:1].lower()}{title[1:]}"
    return title


class RuleBasedParser:
    def parse(self, text: str, ctx: ParseContext) -> ParsedCommand:
        tt = TokenText.from_text(text)
        norm = tt.text.strip(" ,.")
        if not norm:
            return ParsedCommand(Intent.UNKNOWN, raw_text=text)

        if CONFIRM_RE.match(norm):
            return ParsedCommand(Intent.CONFIRM, raw_text=text)
        if DENY_RE.match(norm):
            return ParsedCommand(Intent.DENY, raw_text=text)
        if ctx.awaiting == "choose":
            selection = _parse_selection(norm)
            if selection is not None:
                return ParsedCommand(Intent.ANSWER, raw_text=text, selection=selection)
        if HELP_RE.search(norm):
            return ParsedCommand(Intent.HELP, raw_text=text)

        kind, trigger = self._detect(tt)
        if kind in ("query", "search"):
            return self._parse_query(tt, kind, text, ctx)
        if kind == "delete":
            return self._parse_targeted(tt, Intent.DELETE, text, ctx)
        if kind == "complete":
            return self._parse_targeted(tt, Intent.COMPLETE, text, ctx)
        if kind == "update":
            return self._parse_update(tt, text, ctx)
        if kind in ("reminder", "create"):
            return self._parse_create(tt, kind, trigger, text, ctx)
        # «завтра встреча с Иваном?» без «добавь/поставь» — вопрос, а не просьба создать.
        if kind is None and not ctx.awaiting and text.rstrip().endswith("?"):
            return self._parse_query(tt, "query", text, ctx)

        when = WhenExtractor(ctx.now).extract(tt)
        rest = self._rest(tt, when)
        has_content = bool(content_terms(" ".join(t.norm for t in rest)))
        if MEETING_NOUN_RE.search(norm) and (not when.empty or re.search(r"у меня|будет", norm)):
            return self._parse_create(tt, "meeting", None, text, ctx, when)
        if TASK_NOUN_RE.search(norm) and not ctx.awaiting:
            return self._parse_create(tt, "create", None, text, ctx, when)
        if ctx.awaiting:
            return self._answer(tt, when, text)
        if not when.empty and (has_content or when.day or when.at):
            return self._parse_create(tt, "create", None, text, ctx, when)
        return ParsedCommand(Intent.UNKNOWN, raw_text=text)

    # --- определение намерения ------------------------------------------------

    def _detect(self, tt: TokenText) -> tuple[str | None, tuple[int, int] | None]:
        best: tuple[int, int, str, tuple[int, int]] | None = None
        for priority, (kind, pattern) in enumerate(TRIGGERS):
            for match in pattern.finditer(tt.text):
                start = match.start()
                if kind == "query":
                    # Вопросительное слово должно стоять в начале фразы: «узнать, что с договором» — не вопрос.
                    first_token = min(tt.token_indices(*match.span()), default=99)
                    if first_token > 3:
                        continue
                candidate = (start, priority, kind, match.span())
                if best is None or candidate < best:
                    best = candidate
                break
        if best is None:
            return None, None
        return best[2], best[3]

    def _rest(self, tt: TokenText, when: When, extra_spans: list[tuple[int, int]] = ()) -> list[Token]:
        consumed: set[int] = set()
        for start, end in [*when.spans, *extra_spans]:
            consumed |= tt.token_indices(start, end)
        return tt.remaining(consumed)

    def _all_trigger_spans(self, tt: TokenText) -> list[tuple[int, int]]:
        spans = []
        for _, pattern in TRIGGERS:
            spans.extend(m.span() for m in pattern.finditer(tt.text))
        return spans

    # --- создание -------------------------------------------------------------

    def _parse_create(self, tt: TokenText, kind: str, trigger: tuple[int, int] | None, text: str,
                      ctx: ParseContext, when: When | None = None) -> ParsedCommand:
        when = when or WhenExtractor(ctx.now).extract(tt)
        norm = tt.text
        if kind == "reminder":
            intent = Intent.CREATE_REMINDER
        elif kind == "meeting" or MEETING_NOUN_RE.search(norm):
            intent = Intent.CREATE_MEETING
        elif REMINDER_RE.search(norm):
            intent = Intent.CREATE_REMINDER
        else:
            intent = Intent.CREATE_TASK

        spans: list[tuple[int, int]] = []
        if trigger:
            spans.append(trigger)
        if intent == Intent.CREATE_REMINDER:
            spans.extend(m.span() for m in REMINDER_RE.finditer(norm))
        spans.extend(m.span() for m in CREATE_RE.finditer(norm))
        rest = self._rest(tt, when, spans)

        if intent == Intent.CREATE_MEETING:
            title = _meeting_title(rest)
        else:
            title = _clean_title(rest)
            if intent == Intent.CREATE_REMINDER and title:
                title = self._reminder_title(title)

        cmd = ParsedCommand(
            intent, raw_text=text, title=title or None, day=when.day, at=when.at, is_deadline=when.is_deadline,
            duration_min=when.duration_min, recurrence=when.recurrence, implicit=trigger is None,
        )
        # «до конца недели», «на следующей неделе» — дедлайн в последний день периода.
        if cmd.day is None and when.period is not None and intent == Intent.CREATE_TASK:
            cmd.day, cmd.is_deadline = when.period.end, True
        return cmd

    @staticmethod
    def _reminder_title(title: str) -> str:
        """«Напомни Ивану про отчёт» — напоминание напомнить Ивану, а не «Ивану про отчёт»."""
        first = title.split()[0].lower()
        parse = analyzer().parse(first)[0]
        if "datv" in parse.tag and "NOUN" in parse.tag and "anim" in parse.tag:
            return f"Напомнить {title[:1].upper()}{title[1:]}" if first[0].isupper() else f"Напомнить {title}"
        return title

    # --- просмотр и поиск -----------------------------------------------------

    def _parse_query(self, tt: TokenText, kind: str, text: str, ctx: ParseContext) -> ParsedCommand:
        norm = tt.text
        prefer_past = bool(PAST_RE.search(norm))
        when = WhenExtractor(ctx.now, prefer_past=prefer_past).extract(tt)
        rest = self._rest(tt, when, self._all_trigger_spans(tt))
        topic_tokens = [t for t in rest if self._is_topic(t)]
        types = types_from_text(norm)

        if prefer_past:
            direction = Direction.PAST
        elif FUTURE_RE.search(norm):
            direction = Direction.FUTURE
        else:
            direction = Direction.ANY

        period = when.period
        if period is None and when.day is not None:
            period = Period(when.day, when.day)

        if topic_tokens:
            query = self._query_text(rest)
            if RecordType.TASK in types and not re.search(r"напоминани|напоминалк", norm):
                # «Найди задачу про …» ищет задачи; напоминания — только если задач не нашлось (это решает ассистент).
                types.discard(RecordType.REMINDER)
            return ParsedCommand(Intent.SEARCH, raw_text=text, query=query, record_types=types,
                                 direction=direction, period=period)
        return ParsedCommand(Intent.LIST, raw_text=text, record_types=types, period=period, direction=direction)

    @staticmethod
    def _is_topic(token: Token) -> bool:
        return is_topic(token)

    @staticmethod
    def _query_text(rest: list[Token]) -> str:
        words = [t.orig for t in rest]
        # Убираем вводные и тип записи в начале: «задачу про презентацию» → «про презентацию».
        while words:
            low = words[0].lower().replace("ё", "е")
            if low in LEADING_FILLERS - {"про", "по", "о", "об", "с"} or lemma(low) in QUERY_STOPWORDS or low in PRONOUNS \
                    or lemma(low) in ("задача", "встреча", "напоминание", "запись", "событие", "когда-то", "когда-нибудь"):
                words.pop(0)
                continue
            break
        while words and (words[-1].lower().replace("ё", "е") in TRAILING_FILLERS | PRONOUNS):
            words.pop()
        return " ".join(words)

    # --- изменение, удаление, отметка о выполнении ----------------------------

    def _parse_targeted(self, tt: TokenText, intent: Intent, text: str, ctx: ParseContext) -> ParsedCommand:
        when = nearest_when(tt, ctx.now)
        rest = self._rest(tt, when, self._all_trigger_spans(tt))
        rest = [t for t in rest if t.norm not in COMPLETE_WORDS]
        return self._targeted_command(intent, text, rest, tt.text, when)

    def _targeted_command(self, intent: Intent, text: str, rest: list[Token], norm: str, target_when: When | None) -> ParsedCommand:
        topic = [t for t in rest if self._is_topic(t)]
        refers = not topic and any(t.norm in PRONOUNS for t in rest)
        period = None
        if target_when is not None and target_when.day is not None:
            period = Period(target_when.day, target_when.day)
        elif target_when is not None and target_when.period is not None:
            period = target_when.period
        return ParsedCommand(
            intent, raw_text=text, query=self._query_text(rest) if topic else None,
            record_types=types_from_text(norm), refers_to_previous=refers or (not topic and period is None),
            period=period, target_at=target_when.at if target_when is not None and period is not None else None,
            selection=_parse_selection(norm) if not topic else None,
        )

    def _parse_update(self, tt: TokenText, text: str, ctx: ParseContext) -> ParsedCommand:
        tt = self._normalize_update_tokens(tt)
        norm = tt.text
        rename = re.search(r"(?:переименуй|переименовать|измени название|поменяй название)(?P<target>.*?) (?:в|на) (?P<title>.+)$", norm)
        if rename:
            title_tokens = [tt.tokens[i] for i in sorted(tt.token_indices(*rename.span("title")))]
            target_tokens = [tt.tokens[i] for i in sorted(tt.token_indices(*rename.span("target")))]
            cmd = self._targeted_command(Intent.UPDATE, text, target_tokens, norm, None)
            cmd.title = _clean_title(title_tokens) or None
            return cmd

        # «Перенеси встречу с пятницы на понедельник»: всё после последнего «на» — новые дата и время,
        # всё до него — описание записи (дата в нём помогает выбрать нужную).
        new_when, target_when, split_at = None, None, None
        for match in re.finditer(r"(?<![\w])на (?=\S)", norm):
            split_at = match.start()
        if split_at is not None:
            after = TokenText([tt.tokens[i] for i in sorted(tt.token_indices(split_at, len(norm)))])
            candidate = WhenExtractor(ctx.now).extract(after)
            if not candidate.empty:
                new_when = candidate
                before_tokens = [tt.tokens[i] for i in sorted(tt.token_indices(0, split_at))]
                before = TokenText(before_tokens)
                target_when = nearest_when(before, ctx.now)
                rest = self._rest(before, target_when, self._all_trigger_spans(before))
                rest += self._rest(after, new_when, self._all_trigger_spans(after))
        if new_when is None:
            new_when = WhenExtractor(ctx.now).extract(tt)
            rest = self._rest(tt, new_when, self._all_trigger_spans(tt))

        cmd = self._targeted_command(Intent.UPDATE, text, rest, norm, target_when)
        cmd.day, cmd.at, cmd.shift_minutes = new_when.day, new_when.at, new_when.shift_minutes
        cmd.recurrence = new_when.recurrence
        if new_when.duration_min and new_when.at is None and new_when.shift_minutes is None:
            # «перенеси на 3 часа» — это время 15:00, а не длительность.
            hours = new_when.duration_min // 60
            if new_when.duration_min % 60 == 0 and 1 <= hours <= 12:
                cmd.at = time(hours + 12 if hours <= 7 else hours, 0)
            else:
                cmd.duration_min = new_when.duration_min
        elif new_when.duration_min:
            cmd.duration_min = new_when.duration_min
        if cmd.day is None and new_when.period is not None:
            cmd.day = new_when.period.start
        return cmd

    @staticmethod
    def _normalize_update_tokens(tt: TokenText) -> TokenText:
        """«перенеси на 11» — время 11:00; «отложи на неделю» — сдвиг на неделю вперёд."""
        tokens = list(mark_shifts(tt).tokens)
        units = re.compile(r"^(?:минут\w*|час\w*|день|дня|дней|сутки|суток|недел\w*|месяц\w*|числа|"
                           r"января|февраля|марта|апреля|мая|июня|июля|августа|сентября|октября|ноября|декабря)$")
        for i, tok in enumerate(tokens):
            nxt = tokens[i + 1].norm if i + 1 < len(tokens) else ""
            after = tokens[i + 2].norm if i + 2 < len(tokens) else ""
            if tok.norm == "на" and re.fullmatch(r"\d{1,2}", nxt) and not units.match(after.split()[0] if after else ""):
                tokens[i] = Token(tok.orig, "в")
        return TokenText(tokens)

    # --- ответ на уточняющий вопрос -------------------------------------------

    def _answer(self, tt: TokenText, when: When, text: str) -> ParsedCommand:
        rest = self._rest(tt, when, [m.span() for m in CREATE_RE.finditer(tt.text)])
        title = _clean_title(rest)
        return ParsedCommand(
            Intent.ANSWER, raw_text=text, title=title or None, day=when.day, at=when.at,
            is_deadline=when.is_deadline, duration_min=when.duration_min, recurrence=when.recurrence,
            selection=_parse_selection(tt.text), period=when.period,
        )
