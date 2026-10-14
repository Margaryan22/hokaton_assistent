"""Общее для всех LLM-разборщиков: промпт, JSON-схема ответа и перевод ответа в `ParsedCommand`.

Разделение труда: LLM понимает язык — намерение, содержание, тему поиска,
ссылки «её/эту», тип записи. Даты и время она не вычисляет, а переписывает
нужный фрагмент фразы простыми словами («в следующую среду в 13:00»), и точную
дату из него считает детерминированный `WhenExtractor`. Маленькие локальные
модели ошибаются в календарной арифметике, а парсер дат покрыт тестами; к тому
же без календаря в промпте ответ короче и быстрее.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from app.core.commands import TARGETED_INTENTS, Direction, Intent, ParsedCommand
from app.core.models import Period, RecordType
from app.nlu.dates import When, WhenExtractor, mark_shifts
from app.nlu.rules import has_topic, types_from_text
from app.nlu.text import TokenText, simplify

SYSTEM_PROMPT = """Ты — модуль понимания речи голосового ассистента-планировщика. Пользователь говорит по-русски, \
текст получен распознаванием речи (возможны ошибки, нет пунктуации). Определи намерение и извлеки параметры. \
Отвечай только JSON по схеме.

intent:
- create_task — задача («в пятницу нужно отправить отчёт», «поставь задачу купить билеты»)
- create_reminder — напоминание («напомни завтра в 10 позвонить Ивану», «не дай забыть…»)
- create_meeting — встреча, созвон, планёрка, совещание, собеседование, обед или ужин с кем-то
- list_events — показать записи за период без темы («что у меня сегодня», «мои дела на неделю», «с кем у меня \
завтра встречи»)
- search — найти записи по теме или человеку («найди задачу про презентацию», «когда была встреча с Иваном»)
- update — перенести, сдвинуть, переименовать запись
- delete — удалить, отменить, убрать запись
- complete — отметить выполненной («готово, отчёт отправил», «сделал задачу про смету»)
- confirm / deny — «да» / «нет» в ответ на вопрос ассистента
- answer — ответ на уточняющий вопрос ассистента (время, дата, содержание, номер варианта)
- help — приветствие или вопрос о возможностях; unknown — всё остальное

Поля:
- title: содержание новой записи без слов-команд и без даты, с заглавной буквы («Позвонить Ивану», «Встреча с \
командой»). Для update — только новое название при переименовании. Иначе null.
- when: дата, время, срок, период или повтор, переписанные простыми словами и цифрами: «завтра в 10:00», «в \
пятницу», «в следующую среду в 13:00», «5 ноября в 11:30», «до конца недели», «до 20 числа», «на этой неделе», \
«в сентябре», «через 2 часа», «вечером», «каждый понедельник в 10:00», «на 30 минут позже», «на неделю позже». \
Для update — новые дата и время. Если не названы — null. Ничего не придумывай.
- target: для update/delete/complete — описание записи, которую меняют, с её старой датой, если названа \
(«встреча с Иваном в пятницу»). null, если запись указана словами «её», «это», «эту».
- query: для search — тема поиска коротко («презентация», «встреча с Иваном»). Иначе null.
- types: о каких записях речь: task, reminder, meeting. «Задачи» — task и reminder. Пустой список — любые.
- direction: past — речь о прошлом («была», «в прошлом месяце»), future — о будущем («следующая», «когда у меня \
встреча»), иначе any.
- refers: true, если запись указана словами «её», «его», «это», «эту».
- selection: номер варианта из списка ассистента («вторую» → 2, «последнюю» → -1), иначе null.

Примеры:
«Напомни мне завтра в 10 утра позвонить Ивану» → {"intent":"create_reminder","title":"Позвонить Ивану",\
"when":"завтра в 10:00","target":null,"query":null,"types":["reminder"],"direction":"any","refers":false,"selection":null}
«закинь на послезавтра созвон с Петровым часика в четыре» → {"intent":"create_meeting","title":"Созвон с \
Петровым","when":"послезавтра в 16:00","target":null,"query":null,"types":["meeting"],"direction":"any",\
"refers":false,"selection":null}
«мои дела на эту неделю» → {"intent":"list_events","title":null,"when":"на этой неделе","target":null,\
"query":null,"types":[],"direction":"any","refers":false,"selection":null}
«когда я последний раз созванивался с Партнёром» → {"intent":"search","title":null,"when":null,"target":null,\
"query":"созвон с Партнёром","types":["meeting"],"direction":"past","refers":false,"selection":null}
«перенеси встречу с Иваном с пятницы на понедельник в 12» → {"intent":"update","title":null,\
"when":"в понедельник в 12:00","target":"встреча с Иваном в пятницу","query":null,"types":["meeting"],\
"direction":"any","refers":false,"selection":null}
«готово, отчёт отправил» → {"intent":"complete","title":null,"when":null,"target":"отчёт","query":null,\
"types":["task","reminder"],"direction":"any","refers":false,"selection":null}"""


def _nullable(schema: dict[str, Any]) -> dict[str, Any]:
    return {"anyOf": [schema, {"type": "null"}]}


OUTPUT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "intent": {"type": "string", "enum": [i.value for i in Intent]},
        "title": _nullable({"type": "string"}),
        "when": _nullable({"type": "string"}),
        "target": _nullable({"type": "string"}),
        "query": _nullable({"type": "string"}),
        "types": {"type": "array", "items": {"type": "string", "enum": [t.value for t in RecordType]}},
        "direction": {"type": "string", "enum": [d.value for d in Direction]},
        "refers": {"type": "boolean"},
        "selection": _nullable({"type": "integer"}),
    },
    "required": ["intent", "title", "when", "target", "query", "types", "direction", "refers", "selection"],
    "additionalProperties": False,
}


@dataclass
class LLMContext:
    now: datetime
    timezone: str
    pending_question: str | None = None
    recent_records: list[dict[str, Any]] = field(default_factory=list)
    history: list[tuple[str, str]] = field(default_factory=list)


class LLMError(RuntimeError):
    pass


def build_user_message(text: str, ctx: LLMContext) -> str:
    """Контекст диалога передаётся только когда он есть: короче промпт — быстрее ответ."""
    parts = []
    if ctx.pending_question:
        parts.append(f"Ассистент только что спросил: «{ctx.pending_question}»")
    if ctx.recent_records:
        listed = "; ".join(f"{r['номер']}) {r['название']} ({r['когда']})" for r in ctx.recent_records)
        parts.append(f"Записи из прошлого ответа ассистента: {listed}")
    parts.append(f"Реплика пользователя: «{text}»")
    return "\n".join(parts)


def _clean(value: Any) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _extract_when(text: str | None, now: datetime, prefer_past: bool, shifts: bool = False) -> tuple[When, str | None]:
    """Вычисляет дату из фрагмента и возвращает остаток фрагмента без даты."""
    if not text:
        return When(), None
    tt = TokenText.from_text(text)
    if shifts:
        tt = mark_shifts(tt)
    when = WhenExtractor(now, prefer_past).extract(tt)
    consumed: set[int] = set()
    for start, end in when.spans:
        consumed |= tt.token_indices(start, end)
    rest = " ".join(t.orig for t in tt.remaining(consumed)).strip(" ,") or None
    return when, rest


def to_command(data: dict[str, Any], raw_text: str, now: datetime, source: str = "llm") -> ParsedCommand:
    """Проверяет ответ модели, вычисляет даты и переводит всё в `ParsedCommand`."""
    try:
        intent = Intent(data.get("intent"))
    except ValueError as exc:
        raise LLMError(f"неизвестное намерение {data.get('intent')!r}") from exc
    direction = Direction(data["direction"]) if data.get("direction") in {d.value for d in Direction} else Direction.ANY
    when, _ = _extract_when(_clean(data.get("when")), now, prefer_past=direction == Direction.PAST,
                            shifts=intent == Intent.UPDATE)
    title = _clean(data.get("title"))
    cmd = ParsedCommand(
        intent=intent,
        raw_text=raw_text,
        title=title[:1].upper() + title[1:] if title else None,
        day=when.day,
        at=when.at,
        is_deadline=when.is_deadline,
        duration_min=when.duration_min,
        recurrence=when.recurrence,
        shift_minutes=when.shift_minutes,
        record_types={RecordType(t) for t in data.get("types") or [] if t in {r.value for r in RecordType}},
        query=_clean(data.get("query")),
        direction=direction,
        refers_to_previous=bool(data.get("refers")),
        selection=data.get("selection") if isinstance(data.get("selection"), int) else None,
        source=source,
    )
    if not cmd.record_types and intent not in (Intent.CREATE_TASK, Intent.CREATE_REMINDER, Intent.CREATE_MEETING):
        # Маленькие модели иногда не указывают тип; слова «встречи», «задачи» в самой фразе надёжнее.
        cmd.record_types = types_from_text(simplify(raw_text))
    if intent == Intent.SEARCH and cmd.query:
        # Дата могла попасть в тему («встречи в сентябре»), а темы может не быть вовсе.
        query_when, query_rest = _extract_when(cmd.query, now, prefer_past=direction == Direction.PAST)
        if when.empty and not query_when.empty:
            when, cmd.query = query_when, query_rest
        if not cmd.query or not has_topic(cmd.query):
            cmd.intent, cmd.query = Intent.LIST, None
    if cmd.intent in (Intent.LIST, Intent.SEARCH):
        cmd.period = when.period or (Period(when.day, when.day) if when.day else None)
    elif intent == Intent.CREATE_TASK and cmd.day is None and when.period is not None:
        cmd.day, cmd.is_deadline = when.period.end, True
    elif intent in TARGETED_INTENTS:
        target_when, target_rest = _extract_when(_clean(data.get("target")), now, prefer_past=False)
        cmd.query = target_rest
        if target_when.day:
            cmd.period = Period(target_when.day, target_when.day)
        if not cmd.query and cmd.period is None:
            cmd.refers_to_previous = True
    return cmd


def parse_json(raw: str) -> dict[str, Any]:
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise LLMError("ответ не является JSON") from exc
    if not isinstance(data, dict):
        raise LLMError("ответ не является объектом")
    return data
