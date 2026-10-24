"""Диалоговый менеджер: реплика → команда → действие → ответ.

Помимо выполнения команд он ведёт состояние диалога:
  * `pending` — чего ассистент ждёт от пользователя: недостающие данные ("slot"),
    подтверждение ("confirm") или выбор одной записи из нескольких ("choose");
  * `last_ids` — записи из последнего ответа, чтобы понимать «перенеси её на завтра»;
  * `history` — последние реплики (контекст для LLM).
"""

from __future__ import annotations

import copy
import logging
import re
from dataclasses import dataclass, field, replace
from datetime import date, datetime, timedelta
from typing import Any

from app.core.commands import (
    CREATE_INTENTS, INTENT_TO_TYPE, Direction, Intent, ParsedCommand, command_from_dict, command_to_dict,
)
from app.core.models import Occurrence, Record, RecordType, Status
from app.nlu.rules import THANKS_RE, looks_like_action, types_from_text
from app.nlu.service import NLU
from app.nlu.text import simplify
from app.services import phrases as ph
from app.services.planner import Planner
from app.storage.repository import DialogStateRepository

log = logging.getLogger(__name__)

MAX_LIST_SPOKEN = 4
MAX_CHOICES = 5
MAX_ITEMS = 30
MAX_LINES = 15
HELP_TEXT = (
    "Я веду ваши задачи, напоминания и встречи. Говорите как удобно, например:\n"
    "• «Напомни завтра в 10 утра позвонить Ивану»\n"
    "• «В четверг в 11 у меня встреча с дизайнером»\n"
    "• «Поставь задачу купить билеты до конца недели»\n"
    "• «Что у меня на этой неделе?»\n"
    "• «Найди задачу про презентацию», «Когда у меня была встреча с Иваном?»\n"
    "• «Перенеси её на пятницу», «Удали встречу с Иваном», «Отметь задачу про отчёт выполненной»"
)
HELP_SPEECH = (
    "Я веду задачи, напоминания и встречи. Скажите, например: напомни завтра в десять позвонить Ивану, "
    "или: что у меня на этой неделе."
)


@dataclass
class Reply:
    """Ответ ассистента: `text` — полный (с перечнем записей), `speech` — для озвучки,
    `summary` — для интерфейса, который показывает записи карточками (`items`)."""

    text: str
    speech: str | None = None
    items: list[Occurrence] = field(default_factory=list)
    summary: str | None = None
    numbered: bool = False
    intent: str = ""
    awaiting: str | None = None
    nlu: str = "rules"

    def __post_init__(self) -> None:
        if self.speech is None:
            self.speech = self.text
        if self.summary is None:
            self.summary = self.text


@dataclass
class Turn:
    """Контекст обработки одной реплики."""

    user_id: str
    now: datetime
    state: dict[str, Any]

    @property
    def today(self) -> date:
        return self.now.date()

    @property
    def pending(self) -> dict[str, Any] | None:
        return self.state.get("pending")

    def set_pending(self, kind: str, question: str, **data: Any) -> None:
        self.state["pending"] = {"kind": kind, "question": question, **data}

    def clear_pending(self) -> None:
        self.state.pop("pending", None)

    @property
    def last_ids(self) -> list[int]:
        return list(self.state.get("last_ids", []))

    def remember(self, records: list[Record], focus: Record | None = None) -> None:
        """Запоминает записи из ответа. `focus` — та, о которой шла речь («ближайшая встреча — …»):
        на неё указывает «её»/«эту», даже если на экране был список."""
        ids = list(dict.fromkeys(r.id for r in records if r.id is not None))[:20]
        self.state["last_ids"] = ids
        if focus is None and len(ids) == 1:
            focus = records[0]
        if focus is not None:
            self.state["focus_id"] = focus.id
        else:
            self.state.pop("focus_id", None)

    @property
    def focus_id(self) -> int | None:
        return self.state.get("focus_id")


class Assistant:
    def __init__(self, nlu: NLU, planner: Planner, states: DialogStateRepository):
        self.nlu = nlu
        self.planner = planner
        self.states = states

    # --- вход ------------------------------------------------------------------

    def handle(self, user_id: str, session_id: str, text: str, now: datetime) -> Reply:
        state = self.states.load(session_id)
        turn = Turn(user_id, now, state)
        text = " ".join(text.split())
        if not text:
            return Reply("Я вас не расслышал. Повторите, пожалуйста.", intent="empty", awaiting=self._awaiting(turn))

        pending = turn.pending
        cmd = self.nlu.parse(
            text,
            now=now,
            awaiting=pending["kind"] if pending else None,
            pending_question=pending["question"] if pending else None,
            recent_records=self._recent_for_llm(turn),
            history=[tuple(h) for h in state.get("history", [])],
        )
        reply = self._route(cmd, turn)
        reply.intent = reply.intent or cmd.intent.value
        reply.nlu = cmd.source
        reply.awaiting = self._awaiting(turn)
        state["history"] = (state.get("history", []) + [["user", text], ["assistant", reply.text]])[-10:]
        self.states.save(session_id, user_id, state, now)
        return reply

    @staticmethod
    def _awaiting(turn: Turn) -> str | None:
        return turn.pending["kind"] if turn.pending else None

    def _recent_for_llm(self, turn: Turn) -> list[dict[str, Any]]:
        records = self.planner.get_many(turn.user_id, turn.last_ids[:MAX_CHOICES])
        return [
            {"номер": i, "тип": ph.type_name(r.type), "название": r.title, "когда": ph.record_when(r, turn.today)}
            for i, r in enumerate(records, start=1)
        ]

    # --- маршрутизация --------------------------------------------------------

    def _route(self, cmd: ParsedCommand, turn: Turn) -> Reply:
        pending = turn.pending
        if pending:
            if cmd.intent in (Intent.CONFIRM, Intent.DENY, Intent.ANSWER):
                return self._continue(cmd, pending, turn)
            draft = command_from_dict(pending["command"]) if pending.get("command") else None
            # «Поставь на 15:00» в ответ на «На какое время поставить встречу?» — это ответ, а не новая задача.
            # А «Каждый понедельник в 10 планёрка» со своим содержанием — новая команда, черновик сбрасываем.
            if (pending["kind"] == "slot" and draft is not None and draft.intent in CREATE_INTENTS
                    and cmd.intent in CREATE_INTENTS and self._answers_slot(cmd, draft, pending.get("missing"))):
                turn.clear_pending()
                return self._create(self._merge(draft, cmd, pending.get("missing")), turn)
            turn.clear_pending()

        if cmd.intent in CREATE_INTENTS:
            return self._create(cmd, turn)
        if cmd.intent == Intent.LIST:
            return self._list(cmd, turn)
        if cmd.intent == Intent.SEARCH:
            return self._search(cmd, turn)
        if cmd.intent in (Intent.UPDATE, Intent.DELETE, Intent.COMPLETE):
            return self._targeted(cmd, turn)
        if cmd.intent == Intent.HELP:
            return Reply(HELP_TEXT, HELP_SPEECH)
        if cmd.intent in (Intent.CONFIRM, Intent.DENY, Intent.ANSWER):
            return Reply("Сейчас мне нечего уточнять. Скажите, что сделать: например, «Что у меня завтра?»")
        if THANKS_RE.match(cmd.raw_text.lower().strip(" !.")):
            return Reply("Пожалуйста! Если что-то ещё — я здесь.")
        return Reply(
            "Не совсем понял запрос. Попробуйте сказать иначе, например: «Что у меня завтра?» "
            "или «Поставь задачу купить билеты до конца недели».",
            "Не совсем понял. Попробуйте сказать иначе, например: что у меня завтра?",
        )

    def _continue(self, cmd: ParsedCommand, pending: dict[str, Any], turn: Turn) -> Reply:
        kind = pending["kind"]
        if kind == "slot":
            return self._continue_slot(cmd, pending, turn)
        if kind == "confirm":
            return self._continue_confirm(cmd, pending, turn)
        return self._continue_choose(cmd, pending, turn)

    # --- уточнение недостающих данных -----------------------------------------

    def _ask(self, turn: Turn, kind: str, question: str, speech: str | None = None, **data: Any) -> Reply:
        turn.set_pending(kind, question, **data)
        return Reply(question, speech)

    @staticmethod
    def _answers_slot(cmd: ParsedCommand, draft: ParsedCommand, missing: str | None) -> bool:
        """Новая фраза — ответ на уточнение, только если у неё нет своего содержания или оно то же, что в черновике."""
        if missing == "title" or not cmd.title or not draft.title:
            return True

        def norm(s: str) -> str:
            return " ".join(re.sub(r"[^\w\s-]", " ", s.lower().replace("ё", "е")).split())

        new, old = norm(cmd.title), norm(draft.title)
        return new == old or new in old or old in new

    @staticmethod
    def _merge(draft: ParsedCommand, answer: ParsedCommand, missing: str | None) -> ParsedCommand:
        merged = copy.deepcopy(draft)
        if missing == "title":
            # Пользователь сам ответил, что нужно сделать, — переспрашивать «создать?» незачем.
            merged.implicit = False
        if answer.title and (missing == "title" or not merged.title):
            merged.title = answer.title
        elif missing == "title" and not merged.title and answer.raw_text and not answer.has_when:
            merged.title = answer.raw_text.strip(" .!?")[:1].upper() + answer.raw_text.strip(" .!?")[1:]
        for name in ("day", "at", "duration_min", "recurrence", "shift_minutes"):
            value = getattr(answer, name)
            if value is not None:
                setattr(merged, name, value)
        if answer.day is not None:
            merged.is_deadline = answer.is_deadline
        if answer.day is None and answer.period is not None and missing in ("when", "date"):
            merged.day = answer.period.start
        return merged

    def _continue_slot(self, cmd: ParsedCommand, pending: dict[str, Any], turn: Turn) -> Reply:
        if cmd.intent == Intent.DENY:
            turn.clear_pending()
            return Reply("Хорошо, отменил.")
        if cmd.intent == Intent.CONFIRM:
            return Reply(pending["question"])
        draft = command_from_dict(pending["command"])
        merged = self._merge(draft, cmd, pending.get("missing"))
        turn.clear_pending()
        if draft.intent in CREATE_INTENTS:
            return self._create(merged, turn)
        record = self._record(turn, pending.get("record_id"))
        if record is None:
            return Reply("Эта запись уже удалена.")
        return self._apply_update(record, merged, turn)

    def _continue_confirm(self, cmd: ParsedCommand, pending: dict[str, Any], turn: Turn) -> Reply:
        action = pending["action"]
        draft = command_from_dict(pending["command"]) if pending.get("command") else None
        record = self._record(turn, pending.get("record_id")) if pending.get("record_id") else None
        if pending.get("record_id") and record is None:
            turn.clear_pending()
            return Reply("Эта запись уже удалена.")

        if cmd.intent == Intent.ANSWER and draft is not None and cmd.has_when:
            # Вместо «да/нет» назвали другое время: «давай на 17:00».
            turn.clear_pending()
            merged = self._merge(draft, cmd, "time")
            merged.implicit = False
            if action == "update":
                return self._apply_update(record, merged, turn)
            return self._create(merged, turn, skip_past=action == "create_tomorrow" and cmd.day is not None)
        if cmd.intent not in (Intent.CONFIRM, Intent.DENY):
            return Reply(f"Ответьте, пожалуйста, «да» или «нет». {pending['question']}")

        turn.clear_pending()
        yes = cmd.intent == Intent.CONFIRM
        if action == "create":
            if yes:
                return self._create(draft, turn, skip_conflicts=True, skip_past=True)
            draft.at = None
            question = "Хорошо. На какое время тогда поставить?"
            return self._ask(turn, "slot", question, command=command_to_dict(draft), missing="time")
        if action == "create_implicit":
            if not yes:
                return Reply("Хорошо, не создаю. Что сделать?")
            draft.implicit = False
            return self._create(draft, turn)
        if action == "create_tomorrow":
            if yes:
                draft.day = draft.day + timedelta(days=1)
                return self._create(draft, turn, skip_past=True)
            return self._create(draft, turn, skip_past=True)
        if action == "delete":
            if not yes:
                return Reply("Хорошо, оставляю.")
            self.planner.delete(record)
            turn.remember([])
            return Reply(f"Удалил {ph.type_name(record.type, 1)} «{record.title}».")
        if action == "update":
            if not yes:
                return Reply("Хорошо, оставил как было.")
            return self._apply_update(record, draft, turn, skip_conflicts=True)
        return Reply("Хорошо.")

    def _continue_choose(self, cmd: ParsedCommand, pending: dict[str, Any], turn: Turn) -> Reply:
        if cmd.intent == Intent.DENY:
            turn.clear_pending()
            return Reply("Хорошо, ничего не меняю.")
        candidates = self.planner.get_many(turn.user_id, pending["candidates"])
        if cmd.intent == Intent.CONFIRM:
            # «Да» вместо номера: понятно, о какой речь, только если на ближайшей неделе она одна.
            week = [r for r in candidates if r.occurrences(turn.today, turn.today + timedelta(days=6))]
            if len(week) != 1:
                return Reply("Назовите номер: первую, вторую…")
            chosen = week[0]
        else:
            chosen = self._pick(candidates, cmd, turn)
        if chosen is None:
            return Reply(f"Не понял, какую именно. Назовите номер, например: «вторую».\n{pending['options']}",
                         "Не понял, какую именно. Назовите номер, например: вторую.",
                         items=[Occurrence(r, r.day) for r in candidates], numbered=True,
                         summary="Не понял, какую именно. Назовите номер, например: «вторую».")
        turn.clear_pending()
        return self._perform(command_from_dict(pending["command"]), chosen, turn)

    def _pick(self, candidates: list[Record], cmd: ParsedCommand, turn: Turn) -> Record | None:
        if not candidates:
            return None
        if cmd.selection is not None:
            index = cmd.selection - 1 if cmd.selection > 0 else len(candidates) + cmd.selection
            return candidates[index] if 0 <= index < len(candidates) else None
        if cmd.day is not None:
            same_day = [r for r in candidates if r.day == cmd.day and (cmd.at is None or r.at == cmd.at)]
            if len(same_day) == 1:
                return same_day[0]
        if cmd.raw_text:
            hits = self.planner.engine.search(candidates, cmd.raw_text, now=turn.now)
            if len(hits) == 1 or (len(hits) > 1 and hits[0].score >= 1.5 * hits[1].score):
                return hits[0].record
        return None

    def _record(self, turn: Turn, record_id: int | None) -> Record | None:
        if record_id is None:
            return None
        found = self.planner.get_many(turn.user_id, [record_id])
        return found[0] if found else None

    # --- создание -------------------------------------------------------------

    def _create(self, cmd: ParsedCommand, turn: Turn, skip_conflicts: bool = False, skip_past: bool = False) -> Reply:
        record_type = INTENT_TO_TYPE[cmd.intent]
        draft = copy.deepcopy(cmd)

        if not draft.title:
            question = {
                RecordType.TASK: "Что нужно сделать?",
                RecordType.REMINDER: "О чём напомнить?",
                RecordType.MEETING: "С кем или о чём встреча?",
            }[record_type]
            return self._ask(turn, "slot", question, command=command_to_dict(draft), missing="title")

        if record_type in (RecordType.MEETING, RecordType.REMINDER):
            if draft.day is None and draft.at is None:
                question = "Когда назначить встречу?" if record_type == RecordType.MEETING else "Когда напомнить?"
                return self._ask(turn, "slot", question, command=command_to_dict(draft), missing="when")
            if draft.at is None:
                question = "На какое время поставить встречу?" if record_type == RecordType.MEETING else "Во сколько напомнить?"
                return self._ask(turn, "slot", question, command=command_to_dict(draft), missing="time")

        if draft.at is not None and draft.day is None:
            # Время без даты — ближайшее: сегодня, если ещё не прошло, иначе завтра.
            draft.day = turn.today if datetime.combine(turn.today, draft.at) > turn.now else turn.today + timedelta(days=1)

        if draft.implicit and not looks_like_action(draft.title):
            # «завтра Петров» — может, это и не просьба создать запись: переспрашиваем.
            preview = Record(type=record_type, title=draft.title, user_id=turn.user_id, day=draft.day, at=draft.at,
                             is_deadline=draft.is_deadline, recurrence=draft.recurrence)
            question = f"Создать {ph.describe(preview, turn.today)}?"
            return self._ask(turn, "confirm", question, action="create_implicit", command=command_to_dict(draft))

        if not skip_past and draft.day == turn.today and draft.at and not draft.recurrence:
            if datetime.combine(draft.day, draft.at) < turn.now:
                question = f"Сегодня {ph.time_text(draft.at)} уже прошло. Поставить на завтра в {ph.time_text(draft.at)}?"
                return self._ask(turn, "confirm", question, action="create_tomorrow", command=command_to_dict(draft))

        if record_type == RecordType.MEETING and not skip_conflicts:
            conflicts = self.planner.conflicts(turn.user_id, draft.day, draft.at, draft.duration_min)
            if conflicts:
                other = conflicts[0].record
                question = f"В это время у вас уже запланировано: {self._slot_text(other)}. Всё равно добавить?"
                turn.remember([other])
                return self._ask(turn, "confirm", question, action="create", command=command_to_dict(draft))

        record = self.planner.create(Record(
            type=record_type, title=draft.title, user_id=turn.user_id, day=draft.day, at=draft.at,
            is_deadline=draft.is_deadline, duration_min=draft.duration_min, recurrence=draft.recurrence,
        ), turn.now)
        turn.remember([record])
        return Reply(self._created_text(record, turn), items=[Occurrence(record, record.day)])

    @staticmethod
    def _slot_text(record: Record) -> str:
        end = (datetime.combine(record.day, record.at) + timedelta(minutes=record.duration_min or 60)).time()
        return f"«{record.title}» ({ph.time_text(record.at)}–{ph.time_text(end)})"

    @staticmethod
    def _created_text(record: Record, turn: Turn) -> str:
        if record.type == RecordType.REMINDER:
            return f"Хорошо, напомню {ph.record_when(record, turn.today)}: {record.title[:1].lower()}{record.title[1:]}."
        return f"Готово, добавил {ph.describe(record, turn.today)}."

    # --- просмотр -------------------------------------------------------------

    def _list(self, cmd: ParsedCommand, turn: Turn) -> Reply:
        types = cmd.record_types or None
        if cmd.period is None and cmd.direction == Direction.FUTURE:
            return self._upcoming(cmd, turn, types)
        period = cmd.period or ph.next_days(turn.today, 7)
        occurrences = self.planner.agenda(turn.user_id, period, types)
        when = ph.period_text(period, turn.today)
        past = period.end < turn.today
        turn.remember([o.record for o in occurrences])
        yes_no = self._is_yes_no(cmd)
        if not occurrences:
            if past:
                return Reply(("Нет. " if yes_no else "") + f"{when.capitalize()} записей не было.")
            nothing = {frozenset({RecordType.MEETING}): "встреч нет", frozenset({RecordType.REMINDER}): "напоминаний нет"}
            text = ("Нет. " if yes_no else "") + f"{when.capitalize()} {nothing.get(frozenset(types or ()), 'ничего не запланировано')}."
            if not types or RecordType.TASK in types:
                undated = self.planner.undated(turn.user_id)
                if undated:
                    text += f" Без срока — {ph.records_word(len(undated), {RecordType.TASK})}."
            return Reply(text)

        multi_day = period.start != period.end
        verb = "было" if past else ""
        head = f"{when.capitalize()} у вас {verb + ' ' if verb else ''}{ph.records_word(len(occurrences), types)}"
        if yes_no:
            head = f"Да, {head[:1].lower()}{head[1:]}"
        lines = [ph.occurrence_line(o, turn.today, with_date=multi_day) for o in occurrences[:MAX_LINES]]
        if len(occurrences) > MAX_LINES:
            lines.append(f"…и ещё {len(occurrences) - MAX_LINES}")
        spoken = [ph.occurrence_speech(o, turn.today, with_date=multi_day) for o in occurrences[:MAX_LIST_SPOKEN]]
        rest = len(occurrences) - len(spoken)
        speech = f"{head}: " + "; ".join(spoken) + (f"; и ещё {rest} — на экране." if rest > 0 else ".")
        return Reply(f"{head}:\n" + "\n".join(lines), speech, items=occurrences, summary=f"{head}:")

    @staticmethod
    def _is_yes_no(cmd: ParsedCommand) -> bool:
        return cmd.raw_text.lower().lstrip(" ,а").startswith(("есть ли", "у меня есть", "будет ли", "были ли", "была ли"))

    def _upcoming(self, cmd: ParsedCommand, turn: Turn, types: set[RecordType] | None) -> Reply:
        occurrences = self.planner.upcoming(turn.user_id, turn.now, types, horizon_days=30)
        turn.remember([o.record for o in occurrences])
        if not occurrences:
            return Reply("В ближайший месяц ничего не запланировано.")
        first = occurrences[0]
        turn.remember([o.record for o in occurrences], focus=first.record)
        leads = {RecordType.MEETING: "Ближайшая встреча", RecordType.REMINDER: "Ближайшее напоминание",
                 RecordType.TASK: "Ближайшая задача"}
        lead = leads[first.record.type] + " — " if types else "Ближайшее событие — "
        text = f"{lead}{ph.when_text(first.day, first.record.at, turn.today)}: «{first.record.title}»."
        if len(occurrences) > 1:
            lines = [ph.occurrence_line(o, turn.today, with_date=True) for o in occurrences]
            return Reply(text + "\nДальше:\n" + "\n".join(lines[1:]), text, items=occurrences, summary=text)
        return Reply(text, items=occurrences)

    # --- поиск ----------------------------------------------------------------

    def _search(self, cmd: ParsedCommand, turn: Turn) -> Reply:
        query = cmd.query or cmd.raw_text
        types = cmd.record_types or None

        def run(**overrides: Any) -> list:
            params = {"keywords": cmd.keywords, "types": types, "direction": cmd.direction, "period": cmd.period}
            params.update(overrides)
            return self.planner.search(turn.user_id, query, now=turn.now, limit=None, **params)

        hits, relaxed = run(), ""
        if not hits and types == {RecordType.TASK}:
            # Пользователи часто зовут задачей и напоминание: «найди задачу про отчёт».
            types = {RecordType.TASK, RecordType.REMINDER}
            hits = run()
        if not hits and types:
            types = None
            hits = run(types=None)
            relaxed = "Среди записей этого типа ничего нет, но нашлось похожее. " if hits else ""
        if not hits and (cmd.direction != Direction.ANY or cmd.period is not None):
            types = None
            hits = run(types=None, direction=Direction.ANY, period=None)
            relaxed = "За этот период ничего нет, но нашлось в другое время. " if hits else ""
        yes_no = self._is_yes_no(cmd)
        echo = self._query_echo(query)
        if not hits:
            return Reply(("Нет. " if yes_no else "") + f"Ничего не нашёл по запросу «{echo}».")

        records = [h.record for h in hits]
        items = [Occurrence(r, r.day) for r in records[:MAX_ITEMS]]
        lines = [ph.occurrence_line(o, turn.today, with_date=True) for o in items[:MAX_LINES]]
        if len(items) > MAX_LINES:
            lines.append(f"…и ещё {len(items) - MAX_LINES}")
        top = records[0]
        turn.remember(records, focus=top if len(records) == 1 or cmd.raw_text.lower().lstrip(" ,а").startswith("когда") else None)

        if cmd.raw_text.lower().lstrip(" ,а").startswith("когда"):
            if top.day is None:
                answer = f"У записи «{top.title}» нет даты."
            else:
                lead = {Direction.PAST: "Последний раз", Direction.FUTURE: "Ближайшая"}.get(cmd.direction, "Нашёл")
                answer = f"{lead} — {ph.when_text(top.day, top.at, turn.today)}: «{top.title}»."
            more = f" Всего совпадений: {len(records)}, список на экране." if len(records) > 1 else ""
            return Reply(relaxed + answer + more + ("\n" + "\n".join(lines) if len(records) > 1 else ""),
                         relaxed + answer, items=items, summary=relaxed + answer + more)

        head = f"{relaxed}{'Да, н' if yes_no else 'Н'}ашёл {ph.found_word(len(records), types)} по запросу «{echo}»"
        if len(records) > len(items):
            order = "последних" if cmd.direction == Direction.PAST else "ближайших"
            head += f", показываю {len(items)} {order}"
        spoken = [ph.occurrence_speech(o, turn.today, with_date=True) for o in items[:3]]
        rest = len(items) - len(spoken)
        speech = f"{head}: " + "; ".join(spoken) + (f"; и ещё {rest} — на экране." if rest > 0 else ".")
        return Reply(f"{head}:\n" + "\n".join(lines), speech, items=items, summary=f"{head}:")

    @staticmethod
    def _query_echo(query: str) -> str:
        """«про презентацию» → «презентацию»: предлог в начале запроса в ответе не нужен."""
        words = query.split()
        while len(words) > 1 and words[0].lower().replace("ё", "е") in ("про", "по", "о", "об", "обо", "насчет"):
            words.pop(0)
        return " ".join(words)

    # --- изменение, удаление, выполнение ---------------------------------------

    def _targeted(self, cmd: ParsedCommand, turn: Turn) -> Reply:
        rules = cmd.fallback
        if (rules is not None and not rules.refers_to_previous and not cmd.refers_to_previous
                and not self._find_candidates(cmd, turn) and self._find_candidates(rules, turn)):
            # По разбору модели запись не нашлась, а по разбору правил той же фразы — нашлась.
            log.debug("Запись нашлась только по разбору правил: %s", rules)
            cmd = rules
        resolved = self._resolve(cmd, turn)
        if isinstance(resolved, Reply):
            return resolved
        return self._perform(cmd, resolved, turn)

    def _resolve(self, cmd: ParsedCommand, turn: Turn) -> Record | Reply:
        verbs = {Intent.UPDATE: "изменить", Intent.DELETE: "удалить", Intent.COMPLETE: "отметить"}
        if cmd.refers_to_previous or (not cmd.query and cmd.period is None):
            ids = [turn.focus_id] if turn.focus_id and cmd.selection is None else turn.last_ids
            candidates = self.planner.get_many(turn.user_id, ids)
            if not candidates and ids:
                # Запись, о которой шла речь, тем временем удалили (например, кнопкой в повестке).
                return Reply("Эта запись уже удалена." if len(ids) == 1 else "Этих записей уже нет.")
            named = types_from_text(simplify(cmd.raw_text))
            if named and candidates and not any(r.type in named for r in candidates):
                # «Удали это напоминание», а в фокусе встреча: ищем напоминание среди записей прошлого ответа,
                # а если его там нет — переспрашиваем, а не удаляем встречу.
                candidates = [r for r in self.planner.get_many(turn.user_id, turn.last_ids) if r.type in named]
                if not candidates:
                    which = {frozenset({RecordType.REMINDER}): "какое напоминание",
                             frozenset({RecordType.MEETING}): "какую встречу",
                             frozenset({RecordType.TASK, RecordType.REMINDER}): "какую задачу"}
                    example = {Intent.UPDATE: "перенеси напоминание позвонить Ивану на пятницу",
                               Intent.DELETE: "удали напоминание позвонить Ивану",
                               Intent.COMPLETE: "отметь напоминание позвонить Ивану выполненным"}[cmd.intent]
                    return Reply(f"Уточните, {which.get(frozenset(named), 'какую запись')} {verbs[cmd.intent]}. "
                                 f"Например: «{example}».")
            if not candidates:
                examples = {
                    Intent.UPDATE: "перенеси встречу с Иваном на пятницу",
                    Intent.DELETE: "удали задачу про отчёт",
                    Intent.COMPLETE: "отметь задачу про договор выполненной",
                }
                return Reply(f"Уточните, какую запись {verbs[cmd.intent]}. Например: «{examples[cmd.intent]}».")
        else:
            candidates = self._find_candidates(cmd, turn)
            if not candidates:
                return self._not_found(cmd, turn)

        if cmd.selection is not None:
            picked = self._pick(candidates, ParsedCommand(Intent.ANSWER, selection=cmd.selection), turn)
            if picked is not None:
                return picked
        if len(candidates) == 1:
            return candidates[0]

        options = candidates[:MAX_CHOICES]
        option_lines = ph.ordinal_list([ph.occurrence_line(Occurrence(r, r.day), turn.today, with_date=True) for r in options])
        turn.remember(options)
        question = f"Нашёл несколько подходящих записей:\n{option_lines}\nКакую {verbs[cmd.intent]}?"
        spoken = "; ".join(f"{i}: {ph.occurrence_speech(Occurrence(r, r.day), turn.today, True)}" for i, r in enumerate(options[:3], 1))
        self._ask(turn, "choose", question, command=command_to_dict(cmd), candidates=[r.id for r in options],
                  options=option_lines)
        return Reply(question, f"Нашёл несколько: {spoken}. Какую {verbs[cmd.intent]}?",
                     items=[Occurrence(r, r.day) for r in options],
                     summary=f"Нашёл несколько подходящих записей. Какую {verbs[cmd.intent]}?", numbered=True)

    @staticmethod
    def _not_found(cmd: ParsedCommand, turn: Turn) -> Reply:
        """«Не нашёл напоминание «позвонить Ивану» во вторник, 27 октября» — и пример, как сказать иначе."""
        types = cmd.record_types
        if len(types) == 1:
            what = ph.type_name(next(iter(types)), 1)
        else:
            what = "задачу" if types == {RecordType.TASK, RecordType.REMINDER} else "запись"
        if cmd.query:
            what += f" «{Assistant._query_echo(cmd.query)}»"
        if cmd.period is not None:
            what += f" {ph.period_text(cmd.period, turn.today)}"
            if cmd.target_at is not None:
                what += f", в {ph.time_text(cmd.target_at)}"
        examples = {
            Intent.UPDATE: "перенеси встречу с командой на пятницу",
            Intent.DELETE: "удали напоминание позвонить Ивану",
            Intent.COMPLETE: "отметь задачу про отчёт выполненной",
        }
        return Reply(f"Не нашёл {what}. Скажите, например: «{examples[cmd.intent]}».",
                     f"Не нашёл {what}.")

    def _find_candidates(self, cmd: ParsedCommand, turn: Turn) -> list[Record]:
        types = cmd.record_types or None
        if cmd.period is not None and cmd.target_at is not None:
            # «Удали напоминание 26 октября в 12:00»: запись ровно в это время — та самая.
            exact = list({o.record.id: o.record for o in self.planner.agenda(turn.user_id, cmd.period, types)
                          if o.record.at == cmd.target_at}.values())
            if cmd.query and exact:
                exact = [h.record for h in self.planner.engine.search(exact, cmd.query, now=turn.now)]
            if len(exact) == 1:
                return exact
        if not cmd.query:
            occurrences = self.planner.agenda(turn.user_id, cmd.period, types)
            return list({o.record.id: o.record for o in occurrences}.values())
        found = self._search_target(cmd, turn, types)
        if not found and types and cmd.period is None and not types_from_text(simplify(cmd.raw_text)):
            # Тип назвала модель, а не пользователь (и даты нет) — она могла спутать напоминание с задачей.
            found = self._search_target(cmd, turn, None)
        return found

    def _search_target(self, cmd: ParsedCommand, turn: Turn, types: set[RecordType] | None) -> list[Record]:
        # Сначала ищем среди предстоящих (переносят и отмечают обычно их), затем среди всех.
        for direction in (Direction.FUTURE, Direction.ANY):
            hits = self.planner.search(turn.user_id, cmd.query, now=turn.now, keywords=cmd.keywords, types=types,
                                       direction=direction, period=cmd.period)
            if cmd.intent == Intent.COMPLETE:
                hits = [h for h in hits if h.record.status == Status.PENDING] or hits
            if hits:
                if len(hits) >= 2 and hits[0].score >= 1.6 * hits[1].score:
                    return [hits[0].record]
                return [h.record for h in hits]
        return []

    def _perform(self, cmd: ParsedCommand, record: Record, turn: Turn) -> Reply:
        if cmd.intent == Intent.DELETE:
            turn.remember([record])
            question = f"Удалить {ph.describe(record, turn.today)}?"
            return self._ask(turn, "confirm", question, action="delete", record_id=record.id)
        if cmd.intent == Intent.COMPLETE:
            turn.remember([record])
            if record.status == Status.DONE:
                return Reply(f"«{record.title}» уже отмечена выполненной.")
            self.planner.complete(record, turn.now)
            return Reply(f"Отметил как выполненное: «{record.title}».", items=[Occurrence(record, record.day)])
        return self._apply_update(record, cmd, turn)

    def _apply_update(self, record: Record, cmd: ParsedCommand, turn: Turn, skip_conflicts: bool = False) -> Reply:
        updated = replace(record)
        renamed = bool(cmd.title and cmd.title != record.title and not cmd.has_when and not cmd.shift_minutes)
        if renamed:
            updated.title = cmd.title
        if cmd.shift_minutes:
            if record.day and record.at:
                moved = datetime.combine(record.day, record.at) + timedelta(minutes=cmd.shift_minutes)
                updated.day, updated.at = moved.date(), moved.time()
            elif record.day:
                updated.day = record.day + timedelta(days=round(cmd.shift_minutes / 1440))
        if cmd.day is not None:
            updated.day = cmd.day
            updated.is_deadline = cmd.is_deadline
        if cmd.at is not None:
            updated.at = cmd.at
            if updated.day is None:
                updated.day = turn.today if datetime.combine(turn.today, cmd.at) > turn.now else turn.today + timedelta(days=1)
        if cmd.duration_min:
            updated.duration_min = cmd.duration_min
        if cmd.recurrence:
            updated.recurrence = cmd.recurrence

        turn.remember([record])
        if updated == record:
            question = f"На когда перенести {ph.type_name(record.type, 1)} «{record.title}»?"
            return self._ask(turn, "slot", question, command=command_to_dict(cmd), record_id=record.id, missing="when")

        moved = (updated.day, updated.at) != (record.day, record.at)
        if moved and updated.type == RecordType.MEETING and updated.at and not skip_conflicts:
            conflicts = self.planner.conflicts(turn.user_id, updated.day, updated.at, updated.duration_min, exclude_id=record.id)
            if conflicts:
                other = conflicts[0].record
                question = f"В это время у вас уже запланировано: {self._slot_text(other)}. Всё равно перенести?"
                return self._ask(turn, "confirm", question, action="update", record_id=record.id,
                                 command=command_to_dict(cmd))

        self.planner.save(updated, turn.now)
        if renamed and not moved:
            text = f"Переименовал: «{record.title}» → «{updated.title}»."
        else:
            text = f"Готово, перенёс {ph.describe(updated, turn.today)}."
            if updated.recurrence or updated.is_deadline:
                text = f"Готово, теперь «{updated.title}»: {ph.record_when(updated, turn.today)}."
        return Reply(text, items=[Occurrence(updated, updated.day)])
