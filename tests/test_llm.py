"""LLM-слой без сети и без скачивания моделей: подменяем клиентов и проверяем запрос,
разбор ответа, вычисление дат нашим парсером и откат на правила."""

import json
from datetime import date, time
from types import SimpleNamespace

import pytest

from app.core.commands import Direction, Intent
from app.core.models import Record, RecordType
from app.nlu.claude import ClaudeParser
from app.nlu.llm_prompt import LLMContext, LLMError, to_command
from app.nlu.local_llm import LocalLLMParser
from app.nlu.rules import RuleBasedParser
from app.nlu.service import NLU
from tests.conftest import FRIDAY, NOW, Dialog


def answer(**fields):
    base = {"intent": "unknown", "title": None, "when": None, "target": None, "query": None, "types": [],
            "direction": "any", "refers": False, "selection": None}
    return {**base, **fields}


# --- перевод ответа модели в команду ---------------------------------------------

def test_dates_are_computed_by_code_not_by_model():
    cmd = to_command(answer(intent="create_meeting", title="созвон с Петровым", when="послезавтра в 16:00"), "x", NOW)
    assert (cmd.title, cmd.day, cmd.at) == ("Созвон с Петровым", date(2026, 10, 15), time(16, 0))
    cmd = to_command(answer(intent="create_meeting", title="Обед", when="на следующей неделе в среду в 13:00"), "x", NOW)
    assert (cmd.day, cmd.at) == (date(2026, 10, 21), time(13, 0))


def test_update_target_and_shift():
    cmd = to_command(answer(intent="update", when="в понедельник в 12:00", target="встреча с Иваном в пятницу"), "x", NOW)
    assert (cmd.query, cmd.period.start, cmd.day, cmd.at) == ("встреча с Иваном", date(2026, 10, 16), date(2026, 10, 19), time(12, 0))
    cmd = to_command(answer(intent="update", when="на неделю", target="задача про отчёт"), "x", NOW)
    assert cmd.shift_minutes == 7 * 24 * 60 and cmd.day is None
    cmd = to_command(answer(intent="update", when="на завтра", refers=True), "x", NOW)
    assert cmd.refers_to_previous and cmd.day == date(2026, 10, 14)


def test_search_without_topic_becomes_list():
    cmd = to_command(answer(intent="search", query="встречи в сентябре", direction="past"), "какие встречи были в сентябре", NOW)
    assert cmd.intent == Intent.LIST and (cmd.period.start, cmd.period.end) == (date(2026, 9, 1), date(2026, 9, 30))
    assert cmd.record_types == {RecordType.MEETING}  # тип восстановлен по словам фразы


def test_task_with_period_gets_deadline():
    cmd = to_command(answer(intent="create_task", title="Купить билеты", when="до конца недели"), "x", NOW)
    assert (cmd.day, cmd.is_deadline) == (date(2026, 10, 18), True)


def test_invalid_answer_is_rejected():
    try:
        to_command(answer(intent="dance"), "x", NOW)
    except LLMError:
        return
    raise AssertionError("ожидалась ошибка")


# --- локальная модель (llama.cpp) ---------------------------------------------------

class FakeLlama:
    def __init__(self, payload, finish_reason="stop"):
        self.payload, self.finish_reason, self.prompts = payload, finish_reason, []

    def create_completion(self, prompt, **kwargs):
        self.prompts.append((prompt, kwargs))
        return {"choices": [{"text": json.dumps(self.payload, ensure_ascii=False), "finish_reason": self.finish_reason}]}


def local_parser(payload, **kwargs):
    parser = LocalLLMParser("qwen3-4b-2507")
    parser._llm, parser._grammar = FakeLlama(payload, **kwargs), "grammar"
    return parser


def test_local_parser_prompt_and_grammar():
    parser = local_parser(answer(intent="list_events", when="на этой неделе"))
    cmd = parser.parse("мои дела на эту неделю", LLMContext(now=NOW, timezone="Europe/Moscow", pending_question="На какое время?"))
    prompt, kwargs = parser._llm.prompts[0]
    assert prompt.startswith("<|im_start|>system") and "На какое время?" in prompt and "мои дела" in prompt
    assert kwargs["grammar"] == "grammar" and kwargs["temperature"] == 0.0
    assert cmd.source == "local" and cmd.period.start == date(2026, 10, 12)


def test_local_parser_reports_truncation():
    parser = local_parser(answer(intent="help"), finish_reason="length")
    try:
        parser.parse("что-то", LLMContext(now=NOW, timezone="x"))
    except LLMError:
        return
    raise AssertionError("ожидалась ошибка")


# --- Claude ------------------------------------------------------------------------

class FakeMessages:
    def __init__(self, payload=None, error=None, stop_reason="end_turn"):
        self.payload, self.error, self.stop_reason, self.calls = payload, error, stop_reason, []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        text = SimpleNamespace(type="text", text=json.dumps(self.payload, ensure_ascii=False))
        return SimpleNamespace(stop_reason=self.stop_reason, content=[text])


class FakeClient:
    def __init__(self, messages):
        self.beta = SimpleNamespace(messages=messages)

    def with_options(self, **_):
        return self


def test_claude_request():
    messages = FakeMessages(answer(intent="create_meeting", title="Встреча с Иваном", when="завтра в 15:00"))
    cmd = ClaudeParser(FakeClient(messages), "claude-opus-5-5").parse("x", LLMContext(now=NOW, timezone="x"))
    call = messages.calls[0]
    assert call["output_config"]["format"]["type"] == "json_schema" and call["fallbacks"] == "default"
    assert cmd.source == "claude" and (cmd.day, cmd.at) == (date(2026, 10, 14), time(15, 0))


# --- откат на правила ----------------------------------------------------------------

def test_falls_back_to_rules_on_llm_error():
    nlu = NLU(RuleBasedParser(), ClaudeParser(FakeClient(FakeMessages(error=LLMError("нет сети"))), "m"))
    cmd = nlu.parse("Напомни мне завтра в 10 утра позвонить Ивану", now=NOW)
    assert cmd.source == "rules" and cmd.intent == Intent.CREATE_REMINDER


def test_rules_win_when_llm_does_not_understand():
    nlu = NLU(RuleBasedParser(), local_parser(answer(intent="help")))
    cmd = nlu.parse("есть что-нибудь на выходных?", now=NOW)
    assert cmd.source == "rules" and cmd.intent == Intent.LIST


def test_implicit_rules_create_does_not_override_llm():
    nlu = NLU(RuleBasedParser(), local_parser(answer(intent="unknown")))
    cmd = nlu.parse("завтра Петров", now=NOW)
    assert cmd.source == "local" and cmd.intent == Intent.UNKNOWN
    cmd = nlu.parse("поставь задачу Петров на завтра", now=NOW)
    assert cmd.source == "rules" and cmd.intent == Intent.CREATE_TASK


def test_llm_result_is_used_when_confident():
    nlu = NLU(RuleBasedParser(), local_parser(answer(intent="complete", target="отчёт", direction="any")))
    cmd = nlu.parse("готово, отчёт отправил", now=NOW)
    assert cmd.source == "local" and cmd.intent == Intent.COMPLETE and cmd.query == "отчёт"
    assert cmd.direction == Direction.ANY


# --- удаление и изменение по ответам модели из разбора бага ------------------------------
# Модель раскладывает поля нестабильно: описание то в target, то в title, дата то в target, то в when,
# напоминание называет задачей. Записи из CSV пользователя demo с «Иваном» в названии — помеха для поиска.

class ScriptedLlama:
    """Отвечает на каждую реплику заранее заданным JSON; «да» модель понимает как confirm."""

    def __init__(self, script):
        self.script = {"да": answer(intent="confirm"), **script}

    def create_completion(self, prompt, **kwargs):
        said = prompt.rsplit("Реплика пользователя: «", 1)[1].split("»<|im_end|>", 1)[0]
        payload = self.script[said]
        return {"choices": [{"text": json.dumps(payload, ensure_ascii=False), "finish_reason": "stop"}]}


def llm_dialog(container, script):
    parser = LocalLLMParser("qwen3-4b-2507")
    parser._llm, parser._grammar = ScriptedLlama(script), "grammar"
    container.assistant.nlu = NLU(RuleBasedParser(), parser)
    planner = container.planner
    reminder = planner.create(Record(type=RecordType.REMINDER, title="Позвонить Ивану", user_id="demo",
                                     day=date(2026, 10, 26), at=time(12, 0)), FRIDAY)
    meeting = planner.create(Record(type=RecordType.MEETING, title="Встреча с командой", user_id="demo",
                                    day=date(2026, 10, 30), at=time(19, 0)), FRIDAY)
    return Dialog(container, now=FRIDAY), reminder, meeting


DELETE_REMINDER = "Удали напоминание позвонить Ивану 26 октября в 12:00."
ASK_DELETE_REMINDER = "Можешь удалить напоминание позвонить Ивану 26 октября в 12:00?"
ASK_DELETE_MEETING = "Можешь удалить встречу 30 октября, встреча с командой на семь вечера?"


@pytest.mark.parametrize("text, payload", [
    # Случай 1: дата в when, напоминание названо задачей.
    (DELETE_REMINDER, answer(intent="delete", target="позвонить Ивану", when="26 октября в 12:00", types=["task"])),
    # Случай 2: описание в title вместо target.
    (ASK_DELETE_REMINDER, answer(intent="delete", title="позвонить Ивану", when="26 октября в 12:00",
                                 types=["reminder"])),
    (ASK_DELETE_REMINDER, answer(intent="delete", title="Позвонить Ивану", when="26 октября в 12:00", types=["task"],
                                 refers=True)),
    # Описание вместе с датой в target.
    (DELETE_REMINDER, answer(intent="delete", target="напоминание позвонить Ивану 26 октября в 12:00",
                             types=["reminder"])),
])
def test_llm_delete_reminder(container, text, payload):
    dialog, reminder, _ = llm_dialog(container, {text: payload})
    reply = dialog.say(text)
    assert reply.awaiting == "confirm", reply.text
    assert reply.text == "Удалить напоминание «Позвонить Ивану» на понедельник, 26 октября, в 12:00?"
    assert dialog.say("да").text.startswith("Удалил напоминание")
    assert container.records.get("demo", reminder.id) is None


@pytest.mark.parametrize("payload", [
    answer(intent="delete", target="встреча с командой", when="30 октября в 19:00", types=["meeting"]),
    answer(intent="delete", target="встречу 30 октября, встреча с командой", when="на семь вечера", types=["meeting"]),
    answer(intent="delete", title="Встреча с командой", when="30 октября в 19:00", types=[]),
])
def test_llm_delete_meeting(container, payload):
    dialog, _, meeting = llm_dialog(container, {ASK_DELETE_MEETING: payload})
    reply = dialog.say(ASK_DELETE_MEETING)
    assert reply.text == "Удалить встречу «Встреча с командой» на пятницу, 30 октября, в 19:00?", reply.text
    dialog.say("да")
    assert container.records.get("demo", meeting.id) is None


def test_llm_delete_with_wrong_date_does_not_offer_other_records(container):
    text = "Удали напоминание позвонить Ивану 27 октября"
    dialog, reminder, _ = llm_dialog(container, {text: answer(intent="delete", target="позвонить Ивану",
                                                              when="27 октября", types=["task"])})
    reply = dialog.say(text)
    assert reply.awaiting is None and not reply.items
    assert reply.text.startswith("Не нашёл напоминание «позвонить Ивану» во вторник, 27 октября")
    assert "удали напоминание позвонить Ивану" in reply.text
    assert container.records.get("demo", reminder.id) is not None


def test_llm_update_keeps_when_as_new_time(container):
    text = "перенеси встречу с командой на 20:00"
    dialog, _, meeting = llm_dialog(container, {text: answer(intent="update", target="встреча с командой",
                                                             when="в 20:00", types=["meeting"])})
    reply = dialog.say(text)
    assert "перенёс" in reply.text, reply.text
    moved = container.records.get("demo", meeting.id)
    assert (moved.day, moved.at) == (date(2026, 10, 30), time(20, 0))


def test_rules_win_when_llm_misreads_explicit_delete(container):
    text = "Удали встречу с командой"
    dialog, _, meeting = llm_dialog(container, {text: answer(intent="update", target="встреча с командой",
                                                             types=["meeting"])})
    reply = dialog.say(text)
    assert reply.text.startswith("Удалить встречу «Встреча с командой»"), reply.text
    dialog.say("да")
    assert container.records.get("demo", meeting.id) is None


def test_llm_refers_without_pronoun_is_ignored(container):
    text = "удали встречу с командой"
    dialog, _, meeting = llm_dialog(container, {text: answer(intent="delete", target=None, title=None,
                                                             types=["meeting"], refers=True)})
    reply = dialog.say(text)
    assert reply.text.startswith("Удалить встречу «Встреча с командой»"), reply.text


def test_llm_copies_record_from_context_for_pronoun(container):
    # Реальный ответ Qwen3-4B: тема и дата взяты из «записей прошлого ответа», direction — past.
    dialog, _, _ = llm_dialog(container, {
        "Напомни в понедельник в 9 оплатить интернет": answer(
            intent="create_reminder", title="Оплатить интернет", when="в понедельник в 9:00", types=["reminder"]),
        "удали это напоминание": answer(
            intent="delete", when="понедельник, 26 октября, в 09:00", target="Оплатить интернет",
            types=["reminder"], direction="past", refers=True),
    })
    dialog.say("Напомни в понедельник в 9 оплатить интернет")
    reply = dialog.say("удали это напоминание")
    assert reply.text == "Удалить напоминание «Оплатить интернет» на понедельник, 26 октября, в 09:00?", reply.text


def test_delete_date_from_when_is_not_moved_to_past():
    cmd = to_command(answer(intent="delete", target="Оплатить интернет", when="понедельник, 26 октября, в 09:00",
                            direction="past"), "удали напоминание оплатить интернет", FRIDAY)
    assert (cmd.period.start, cmd.target_at) == (date(2026, 10, 26), time(9, 0))


@pytest.mark.parametrize("direction", ["past", "any"])
def test_llm_delete_past_date_without_year(container, direction):
    text = "удали напоминание за 20 октября"
    dialog, _, _ = llm_dialog(container, {text: answer(intent="delete", target="напоминание", when="20 октября",
                                                       types=["reminder"], direction=direction)})
    past = container.planner.create(Record(type=RecordType.REMINDER, title="Оплатить связь", user_id="demo",
                                           day=date(2026, 10, 20), at=time(9, 0)), FRIDAY)
    reply = dialog.say(text)
    assert reply.text == "Удалить напоминание «Оплатить связь» на вторник, 20 октября, в 09:00?", reply.text
    dialog.say("да")
    assert container.records.get("demo", past.id) is None
