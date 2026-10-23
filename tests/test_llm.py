"""LLM-слой без сети и без скачивания моделей: подменяем клиентов и проверяем запрос,
разбор ответа, вычисление дат нашим парсером и откат на правила."""

import json
from datetime import date, time
from types import SimpleNamespace

from app.core.commands import Direction, Intent
from app.core.models import RecordType
from app.nlu.claude import ClaudeParser
from app.nlu.llm_prompt import LLMContext, LLMError, to_command
from app.nlu.local_llm import LocalLLMParser
from app.nlu.rules import RuleBasedParser
from app.nlu.service import NLU
from tests.conftest import NOW


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
