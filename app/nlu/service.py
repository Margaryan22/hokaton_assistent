"""Точка входа NLU: сначала LLM (локальная или Claude, если настроена), при любой её ошибке — правила."""

from __future__ import annotations

import logging
from datetime import datetime
from typing import Any, Protocol

from app.core.commands import Intent, ParsedCommand
from app.nlu.llm_prompt import LLMContext, LLMError
from app.nlu.rules import ParseContext, RuleBasedParser

log = logging.getLogger(__name__)


class LLMBackend(Protocol):
    name: str

    def parse(self, text: str, ctx: LLMContext) -> ParsedCommand: ...


class NLU:
    def __init__(self, rules: RuleBasedParser, llm: LLMBackend | None = None, timezone: str = "Europe/Moscow"):
        self.rules = rules
        self.llm = llm
        self.timezone = timezone

    @property
    def mode(self) -> str:
        return f"{self.llm.name}+rules" if self.llm else "rules"

    def parse(
        self,
        text: str,
        *,
        now: datetime,
        awaiting: str | None = None,
        pending_question: str | None = None,
        recent_records: list[dict[str, Any]] | None = None,
        history: list[tuple[str, str]] | None = None,
    ) -> ParsedCommand:
        if self.llm is None:
            return self._rules(text, now, awaiting)
        try:
            cmd = self.llm.parse(text, LLMContext(
                now=now, timezone=self.timezone, pending_question=pending_question,
                recent_records=recent_records or [], history=history or [],
            ))
        except LLMError as exc:
            log.warning("LLM недоступна, разбираю правилами: %s", exc)
            return self._rules(text, now, awaiting)
        # Если модель не поняла команду (или «ответила» без вопроса), а правила уверенно её узнали — верим правилам.
        # Создание, угаданное правилами без явного триггера («завтра Петров»), уверенным не считается.
        if cmd.intent in (Intent.UNKNOWN, Intent.HELP) or (cmd.intent == Intent.ANSWER and not awaiting):
            fallback = self._rules(text, now, awaiting)
            if fallback.intent not in (Intent.UNKNOWN, Intent.HELP, Intent.ANSWER) and not fallback.implicit:
                return fallback
        return cmd

    def _rules(self, text: str, now: datetime, awaiting: str | None) -> ParsedCommand:
        return self.rules.parse(text, ParseContext(now=now, awaiting=awaiting))
