"""Разбор реплики через Claude (Anthropic API, structured outputs) — опциональный платный вариант."""

from __future__ import annotations

import anthropic

from app.core.commands import ParsedCommand
from app.nlu.llm_prompt import (
    OUTPUT_SCHEMA, SYSTEM_PROMPT, LLMContext, LLMError, build_user_message, parse_json, to_command,
)


class ClaudeParser:
    name = "claude"

    def __init__(self, client: anthropic.Anthropic, model: str, effort: str = "low", timeout: float = 20.0):
        self.client = client.with_options(timeout=timeout, max_retries=1)
        self.model = model
        self.effort = effort

    def parse(self, text: str, ctx: LLMContext) -> ParsedCommand:
        try:
            response = self.client.beta.messages.create(
                model=self.model,
                max_tokens=8000,
                betas=["server-side-fallback-2026-07-01"],
                fallbacks="default",
                system=[{"type": "text", "text": SYSTEM_PROMPT, "cache_control": {"type": "ephemeral"}}],
                output_config={"effort": self.effort, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
                messages=[{"role": "user", "content": build_user_message(text, ctx)}],
            )
        except anthropic.AuthenticationError as exc:
            raise LLMError("неверный API-ключ") from exc
        except anthropic.RateLimitError as exc:
            raise LLMError("превышен лимит запросов") from exc
        except anthropic.APIStatusError as exc:
            raise LLMError(f"ошибка API {exc.status_code}: {exc.message}") from exc
        except anthropic.APIConnectionError as exc:
            raise LLMError("нет соединения с API") from exc

        if response.stop_reason in ("refusal", "max_tokens"):
            raise LLMError(f"модель не вернула результат (stop_reason={response.stop_reason})")
        raw = next((b.text for b in response.content if b.type == "text"), None)
        if raw is None:
            raise LLMError("в ответе нет текста")
        return to_command(parse_json(raw), text, ctx.now, source="claude")
