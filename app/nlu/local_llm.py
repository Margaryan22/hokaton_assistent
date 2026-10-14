"""Разбор реплики локальной LLM с Hugging Face (GGUF через llama.cpp).

Модель скачивается с Hugging Face при первом запросе и работает без внешних
API: на Apple Silicon — на GPU (Metal), на сервере — на CPU. Ответ ограничен
грамматикой, построенной из JSON-схемы, поэтому даже маленькая модель всегда
возвращает валидный JSON нужной структуры. Системный промпт одинаков для всех
запросов: у классических трансформеров (Qwen3) llama.cpp переиспользует его
KV-кэш, у гибридной архитектуры Qwen3.5 — нет, поэтому она заметно медленнее.
"""

from __future__ import annotations

import json
import logging
import threading
from dataclasses import dataclass

from app.core.commands import ParsedCommand
from app.nlu.llm_prompt import (
    OUTPUT_SCHEMA, SYSTEM_PROMPT, LLMContext, LLMError, build_user_message, parse_json, to_command,
)

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class LocalModel:
    repo_id: str
    filename: str
    # Гибридные модели Qwen3/Qwen3.5 по умолчанию «размышляют»; пустой блок <think> это отключает.
    skip_thinking: bool


# Модели, которые сравнивались на наборе фраз eval/nlu_cases.jsonl (результаты — в README).
LOCAL_MODELS: dict[str, LocalModel] = {
    "qwen3.5-4b": LocalModel("unsloth/Qwen3.5-4B-GGUF", "Qwen3.5-4B-Q4_K_M.gguf", True),
    "qwen3-4b-2507": LocalModel("unsloth/Qwen3-4B-Instruct-2507-GGUF", "Qwen3-4B-Instruct-2507-Q4_K_M.gguf", False),
    "qwen3.5-2b": LocalModel("unsloth/Qwen3.5-2B-GGUF", "Qwen3.5-2B-Q4_K_M.gguf", True),
    "qvikhr-3-4b": LocalModel("mradermacher/QVikhr-3-4B-Instruction-GGUF", "QVikhr-3-4B-Instruction.Q4_K_M.gguf", True),
}

def chatml(system: str, user: str, skip_thinking: bool) -> str:
    prompt = f"<|im_start|>system\n{system}<|im_end|>\n<|im_start|>user\n{user}<|im_end|>\n<|im_start|>assistant\n"
    return prompt + ("<think>\n\n</think>\n\n" if skip_thinking else "")


class LocalLLMParser:
    name = "local"

    def __init__(self, model_key: str = "qwen3.5-4b", *, model_path: str | None = None, n_ctx: int = 4096,
                 n_threads: int | None = None, gpu_layers: int = -1, max_tokens: int = 400):
        if model_key not in LOCAL_MODELS:
            raise ValueError(f"Неизвестная модель {model_key!r}; доступны: {', '.join(LOCAL_MODELS)}")
        self.model_key = model_key
        self.spec = LOCAL_MODELS[model_key]
        self.model_path = model_path
        self.n_ctx, self.n_threads, self.gpu_layers, self.max_tokens = n_ctx, n_threads, gpu_layers, max_tokens
        self._llm = None
        self._grammar = None
        self._lock = threading.Lock()

    @property
    def model_name(self) -> str:
        return self.spec.filename.removesuffix(".gguf")

    def _load(self):
        if self._llm is None:
            from huggingface_hub import hf_hub_download
            from llama_cpp import Llama, LlamaGrammar

            path = self.model_path or hf_hub_download(self.spec.repo_id, self.spec.filename)
            self._llm = Llama(model_path=path, n_ctx=self.n_ctx, n_threads=self.n_threads,
                              n_gpu_layers=self.gpu_layers, seed=0, verbose=False)
            self._grammar = LlamaGrammar.from_json_schema(json.dumps(OUTPUT_SCHEMA), verbose=False)
            log.info("Локальная LLM %s загружена", self.model_name)
        return self._llm

    def warm_up(self) -> None:
        """Загружает модель и заранее вычисляет KV-кэш системного промпта."""
        with self._lock:
            llm = self._load()
            llm.create_completion(chatml(SYSTEM_PROMPT, "", self.spec.skip_thinking), max_tokens=1)

    def parse(self, text: str, ctx: LLMContext) -> ParsedCommand:
        prompt = chatml(SYSTEM_PROMPT, build_user_message(text, ctx), self.spec.skip_thinking)
        try:
            with self._lock:
                llm = self._load()
                result = llm.create_completion(prompt, grammar=self._grammar, max_tokens=self.max_tokens,
                                               temperature=0.0, stop=["<|im_end|>"])
        except (OSError, ValueError, RuntimeError) as exc:
            raise LLMError(f"локальная модель недоступна: {exc}") from exc
        choice = result["choices"][0]
        if choice.get("finish_reason") == "length":
            raise LLMError("модель не уложилась в лимит токенов")
        return to_command(parse_json(choice["text"]), text, ctx.now, source="local")
