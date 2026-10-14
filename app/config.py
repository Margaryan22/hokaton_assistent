"""Настройки из переменных окружения (и файла .env, если он есть)."""

from __future__ import annotations

import importlib.util
import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from zoneinfo import ZoneInfo

ROOT = Path(__file__).resolve().parent.parent


def _load_dotenv(path: Path) -> None:
    if not path.is_file():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


def installed(module: str) -> bool:
    return importlib.util.find_spec(module) is not None


@dataclass(frozen=True)
class Settings:
    db_path: str
    timezone: str
    seed_csv: str | None
    seed_user: str
    # LLM для понимания фраз: local (модель с Hugging Face через llama.cpp), claude, off.
    llm_backend: str = "off"
    local_llm_model: str = "qwen3-4b-2507"
    local_llm_path: str | None = None
    claude_model: str = "claude-opus-5-5"
    claude_effort: str = "low"
    llm_timeout: float = 20.0
    # Голос на сервере: распознавание (GigaAM) и синтез (Silero). off — голос остаётся в браузере.
    stt_model: str = "off"
    tts_speaker: str = "off"
    # Модель эмбеддингов для смыслового поиска (sentence-transformers) или off.
    embeddings_model: str = "off"
    # Загружать модели при старте, а не при первом запросе.
    preload_models: bool = False

    @property
    def zone(self) -> ZoneInfo:
        return ZoneInfo(self.timezone)


def _auto(value: str, default: str, available: bool) -> str:
    value = value.strip()
    if value.lower() == "auto":
        return default if available else "off"
    return value


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    _load_dotenv(ROOT / ".env")
    env = os.environ.get
    seed = env("SEED_CSV", str(ROOT / "data" / "voice_assistant_events.csv"))

    llm_backend = env("LLM_BACKEND", "auto").lower()
    if llm_backend == "auto":
        if installed("llama_cpp"):
            llm_backend = "local"
        elif env("ANTHROPIC_API_KEY") or env("ANTHROPIC_AUTH_TOKEN"):
            llm_backend = "claude"
        else:
            llm_backend = "off"

    return Settings(
        db_path=env("DB_PATH", str(ROOT / "data" / "assistant.db")),
        timezone=env("APP_TIMEZONE", "Europe/Moscow"),
        seed_csv=seed or None,
        seed_user=env("SEED_USER", "demo"),
        llm_backend=llm_backend,
        local_llm_model=env("LOCAL_LLM_MODEL", "qwen3-4b-2507"),
        local_llm_path=env("LOCAL_LLM_PATH") or None,
        claude_model=env("CLAUDE_MODEL", "claude-opus-5-5"),
        claude_effort=env("CLAUDE_EFFORT", "low"),
        llm_timeout=float(env("LLM_TIMEOUT", "20")),
        stt_model=_auto(env("STT_MODEL", "auto"), "gigaam-v3-e2e-rnnt", installed("onnx_asr")),
        tts_speaker=_auto(env("TTS_SPEAKER", "auto"), "xenia", installed("silero")),
        embeddings_model=_auto(env("EMBEDDINGS_MODEL", "auto"), "deepvk/USER2-small", installed("sentence_transformers")),
        preload_models=env("PRELOAD_MODELS", "1") not in ("0", "false", "no"),
    )
