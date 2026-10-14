"""Морфология: приведение слов к начальной форме (pymorphy3).

«Иваном», «Ивану», «Ивана» → «иван»; «презентацию» → «презентация». Это
снимает проблему падежей и в запросах пользователя, и в самих данных (в
исходном CSV встречаются формы вроде «Позвонить Иваном»).
"""

from __future__ import annotations

import re
from functools import lru_cache

import pymorphy3

_WORD_RE = re.compile(r"[a-zа-яё0-9]+(?:-[a-zа-яё]+)?", re.IGNORECASE)
# Слова, которые словарь pymorphy разбирает неверно: «демо» → «дть».
_OVERRIDES = {"демо": "демо", "тз": "тз", "кп": "кп", "срм": "срм"}


@lru_cache(maxsize=1)
def analyzer() -> pymorphy3.MorphAnalyzer:
    return pymorphy3.MorphAnalyzer()


@lru_cache(maxsize=100_000)
def lemma(word: str) -> str:
    word = word.lower().replace("ё", "е")
    if word in _OVERRIDES:
        return _OVERRIDES[word]
    if not re.search(r"[а-я]", word):
        return word
    return analyzer().parse(word)[0].normal_form.replace("ё", "е")


def words(text: str) -> list[str]:
    return _WORD_RE.findall(text)


def lemmas(text: str) -> list[str]:
    return [lemma(w) for w in words(text)]
