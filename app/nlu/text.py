"""Нормализация русской реплики.

Распознавание речи выдаёт то «в 15:00», то «в пятнадцать ноль-ноль», поэтому
реплика разбивается на токены, у каждого из которых есть исходная форма (для
названия записи) и нормализованная (для правил): нижний регистр, «ё» → «е»,
числительные словами → цифры.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

_UNITS = {
    "ноль": 0, "ноля": 0, "нуля": 0,
    "один": 1, "одна": 1, "одну": 1, "одного": 1, "одной": 1, "одно": 1,
    "два": 2, "две": 2, "двух": 2,
    "три": 3, "трех": 3,
    "четыре": 4, "четырех": 4,
    "пять": 5, "пяти": 5,
    "шесть": 6, "шести": 6,
    "семь": 7, "семи": 7,
    "восемь": 8, "восьми": 8,
    "девять": 9, "девяти": 9,
}
_TEENS = {
    "десять": 10, "десяти": 10,
    "одиннадцать": 11, "одиннадцати": 11,
    "двенадцать": 12, "двенадцати": 12,
    "тринадцать": 13, "тринадцати": 13,
    "четырнадцать": 14, "четырнадцати": 14,
    "пятнадцать": 15, "пятнадцати": 15,
    "шестнадцать": 16, "шестнадцати": 16,
    "семнадцать": 17, "семнадцати": 17,
    "восемнадцать": 18, "восемнадцати": 18,
    "девятнадцать": 19, "девятнадцати": 19,
}
_TENS = {
    "двадцать": 20, "двадцати": 20,
    "тридцать": 30, "тридцати": 30,
    "сорок": 40, "сорока": 40,
    "пятьдесят": 50, "пятидесяти": 50,
}
# Порядковые в родительном падеже: «пятнадцатого октября», «в половине третьего».
_ORDINAL_UNITS = {
    "первого": 1, "второго": 2, "третьего": 3, "четвертого": 4, "пятого": 5,
    "шестого": 6, "седьмого": 7, "восьмого": 8, "девятого": 9,
}
_ORDINAL_OTHER = {
    "десятого": 10, "одиннадцатого": 11, "двенадцатого": 12, "тринадцатого": 13,
    "четырнадцатого": 14, "пятнадцатого": 15, "шестнадцатого": 16, "семнадцатого": 17,
    "восемнадцатого": 18, "девятнадцатого": 19, "двадцатого": 20, "тридцатого": 30,
}
_SPECIAL = {
    "полтора": "1.5", "полторы": "1.5", "полчаса": "0.5 часа",
    "ноль-ноль": "00", "полдень": "12:00", "полдня": "12:00", "полночь": "0:00",
}

_STRIP_CHARS = " \t\n,;!?()[]{}«»\"'“”„…"


@dataclass
class Token:
    orig: str
    norm: str


def simplify(text: str) -> str:
    return text.lower().replace("ё", "е")


def _clean_token(word: str) -> str:
    word = word.strip(_STRIP_CHARS)
    # Точку в конце убираем, но сохраняем внутри: «15.10», «15.30».
    return word.rstrip(".")


def tokenize(text: str) -> list[Token]:
    raw = [_clean_token(w) for w in re.split(r"\s+", text.replace("—", " ").replace("–", " "))]
    words = [w for w in raw if w and w != "-"]
    tokens: list[Token] = []
    i = 0
    while i < len(words):
        orig = words[i]
        low = simplify(orig)
        # «15-го», «1-е» → «15», «1».
        low = re.sub(r"^(\d{1,2})-?(?:го|е|ого|ое)$", r"\1", low)
        if low in _SPECIAL:
            tokens.append(Token(orig, _SPECIAL[low]))
            i += 1
            continue
        if low in _TENS or low in _TEENS or low in _UNITS:
            value, consumed = _read_cardinal(words, i)
            tokens.append(Token(" ".join(words[i:i + consumed]), str(value)))
            i += consumed
            continue
        ordinal = _read_ordinal(words, i)
        if ordinal:
            value, consumed = ordinal
            tokens.append(Token(" ".join(words[i:i + consumed]), str(value)))
            i += consumed
            continue
        tokens.append(Token(orig, low))
        i += 1
    return tokens


def _read_cardinal(words: list[str], i: int) -> tuple[int, int]:
    low = simplify(words[i])
    if low in _TENS:
        value = _TENS[low]
        if i + 1 < len(words):
            nxt = simplify(words[i + 1])
            if nxt in _UNITS and _UNITS[nxt] > 0:
                return value + _UNITS[nxt], 2
            nxt_ord = _ORDINAL_UNITS.get(nxt)
            if nxt_ord:
                return value + nxt_ord, 2
        return value, 1
    if low in _TEENS:
        return _TEENS[low], 1
    return _UNITS[low], 1


def _read_ordinal(words: list[str], i: int) -> tuple[int, int] | None:
    low = simplify(words[i])
    if low in _ORDINAL_UNITS:
        return _ORDINAL_UNITS[low], 1
    if low in _ORDINAL_OTHER:
        return _ORDINAL_OTHER[low], 1
    return None


class TokenText:
    """Нормализованный текст с привязкой символов к токенам.

    Правила работают регулярными выражениями по строке `text`, а затем по
    найденным интервалам можно узнать, какие исходные слова «съедены» датой,
    временем или командой — оставшиеся слова и есть содержание записи.
    """

    def __init__(self, tokens: list[Token]):
        self.tokens = tokens
        self.offsets: list[tuple[int, int]] = []
        parts: list[str] = []
        pos = 0
        for tok in tokens:
            parts.append(tok.norm)
            self.offsets.append((pos, pos + len(tok.norm)))
            pos += len(tok.norm) + 1
        self.text = " ".join(parts)

    @classmethod
    def from_text(cls, text: str) -> "TokenText":
        return cls(tokenize(text))

    def token_indices(self, start: int, end: int) -> set[int]:
        return {i for i, (s, e) in enumerate(self.offsets) if s < end and e > start}

    def remaining(self, consumed: set[int]) -> list[Token]:
        return [t for i, t in enumerate(self.tokens) if i not in consumed]
