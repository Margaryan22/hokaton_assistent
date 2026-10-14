"""Числительные словами: количественные (с родом) и порядковые (с падежом).

Синтезатор Silero пропускает цифры, поэтому всё, что ассистент произносит,
сначала переводится в слова: «14 октября» → «четырнадцатого октября».
"""

from __future__ import annotations

from functools import lru_cache

from app.search.morph import analyzer

_UNITS = ["ноль", "один", "два", "три", "четыре", "пять", "шесть", "семь", "восемь", "девять"]
_TEENS = ["десять", "одиннадцать", "двенадцать", "тринадцать", "четырнадцать", "пятнадцать", "шестнадцать",
          "семнадцать", "восемнадцать", "девятнадцать"]
_TENS = ["", "", "двадцать", "тридцать", "сорок", "пятьдесят", "шестьдесят", "семьдесят", "восемьдесят", "девяносто"]
_HUNDREDS = ["", "сто", "двести", "триста", "четыреста", "пятьсот", "шестьсот", "семьсот", "восемьсот", "девятьсот"]

_ORD_UNITS = ["нулевой", "первый", "второй", "третий", "четвёртый", "пятый", "шестой", "седьмой", "восьмой", "девятый"]
_ORD_TEENS = ["десятый", "одиннадцатый", "двенадцатый", "тринадцатый", "четырнадцатый", "пятнадцатый",
              "шестнадцатый", "семнадцатый", "восемнадцатый", "девятнадцатый"]
_ORD_TENS = ["", "", "двадцатый", "тридцатый", "сороковой", "пятидесятый", "шестидесятый", "семидесятый",
             "восьмидесятый", "девяностый"]
_ORD_HUNDREDS = ["", "сотый", "двухсотый", "трёхсотый", "четырёхсотый", "пятисотый", "шестисотый", "семисотый",
                 "восьмисотый", "девятисотый"]


def _below_thousand(n: int, gender: str) -> list[str]:
    words: list[str] = []
    hundreds, rest = divmod(n, 100)
    if hundreds:
        words.append(_HUNDREDS[hundreds])
    if 10 <= rest <= 19:
        words.append(_TEENS[rest - 10])
        return words
    tens, units = divmod(rest, 10)
    if tens:
        words.append(_TENS[tens])
    if units or not words:
        if units == 1:
            words.append({"femn": "одна", "neut": "одно"}.get(gender, "один"))
        elif units == 2:
            words.append("две" if gender == "femn" else "два")
        else:
            words.append(_UNITS[units])
    return words


def _thousands_word(n: int) -> str:
    last_two, last = n % 100, n % 10
    if 11 <= last_two <= 14:
        return "тысяч"
    if last == 1:
        return "тысяча"
    if 2 <= last <= 4:
        return "тысячи"
    return "тысяч"


def cardinal(n: int, gender: str = "masc", case: str = "nomn") -> str:
    """Количественное числительное: cardinal(2, "femn") → «две», cardinal(15, case="gent") → «пятнадцати»."""
    if n < 0:
        return "минус " + cardinal(-n, gender, case)
    if n >= 1_000_000:
        return " ".join(str(n))
    thousands, rest = divmod(n, 1000)
    words: list[str] = []
    if thousands:
        head = [] if thousands == 1 else _below_thousand(thousands, "femn")
        words += head + [_thousands_word(thousands)]
    if rest or not words:
        words += _below_thousand(rest, gender)
    if case != "nomn":
        words = [_inflect_numeral(w, case) for w in words]
    return " ".join(words)


@lru_cache(maxsize=1024)
def _inflect_numeral(word: str, case: str) -> str:
    for parse in analyzer().parse(word):
        if parse.tag.POS in ("NUMR", "ADJF", "NOUN") and parse.word == word:
            form = parse.inflect({case})
            if form:
                return form.word
    return word


@lru_cache(maxsize=4096)
def _inflect(word: str, case: str, gender: str) -> str:
    parse = next((p for p in analyzer().parse(word) if "ADJF" in p.tag), None)
    if parse is None:
        return word
    grammemes = {case, "sing", gender} if gender in ("masc", "femn", "neut") else {case, "plur"}
    if case == "accs" and gender == "masc":
        grammemes.add("inan")
    form = parse.inflect(grammemes)
    return form.word if form else word


def ordinal(n: int, case: str = "nomn", gender: str = "masc") -> str:
    """Порядковое: ordinal(14, "gent", "neut") → «четырнадцатого»; ordinal(2027, "gent") → «две тысячи двадцать седьмого»."""
    thousands, rest = divmod(n, 1000)
    prefix: list[str] = []
    if thousands and rest:
        prefix = _below_thousand(thousands, "femn") + [_thousands_word(thousands)]
    elif thousands:
        head = "" if thousands == 1 else cardinal(thousands, case="gent").replace(" ", "")
        word = (head + "тысячный") if head else "тысячный"
        return _inflect(word, case, gender)
    hundreds, below = divmod(rest, 100)
    if below == 0:
        return " ".join(prefix + [_inflect(_ORD_HUNDREDS[hundreds], case, gender)])
    if hundreds:
        prefix.append(_HUNDREDS[hundreds])
    if 10 <= below <= 19:
        last = _ORD_TEENS[below - 10]
    else:
        tens, units = divmod(below, 10)
        if units == 0:
            last = _ORD_TENS[tens]
        else:
            if tens:
                prefix.append(_TENS[tens])
            last = _ORD_UNITS[units]
    return " ".join(prefix + [_inflect(last, case, gender)])
