"""Подготовка текста ответа к озвучке.

Silero произносит только слова: цифры, латиница и типографские знаки
пропускаются. Здесь они переводятся в произносимую форму с согласованием:
«в 15:00» → «в пятнадцать ноль-ноль», «14 октября 2027» → «четырнадцатого
октября две тысячи двадцать седьмого года», «73 записи» → «семьдесят три записи».
"""

from __future__ import annotations

import re

from app.search.morph import analyzer
from app.voice.numerals import cardinal, ordinal

MONTHS_GEN = ["января", "февраля", "марта", "апреля", "мая", "июня", "июля", "августа",
              "сентября", "октября", "ноября", "декабря"]
_LATIN_LETTERS = {
    "a": "эй", "b": "би", "c": "си", "d": "ди", "e": "и", "f": "эф", "g": "джи", "h": "эйч", "i": "ай",
    "j": "джей", "k": "кей", "l": "эл", "m": "эм", "n": "эн", "o": "оу", "p": "пи", "q": "кью", "r": "ар",
    "s": "эс", "t": "ти", "u": "ю", "v": "ви", "w": "дабл-ю", "x": "икс", "y": "уай", "z": "зед",
}

_TIME = r"(?P<h{0}>\d{{1,2}}):(?P<m{0}>\d{{2}})"
_TIME_RANGE_RE = re.compile(_TIME.format(1) + r"\s*[–-]\s*" + _TIME.format(2))
_TIME_RE = re.compile(r"(?<![\d:])" + _TIME.format("") + r"(?!\d|:\d)")
_DATE_RE = re.compile(r"(?<!\d)(?P<d>\d{1,2}) (?P<mon>" + "|".join(MONTHS_GEN) + r")(?: (?P<y>\d{4}))?(?! год)")
_ORDINAL_SUFFIX_RE = re.compile(r"(?<!\d)(?P<n>\d{1,2})-?(?:го|е|ого)(?![а-яё])")
_COUNT_RE = re.compile(r"(?<![\d.,:])(?P<n>\d+) (?P<word>[а-яё]+)", re.IGNORECASE)
_NUMBER_RE = re.compile(r"\d+")
_ACRONYM_RE = re.compile(r"\b[A-Z]{2,5}\b")


def _time_words(hours: str, minutes: str, case: str = "nomn") -> str:
    h, m = int(hours), int(minutes)
    if m == 0:
        tail = "ноль-ноль"
    elif m < 10:
        tail = f"ноль {cardinal(m, 'femn', case)}"
    else:
        tail = cardinal(m, "femn", case)
    return f"{cardinal(h, case=case)} {tail}"


def _date_words(match: re.Match[str]) -> str:
    text = f"{ordinal(int(match.group('d')), 'gent', 'neut')} {match.group('mon')}"
    if match.group("y"):
        text += f" {ordinal(int(match.group('y')), 'gent', 'masc')} года"
    return text


def _noun_gender(word: str) -> str:
    for parse in analyzer().parse(word.lower()):
        if "NOUN" in parse.tag and parse.tag.gender:
            return parse.tag.gender
    return "masc"


def _count_words(match: re.Match[str]) -> str:
    word = match.group("word")
    return f"{cardinal(int(match.group('n')), _noun_gender(word))} {word}"


def _acronym(match: re.Match[str]) -> str:
    return "-".join(_LATIN_LETTERS[c] for c in match.group(0).lower())


def speakable(text: str) -> str:
    text = re.sub(r"[«»\"“”„]", "", text)
    text = re.sub(r"\s*\n+\s*", ". ", text)
    text = _TIME_RANGE_RE.sub(
        lambda m: f"с {_time_words(m['h1'], m['m1'], 'gent')} до {_time_words(m['h2'], m['m2'], 'gent')}", text)
    text = _TIME_RE.sub(lambda m: _time_words(m["h"], m["m"]), text)
    text = _DATE_RE.sub(_date_words, text)
    text = _ORDINAL_SUFFIX_RE.sub(lambda m: ordinal(int(m["n"]), "gent", "neut"), text)
    text = _COUNT_RE.sub(_count_words, text)
    text = _NUMBER_RE.sub(lambda m: cardinal(int(m.group(0))), text)
    text = _ACRONYM_RE.sub(_acronym, text)
    text = re.sub(r"\s+—\s+", ", ", text)
    text = re.sub(r"[()]", ", ", text)
    text = re.sub(r"\s+([,.:;!?])", r"\1", text)
    text = re.sub(r",(\s*[,.])+", lambda m: m.group(0)[-1], text)
    return re.sub(r"\s{2,}", " ", text).strip(" ,")
