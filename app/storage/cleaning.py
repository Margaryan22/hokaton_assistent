"""Подготовка записей из исходного CSV.

В файле нет типа записи, поэтому он выводится по началу текста. Кроме того,
тексты сгенерированы по шаблонам с ошибками в падежах («Позвонить Иваном»,
«Напомнить Ольгой про сайту»). Ассистент озвучивает названия, поэтому при
импорте они исправляются; исходный текст сохраняется в `original_text`.
"""

from __future__ import annotations

import re

from app.core.models import RecordType
from app.search.morph import analyzer

_MEETING_PREFIXES = ("встреча", "созвон", "демо", "провести планерку", "планерка", "совещание", "обсудить", "митинг", "звонок с")
_REMINDER_PREFIXES = ("напомнить", "напоминание")


def infer_type(text: str) -> RecordType:
    """«Подготовиться к встрече…» — задача, а «Встреча с…» — встреча: смотрим на начало."""
    low = text.lower().replace("ё", "е").strip()
    if low.startswith(_MEETING_PREFIXES):
        return RecordType.MEETING
    if low.startswith(_REMINDER_PREFIXES):
        return RecordType.REMINDER
    return RecordType.TASK


# Предлоги, после которых творительный падеж уместен.
_ABLT_PREPS = {"с", "со", "между", "над", "под", "за", "перед", "передо"}
_CONSONANTS = set("бвгджзйклмнпрстфхцчшщ")


def _match_case(original: str, new: str) -> str:
    return new[:1].upper() + new[1:] if original[:1].isupper() else new


def _noun_parse(word: str, cases: set[str], animate: bool = False):
    parses = [p for p in analyzer().parse(word.lower()) if "NOUN" in p.tag and p.tag.case in cases]
    # Для предметов предпочитаем неодушевлённый разбор («аналитике» — это «аналитика», а не
    # «аналитик»), для адресатов — одушевлённый («Максимом» — это имя).
    parses.sort(key=lambda p: (("anim" in p.tag) != animate, -p.score))
    return parses[0] if parses else None


def _adj_parse(word: str, cases: set[str]):
    for p in analyzer().parse(word.lower()):
        if "ADJF" in p.tag and p.tag.case in cases:
            return p
    return None


def _to_accusative(words: list[str], start: int) -> int:
    """Ставит группу «прилагательные + существительное» с позиции `start` в винительный падеж."""
    oblique = {"datv", "loct", "gent"}
    i = start
    adjectives = []
    while i < len(words) and _adj_parse(words[i], oblique):
        adjectives.append(i)
        i += 1
    if i >= len(words):
        return start
    noun = _noun_parse(words[i], oblique)
    if noun is None or "Fixd" in noun.tag:
        return start
    new_noun = noun.inflect({"accs", "sing"})
    if new_noun is None:
        return start
    words[i] = _match_case(words[i], new_noun.word)
    gender = noun.tag.gender
    for j in adjectives:
        adj = _adj_parse(words[j], oblique)
        grammemes = {"accs", "sing", gender} if gender else {"accs", "sing"}
        if gender == "masc":
            grammemes.add("anim" if "anim" in noun.tag else "inan")
        new_adj = adj.inflect(grammemes) if adj else None
        if new_adj:
            words[j] = _match_case(words[j], new_adj.word)
    return i + 1


def fix_grammar(text: str) -> str:
    words = text.split()
    i = 0
    while i < len(words):
        low = words[i].lower()
        prev = words[i - 1].lower() if i > 0 else ""
        if low == "про" and i + 1 < len(words):
            i = _to_accusative(words, i + 1)
            continue
        if i == 0 and low == "обсудить" and len(words) > 1:
            _to_accusative(words, 1)
        noun = _noun_parse(words[i], {"ablt"}, animate=True) if i > 0 else None
        if noun is not None and prev not in _ABLT_PREPS and ("anim" in noun.tag or "Name" in noun.tag):
            target = "gent" if prev == "для" else "datv"
            new = noun.inflect({target, "sing"})
            if new is not None:
                words[i] = _match_case(words[i], new.word)
        i += 1
    fixed = " ".join(words)
    # «с Стартом» → «со Стартом»
    return re.sub(
        r"\b([Сс]) ([СсЗзШшЖж])(?=([А-Яа-яЁё]))",
        lambda m: f"{m.group(1)}о {m.group(2)}" if m.group(3).lower() in _CONSONANTS else m.group(0),
        fixed,
    )
