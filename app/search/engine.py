"""Поиск записей по естественному запросу.

Ранжирование — BM25 по леммам, где каждое слово запроса расширяется:
  * синонимами из словаря (вес 0.6) и словами, которые подсказала LLM (0.7);
  * нечёткими совпадениями со словарём коллекции (опечатки распознавания речи);
  * совпадением по началу слова («презентационный» ~ «презентация»).
Для каждого слова запроса берётся лучшее из его расширений, а итог умножается на
долю покрытых слов — запись, где есть и «Иван», и «договор», выше записи только
с «Иваном».

Если подключена модель эмбеддингов (`SemanticIndex`), к нормированной лексической
оценке добавляется смысловое сходство: «финансы» находит «бюджет» и «смету»,
«звонок Ивану» — «Позвонить Ивану», даже без общих слов и синонимов в словаре.
"""

from __future__ import annotations

import logging
import math
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime

from rapidfuzz import fuzz, process

from app.core.commands import Direction
from app.core.models import Period, Record, RecordType, SearchHit
from app.search.morph import lemma, lemmas
from app.search.synonyms import SYNONYMS, expand_abbreviations

STOPWORDS = {
    "и", "в", "во", "на", "с", "со", "по", "про", "о", "об", "обо", "для", "к", "ко", "у", "я", "мне", "меня",
    "мой", "моя", "мое", "мои", "это", "этот", "тот", "та", "что", "когда", "где", "какой", "который", "как",
    "найти", "поискать", "искать", "показать", "напомнить", "найди", "был", "быть", "есть", "весь", "все",
    "когда-то", "когда-нибудь", "ставить", "поставить", "создать", "добавить", "она", "он", "оно", "они",
    "её", "ее", "его", "их", "мы", "нас", "наш", "там", "тут", "же", "ли", "бы", "не", "а", "но", "или",
    "от", "до", "за", "из", "под", "над", "при", "через", "тот", "такой", "какой-то", "некий", "один",
    "нужно", "надо", "можно", "пожалуйста", "свой", "запланировать", "запланированный", "насчет", "касательно",
    "раньше", "давно", "недавно", "прошлый", "старый", "ранее", "вспомнить", "помнить", "посмотреть",
    "последний", "раз", "ну", "вот", "все", "что-нибудь", "что-то",
}

# Слова, обозначающие тип записи: в поиске они превращаются в фильтр, а не в слова запроса.
TYPE_WORDS: dict[str, set[RecordType]] = {
    "задача": {RecordType.TASK, RecordType.REMINDER},
    "задачка": {RecordType.TASK, RecordType.REMINDER},
    "дело": set(),
    "напоминание": {RecordType.REMINDER},
    "напоминалка": {RecordType.REMINDER},
    "встреча": {RecordType.MEETING},
    "событие": set(),
    "запись": set(),
}

K1, B = 1.2, 0.75
# Косинусы эмбеддингов сжаты (медиана по коллекции ~0.7), поэтому сигнал — превышение над медианой.
SEMANTIC_MARGIN, SEMANTIC_SPAN, SEMANTIC_WEIGHT = 0.10, 0.15, 0.5

log = logging.getLogger(__name__)


@dataclass
class _Doc:
    record: Record
    terms: Counter


def content_terms(text: str) -> list[str]:
    terms = expand_abbreviations(lemmas(text))
    return [t for t in terms if t.isdigit() or (t not in STOPWORDS and t not in TYPE_WORDS and len(t) > 1)]


class SearchEngine:
    def __init__(self, fuzzy_cutoff: float = 82.0, relative_threshold: float = 0.45, semantic=None):
        self.fuzzy_cutoff = fuzzy_cutoff
        self.relative_threshold = relative_threshold
        self.semantic = semantic

    def search(
        self,
        records: list[Record],
        query: str,
        *,
        now: datetime,
        keywords: list[str] | None = None,
        types: set[RecordType] | None = None,
        direction: Direction = Direction.ANY,
        period: Period | None = None,
        limit: int | None = 10,
    ) -> list[SearchHit]:
        candidates = [r for r in records if self._passes(r, types, direction, period, now)]
        if not candidates:
            return []
        query_terms = list(dict.fromkeys(content_terms(query)))
        extra = [lemma(k) for k in (keywords or [])]
        extra = [k for k in dict.fromkeys(expand_abbreviations(extra)) if k not in query_terms and k not in STOPWORDS]
        if not query_terms and not extra and self.semantic is None:
            return []

        docs = [_Doc(r, Counter(content_terms(r.title))) for r in candidates]
        df: Counter = Counter()
        for d in docs:
            df.update(set(d.terms))
        vocab = list(df)
        n_docs = len(docs)
        avgdl = sum(sum(d.terms.values()) for d in docs) / n_docs or 1.0

        groups = [self._expand(t, vocab) for t in query_terms]
        if extra:
            groups.append({k: 0.7 for k in extra})

        def bm25(term: str, doc: _Doc) -> float:
            tf = doc.terms.get(term, 0)
            if not tf:
                return 0.0
            idf = math.log(1 + (n_docs - df[term] + 0.5) / (df[term] + 0.5))
            dl = sum(doc.terms.values())
            return idf * tf * (K1 + 1) / (tf + K1 * (1 - B + B * dl / avgdl))

        lexical: list[tuple[float, list[str]]] = []
        for doc in docs:
            total, covered, matched = 0.0, 0, []
            for i, group in enumerate(groups):
                best, best_term = 0.0, None
                for term, weight in group.items():
                    score = weight * bm25(term, doc)
                    if score > best:
                        best, best_term = score, term
                if best > 0:
                    total += best
                    matched.append(best_term)
                    if i < len(query_terms):
                        covered += 1
            coverage = covered / len(query_terms) if query_terms else 1.0
            lexical.append((total * (0.4 + 0.6 * coverage), matched))

        best_lexical = max((score for score, _ in lexical), default=0.0)
        semantic = self._semantic_scores(query, docs)
        hits: list[SearchHit] = []
        for doc, (score, matched), sem in zip(docs, lexical, semantic):
            combined = (score / best_lexical if best_lexical else 0.0) + SEMANTIC_WEIGHT * sem
            if combined > 0:
                hits.append(SearchHit(doc.record, combined, matched))
        if not hits:
            return []
        best = max(h.score for h in hits)
        hits = [h for h in hits if h.score >= best * self.relative_threshold]
        hits.sort(key=lambda h: self._order_key(h, best, direction, now.date()))
        return hits if limit is None else hits[:limit]

    def _semantic_scores(self, query: str, docs: list[_Doc]) -> list[float]:
        if self.semantic is None or not query.strip():
            return [0.0] * len(docs)
        try:
            sims = self.semantic.similarities(query, [d.record.title for d in docs])
        except Exception:  # noqa: BLE001 — без эмбеддингов поиск продолжает работать по словам
            log.exception("Смысловой поиск недоступен")
            return [0.0] * len(docs)
        floor = float(sorted(sims)[len(sims) // 2]) + SEMANTIC_MARGIN
        return [min(max((float(s) - floor) / SEMANTIC_SPAN, 0.0), 1.0) for s in sims]

    def _expand(self, term: str, vocab: list[str]) -> dict[str, float]:
        group = {term: 1.0}
        for syn in SYNONYMS.get(term, ()):
            group.setdefault(syn, 0.6)
        if len(term) >= 4:
            for candidate, similarity, _ in process.extract(term, vocab, scorer=fuzz.ratio, score_cutoff=self.fuzzy_cutoff, limit=5):
                if candidate != term:
                    group[candidate] = max(group.get(candidate, 0), 0.8 * similarity / 100)
            prefix = term[:5] if len(term) >= 6 else term[:4]
            for candidate in vocab:
                if candidate != term and len(candidate) >= 4 and candidate.startswith(prefix):
                    group[candidate] = max(group.get(candidate, 0), 0.5)
        return group

    @staticmethod
    def _passes(record: Record, types: set[RecordType] | None, direction: Direction, period: Period | None, now: datetime) -> bool:
        if types and record.type not in types:
            return False
        start = record.starts_at
        if period is not None:
            if record.recurrence is not None:
                return bool(record.occurrences(period.start, period.end))
            return record.day is not None and period.contains(record.day)
        if direction == Direction.PAST:
            return start is not None and start < now
        if direction == Direction.FUTURE:
            if start is None:
                return True
            if record.recurrence is not None:
                return True
            return start >= now or (record.at is None and record.day >= now.date())
        return True

    @staticmethod
    def _order_key(hit: SearchHit, best: float, direction: Direction, today: date) -> tuple:
        """Сначала уровень релевантности (крупными ступенями), внутри уровня — по дате:
        для прошлого — от недавних, для будущего — от ближайших, иначе предстоящие, затем недавние."""
        relative = hit.score / best
        tier = 0 if relative >= 0.75 else 1 if relative >= 0.55 else 2
        day = hit.record.day
        if day is None:
            return (tier, 2, 0, hit.record.id or 0)
        distance = (day - today).days
        if direction == Direction.PAST:
            return (tier, 0, -distance, -(hit.record.id or 0))
        if direction == Direction.FUTURE:
            return (tier, 0, distance, hit.record.id or 0)
        return (tier, 0 if distance >= 0 else 1, abs(distance), hit.record.id or 0)
