"""Сравнение разборщиков реплик на наборе фраз eval/nlu_cases.jsonl.

    python -m eval.run_nlu_eval --backend rules --backend local:qwen3-4b-2507 --backend hybrid:qwen3-4b-2507

rules — только правила; local:<модель> — только LLM; hybrid:<модель> — как в приложении:
LLM, а если она не поняла команду или недоступна — правила.

Для каждой фразы проверяются только указанные в ней поля (намерение, дата, время,
период, ключевые слова названия или запроса и т. д.). «Сегодня» — вторник,
13 октября 2026, 12:00, как в тестах.
"""

from __future__ import annotations

import argparse
import json
import statistics
import time
from datetime import date, datetime, time as dtime, timedelta
from pathlib import Path
from typing import Callable

from app.core.commands import ParsedCommand
from app.nlu.llm_prompt import LLMContext
from app.nlu.rules import ParseContext, RuleBasedParser
from app.search.morph import lemmas

NOW = datetime(2026, 10, 13, 12, 0)
CASES = Path(__file__).with_name("nlu_cases.jsonl")
HOLDOUT = Path(__file__).with_name("nlu_holdout.jsonl")


def load_cases(path: Path = CASES) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def make_backend(spec: str) -> tuple[Callable[[str], ParsedCommand], Callable[[], None]]:
    if spec == "rules":
        parser = RuleBasedParser()
        return (lambda text: parser.parse(text, ParseContext(now=NOW))), (lambda: None)
    if spec.startswith(("local:", "hybrid:")):
        from app.nlu.local_llm import LocalLLMParser
        from app.nlu.service import NLU

        parser = LocalLLMParser(spec.split(":", 1)[1])
        if spec.startswith("hybrid:"):
            nlu = NLU(RuleBasedParser(), parser)
            return (lambda text: nlu.parse(text, now=NOW)), parser.warm_up
        return (lambda text: parser.parse(text, LLMContext(now=NOW, timezone="Europe/Moscow"))), parser.warm_up
    raise SystemExit(f"Неизвестный разборщик {spec!r}")


def check(case: dict, cmd: ParsedCommand) -> list[str]:
    errors = []
    if "intent" in case and cmd.intent.value != case["intent"]:
        errors.append(f"intent={cmd.intent.value}")
    if "intent_any" in case and cmd.intent.value not in case["intent_any"]:
        errors.append(f"intent={cmd.intent.value}")
    if "day" in case:
        expected = date.fromisoformat(case["day"]) if case["day"] else None
        day = cmd.day
        if day is None and cmd.at is not None:
            # Время без даты ассистент ставит на ближайший момент — так же считаем и здесь.
            day = NOW.date() if datetime.combine(NOW.date(), cmd.at) > NOW else NOW.date() + timedelta(days=1)
        if day != expected:
            errors.append(f"day={cmd.day}")
    if "at" in case:
        expected = dtime.fromisoformat(case["at"]) if case["at"] else None
        if cmd.at != expected:
            errors.append(f"at={cmd.at}")
    if "period" in case:
        got = [cmd.period.start.isoformat(), cmd.period.end.isoformat()] if cmd.period else None
        if got != case["period"]:
            errors.append(f"period={got}")
    if "types" in case and sorted(t.value for t in cmd.record_types) != sorted(case["types"]):
        errors.append(f"types={sorted(t.value for t in cmd.record_types)}")
    if case.get("deadline") and not cmd.is_deadline:
        errors.append("deadline=False")
    if "recurrence" in case and (cmd.recurrence.value if cmd.recurrence else None) != case["recurrence"]:
        errors.append(f"recurrence={cmd.recurrence}")
    if "shift" in case and cmd.shift_minutes != case["shift"]:
        errors.append(f"shift={cmd.shift_minutes}")
    if case.get("refers") and not cmd.refers_to_previous:
        errors.append("refers=False")
    if "direction" in case and cmd.direction.value != case["direction"]:
        errors.append(f"direction={cmd.direction.value}")
    if "title_has" in case:
        got = set(lemmas(cmd.title or ""))
        if not set(case["title_has"]) <= got:
            errors.append(f"title={cmd.title!r}")
    if "query_has" in case:
        got = set(lemmas(" ".join([cmd.query or "", *cmd.keywords])))
        if not set(case["query_has"]) <= got:
            errors.append(f"query={cmd.query!r}")
    return errors


def evaluate(spec: str, verbose: bool, path: Path = CASES) -> dict:
    parse, warm_up = make_backend(spec)
    warm_up()
    cases = load_cases(path)
    latencies, failures, intent_ok = [], [], 0
    for case in cases:
        started = time.perf_counter()
        cmd = parse(case["text"])
        latencies.append(time.perf_counter() - started)
        errors = check(case, cmd)
        if not any(e.startswith("intent=") for e in errors):
            intent_ok += 1
        if errors:
            failures.append((case["text"], errors))
    result = {
        "backend": spec,
        "cases": len(cases),
        "correct": len(cases) - len(failures),
        "intent_accuracy": intent_ok / len(cases),
        "accuracy": (len(cases) - len(failures)) / len(cases),
        "latency_mean": statistics.mean(latencies),
        "latency_p90": sorted(latencies)[int(len(latencies) * 0.9) - 1],
        "failures": failures,
    }
    if verbose:
        for text, errors in failures:
            print(f"  ✗ {text!r}: {', '.join(errors)}")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--backend", action="append", default=[], help="rules или local:<модель>")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--save", type=Path, help="сохранить результаты в JSON")
    parser.add_argument("--holdout", action="store_true",
                        help="отложенный набор фраз, на котором ничего не настраивалось")
    args = parser.parse_args()
    results = []
    for spec in args.backend or ["rules"]:
        print(f"== {spec}")
        results.append(evaluate(spec, args.verbose, HOLDOUT if args.holdout else CASES))
    print("\n| Разборщик | Верно целиком | Намерение | Задержка, сред. | p90 |")
    print("|---|---|---|---|---|")
    for r in results:
        print(f"| {r['backend']} | {r['correct']}/{r['cases']} ({r['accuracy']:.0%}) | {r['intent_accuracy']:.0%} "
              f"| {r['latency_mean']:.2f} с | {r['latency_p90']:.2f} с |")
    if args.save:
        args.save.write_text(json.dumps(results, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
