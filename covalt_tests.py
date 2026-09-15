"""Small, readable self-checks shown on Covalt's «Наши тесты» page."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Callable

from covalt_brain import chat, ollama_status, understand_prompt


TESTS = [
    {
        "id": "identity",
        "title": "Имя и создатель",
        "description": "Covalt должен называть себя и RYNico corp.",
        "expected": "В ответе есть Covalt и RYNico corp.",
    },
    {
        "id": "text-intent",
        "title": "Понимание смысла",
        "description": "Проверяем, что промпт про волны не превращается в orb.",
        "expected": "waves / ocean / calm / motion",
    },
    {
        "id": "different-answers",
        "title": "Ответы не одинаковые",
        "description": "Приветствие и просьба составить план должны давать разные ответы.",
        "expected": "Тексты ответов отличаются",
    },
    {
        "id": "honest-unknown",
        "title": "Честное незнание",
        "description": "Без поиска Covalt не должен выдумывать фактологический ответ.",
        "expected": "Предупреждает о необходимости поиска",
    },
    {
        "id": "first-dialogue",
        "title": "Первый диалог",
        "description": "Covalt должен пережить два связанных текстовых сообщения и не дать один и тот же ответ.",
        "expected": "Два разных ответа, оба сформированы после этапов анализа",
    },
    {
        "id": "model-provider",
        "title": "Настоящая модель подключена",
        "description": "Проверяем, что Ollama доступна и модель действительно загружена.",
        "expected": "available = true и model_found = true",
    },
    {
        "id": "web-switch",
        "title": "Режим поиска",
        "description": "Флаг поиска должен включать поиск и показать статус.",
        "expected": "searched = true",
    },
]


def _result(test_id: str, passed: bool, actual: str, details: str = "") -> dict:
    definition = next(item for item in TESTS if item["id"] == test_id)
    return {
        **definition,
        "status": "PASS" if passed else "CHECK",
        "passed": passed,
        "actual": actual,
        "details": details,
    }


def run_tests() -> dict:
    results: list[dict] = []
    try:
        identity = chat("Кто тебя создал?")
        answer = identity["answer"]
        results.append(_result("identity", "Covalt" in answer and "RYNico corp." in answer, answer))
    except Exception as error:
        results.append(_result("identity", False, "Ошибка", str(error)))

    try:
        intent = understand_prompt("Синие волны спокойно движутся по океану")
        passed = all(intent[key] == expected for key, expected in {
            "scene": "waves", "palette": "ocean", "action": "motion", "mood": "calm"
        }.items())
        results.append(_result("text-intent", passed, f"{intent['scene']} / {intent['palette']} / {intent['mood']} / {intent['action']}", str(intent)))
    except Exception as error:
        results.append(_result("text-intent", False, "Ошибка", str(error)))

    try:
        hello = chat("Привет")["answer"]
        plan = chat("Составь план запуска проекта")["answer"]
        results.append(_result("different-answers", hello != plan, f"Приветствие: {hello[:120]} | План: {plan[:120]}"))
    except Exception as error:
        results.append(_result("different-answers", False, "Ошибка", str(error)))

    try:
        unknown = chat("Какая погода будет завтра на Марсе?")
        honest = any(word in unknown["answer"].lower() for word in ("не знаю", "поиск", "не буду выдумывать", "не получил"))
        results.append(_result("honest-unknown", honest, unknown["answer"]))
    except Exception as error:
        results.append(_result("honest-unknown", False, "Ошибка", str(error)))

    try:
        first = chat("Привет, я хочу проверить, понимаешь ли ты обычный текст.")
        second = chat("Составь короткий план нашего тестирования.", history=[
            {"role": "user", "content": "Привет, я хочу проверить, понимаешь ли ты обычный текст."},
            {"role": "assistant", "content": first["answer"]},
        ])
        real_model = first["model"].startswith(("ollama", "openai-compatible"))
        passed = real_model and first["answer"] != second["answer"] and "1." in second["answer"] and first["thinking_ms"] >= 700 and second["thinking_ms"] >= 700
        actual = f"model: {first['model']}; ответ 1: {first['thinking_ms']} ms; ответ 2: {second['thinking_ms']} ms"
        details = f"1) {first['answer'][:160]} | 2) {second['answer'][:160]}"
        if not real_model: details = "Настоящая модель не подключена; проверка диалога остановлена на fallback. " + details
        results.append(_result("first-dialogue", passed, actual, details))
    except Exception as error:
        results.append(_result("first-dialogue", False, "Ошибка", str(error)))

    try:
        status = ollama_status()
        actual = f"available = {status['available']}; model_found = {status['model_found']}; model = {status['model']}"
        results.append(_result("model-provider", status["available"] and status["model_found"], actual, "Если CHECK — текст пока отвечает прозрачным fallback, а не настоящей LLM."))
    except Exception as error:
        results.append(_result("model-provider", False, "Ошибка", str(error)))

    try:
        web = chat("Проверь свежие новости о Python", use_web=True)
        source_count = len(web["sources"])
        actual = f"searched = {web['searched']}; источников: {source_count}"
        details = "Поиск включён, но сеть не вернула источники — это CHECK, а не выдуманный PASS." if source_count == 0 else "Источники получены."
        results.append(_result("web-switch", web["searched"] is True and source_count > 0, actual, details))
    except Exception as error:
        results.append(_result("web-switch", False, "Ошибка", str(error)))

    passed = sum(1 for result in results if result["passed"])
    return {
        "name": "Covalt text core",
        "creator": "RYNico corp.",
        "ran_at": datetime.now(timezone.utc).isoformat(),
        "passed": passed,
        "total": len(results),
        "results": results,
        "note": "PASS означает, что проверка прошла. CHECK означает, что нужна ручная проверка, а не скрытая ошибка.",
    }
