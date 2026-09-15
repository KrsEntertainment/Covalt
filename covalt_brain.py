"""Covalt's local assistant layer.

The video engine is the trainable MLP in ``video_generator.py``.  This module
adds the product layer around it: a stable identity, prompt understanding,
optional web search, and an optional OpenAI-compatible chat endpoint.  With no
API key it remains useful offline instead of pretending that a hidden model is
being used.
"""

from __future__ import annotations

import html
import json
import os
import re
import time
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np

from video_generator import SCENE_WORDS, TinyMotionNetwork, parse_prompt, png_bytes, render_frame

COVALT_NAME = "Covalt"
COVALT_CREATOR = "RYNico corp."
MAX_CHAT_LENGTH = 4000


@dataclass
class SearchResult:
    title: str
    url: str
    snippet: str


class _DuckParser(HTMLParser):
    """Small dependency-free parser for DuckDuckGo HTML results."""

    def __init__(self):
        super().__init__()
        self.results: list[SearchResult] = []
        self._in_title = False
        self._in_snippet = False
        self._current: dict[str, str] = {}
        self._title_depth = 0
        self._snippet_depth = 0

    def handle_starttag(self, tag, attrs):
        attributes = dict(attrs)
        classes = attributes.get("class", "")
        if tag == "a" and "result__a" in classes:
            href = attributes.get("href", "")
            if href.startswith("//"):
                href = "https:" + href
            self._current = {"url": href, "title": "", "snippet": ""}
            self._in_title = True
            self._title_depth = 1
        elif self._current and "result__snippet" in classes:
            self._in_snippet = True
            self._snippet_depth = 1
        elif self._in_title:
            self._title_depth += 1
        elif self._in_snippet:
            self._snippet_depth += 1

    def handle_endtag(self, tag):
        if self._in_title:
            self._title_depth -= 1
            if self._title_depth <= 0:
                self._in_title = False
        elif self._in_snippet:
            self._snippet_depth -= 1
            if self._snippet_depth <= 0:
                self._in_snippet = False
                if self._current.get("url") and self._current.get("title"):
                    self.results.append(SearchResult(**self._current))
                    self._current = {}

    def handle_data(self, data):
        if self._in_title:
            self._current["title"] += data
        elif self._in_snippet:
            self._current["snippet"] += data


def browser_search_links(query: str) -> list[dict[str, str]]:
    """Fallback links that the user's browser can open when the server has no network."""
    encoded = urllib.parse.quote_plus(re.sub(r"\s+", " ", query).strip()[:400])
    return [
        {"title": "Открыть Google Search", "url": f"https://www.google.com/search?q={encoded}"},
        {"title": "Открыть DuckDuckGo", "url": f"https://duckduckgo.com/?q={encoded}"},
        {"title": "Открыть Yandex", "url": f"https://yandex.ru/search/?text={encoded}"},
    ]


def search_web(query: str, limit: int = 5) -> list[SearchResult]:
    """Search the public web without an API key.

    DuckDuckGo's HTML endpoint is used only when the user explicitly enables
    the search switch.  A network error is returned to the caller as an empty
    result list, so chat still works offline.
    """
    query = re.sub(r"\s+", " ", query).strip()[:400]
    if not query:
        return []
    url = "https://html.duckduckgo.com/html/?" + urllib.parse.urlencode({"q": query})
    request = urllib.request.Request(url, headers={"User-Agent": "Covalt/1.0 (RYNico corp.)"})
    try:
        with urllib.request.urlopen(request, timeout=8) as response:
            content = response.read(450_000).decode("utf-8", "replace")
        parser = _DuckParser()
        parser.feed(content)
        results = []
        for result in parser.results:
            parsed = urllib.parse.urlparse(result.url)
            params = urllib.parse.parse_qs(parsed.query)
            if "uddg" in params:
                result.url = params["uddg"][0]
            result.title = html.unescape(re.sub(r"\s+", " ", result.title)).strip()
            result.snippet = html.unescape(re.sub(r"\s+", " ", result.snippet)).strip()
            results.append(result)
        return results[:limit]
    except (OSError, ValueError, UnicodeError):
        return []


def understand_prompt(prompt: str) -> dict[str, Any]:
    """Turn a natural-language prompt into transparent, editable intent."""
    text = re.sub(r"\s+", " ", prompt.strip())
    lower = text.lower()
    scene = parse_prompt(text)
    known_scene = any(word in lower for words, _ in SCENE_WORDS.values() for word in words.split())
    action_words = {
        "летит": "flight", "лететь": "flight", "движется": "motion", "движутся": "motion", "двигается": "motion",
        "вращается": "rotation", "вращаться": "rotation", "пульсирует": "pulse", "сияет": "glow",
        "waves": "motion", "вращается": "rotation", "flies": "flight", "moves": "motion",
        "spins": "rotation", "glows": "glow", "плывёт": "motion", "плывет": "motion",
    }
    action = next((value for word, value in action_words.items() if word in lower), "unspecified")
    moods = {
        "спокой": "calm", "мяг": "soft", "ярк": "bright", "неон": "neon", "ноч": "night",
        "тём": "dark", "темн": "dark", "warm": "warm", "calm": "calm", "neon": "neon",
    }
    mood = next((value for word, value in moods.items() if word in lower), "neutral")
    camera = "close-up" if any(word in lower for word in ("крупный план", "close-up", "портрет")) else "wide"
    language = "ru" if re.search(r"[а-яё]", lower) else "en"
    recognized_scene = scene.kind if known_scene else "text"
    recognized_palette = scene.palette if known_scene else "unspecified"
    return {
        "raw": text,
        "language": language,
        "scene": recognized_scene,
        "palette": recognized_palette,
        "action": action,
        "mood": mood,
        "camera": camera,
        "entities": [recognized_scene] if not known_scene else [scene.kind, scene.palette],
        "confidence": 0.82 if known_scene else 0.58,
    }


OLLAMA_URL = os.environ.get("COVALT_OLLAMA_URL", "http://127.0.0.1:11434/api/chat").strip()
OLLAMA_MODEL = os.environ.get("COVALT_OLLAMA_MODEL", "").strip()


def _choose_ollama_model(models: list[str]) -> str:
    """Prefer an explicitly configured model, otherwise detect DeepSeek/Qwen."""
    if OLLAMA_MODEL:
        return OLLAMA_MODEL
    preferred = ("deepseek-r1", "deepseek-v3", "qwen2.5", "qwen3", "llama3.2")
    for family in preferred:
        for name in models:
            if name == family or name.startswith(family + ":"):
                return name
    return models[0] if models else "deepseek-r1:7b"


def _ollama_request(messages: list[dict[str, str]], model: str, timeout: float = 30.0) -> str | None:
    payload = {"model": model, "messages": messages, "stream": False}
    request = urllib.request.Request(
        OLLAMA_URL,
        data=json.dumps(payload).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            result = json.loads(response.read().decode("utf-8"))
        content = result.get("message", {}).get("content")
        return str(content).strip() if content else None
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return None


def ollama_status() -> dict[str, Any]:
    """Return provider state without hiding that a real model is absent."""
    tags_url = OLLAMA_URL.rsplit("/api/chat", 1)[0] + "/api/tags"
    request = urllib.request.Request(tags_url, headers={"User-Agent": "Covalt/1.0"})
    try:
        with urllib.request.urlopen(request, timeout=1.5) as response:
            result = json.loads(response.read().decode("utf-8"))
        models = [item.get("name", "") for item in result.get("models", [])]
        selected = _choose_ollama_model(models)
        loaded = selected in models or any(name.startswith(selected + ":") for name in models)
        return {"provider": "ollama", "available": True, "model": selected, "model_found": loaded, "models": models[:20]}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError):
        return {"provider": "ollama", "available": False, "model": OLLAMA_MODEL or "deepseek-r1:7b", "model_found": False, "models": []}


def _ollama_answer(messages: list[dict[str, str]], message: str, use_web: bool) -> tuple[str | None, list[SearchResult], str]:
    """Ask Ollama, with a simple explicit SEARCH tool round-trip."""
    wants_search = use_web or any(word in message.lower() for word in ("найди", "источники", "новости", "сейчас", "find", "latest", "search"))
    status = ollama_status()
    if not status["available"] or not status["model_found"]:
        return None, [], "ollama-unavailable"
    model = status["model"]
    tool_instruction = (
        " If current facts or web sources are needed, answer only TOOL_SEARCH: followed by a short query. "
        "Otherwise answer the user directly."
    )
    first_messages = [dict(item) for item in messages]
    first_messages[0] = {**first_messages[0], "content": first_messages[0]["content"] + tool_instruction}
    draft = _ollama_request(first_messages, model)
    if not draft:
        return None, [], "ollama-unavailable"
    tool_query = message
    tool_match = re.search(r"TOOL_SEARCH\s*:\s*(.+)", draft, flags=re.IGNORECASE | re.DOTALL)
    if tool_match:
        tool_query = tool_match.group(1).strip().split("\n", 1)[0] or message
    results = search_web(tool_query) if wants_search or tool_match else []
    if not (wants_search or tool_match):
        return re.sub(r"<think>.*?</think>", "", draft, flags=re.IGNORECASE | re.DOTALL).strip(), [], "ollama"
    evidence = "\n".join(f"- {item.title}: {item.snippet} ({item.url})" for item in results)
    if not evidence:
        evidence = "NO SOURCES RECEIVED. Say clearly that the server search was unavailable; do not invent sources."
    final_messages = first_messages + [
        {"role": "assistant", "content": draft},
        {"role": "user", "content": f"Search tool output for {tool_query}:\n{evidence}\nNow answer the original user in their language and be explicit about source availability."},
    ]
    answer = _ollama_request(final_messages, model)
    cleaned = re.sub(r"<think>.*?</think>", "", answer or draft, flags=re.IGNORECASE | re.DOTALL).strip()
    return cleaned, results, "ollama-tool-search"


def _remote_answer(messages: list[dict[str, str]]) -> str | None:
    """Call an optional OpenAI-compatible endpoint when configured."""
    endpoint = os.environ.get("COVALT_LLM_URL", "").strip()
    if not endpoint:
        return None
    payload = {
        "model": os.environ.get("COVALT_LLM_MODEL", "covalt"),
        "messages": messages,
        "temperature": 0.35,
    }
    headers = {"Content-Type": "application/json"}
    key = os.environ.get("COVALT_LLM_KEY", "").strip()
    if key:
        headers["Authorization"] = f"Bearer {key}"
    request = urllib.request.Request(endpoint, data=json.dumps(payload).encode(), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=25) as response:
            result = json.loads(response.read().decode("utf-8"))
        return str(result["choices"][0]["message"]["content"]).strip()
    except (OSError, KeyError, IndexError, TypeError, ValueError, json.JSONDecodeError):
        return None


def _local_answer(message: str, intent: dict[str, Any], results: list[SearchResult], searched: bool = False, history: list[dict[str, str]] | None = None) -> str:
    lower = message.lower()
    previous_user = next((item.get("content", "") for item in reversed(history or []) if item.get("role") == "user" and item.get("content") != message), "")
    if any(word in lower for word in ("кто ты", "твоё имя", "твое имя", "как тебя зовут", "who are you", "your name")):
        return f"Я {COVALT_NAME} — локальный AI-движок, созданный компанией {COVALT_CREATOR} Могу вести текстовый диалог, уточнять идеи и честно отмечать, где мне нужен поиск."
    if any(word in lower for word in ("создатель", "кто тебя создал", "creator", "made you")):
        return f"Мой создатель — {COVALT_CREATOR} Моё имя — {COVALT_NAME}. Сейчас мы тестируем именно моё понимание текста."
    if any(word in lower for word in ("что ты умеешь", "возможности", "what can you do")):
        return "Сейчас основной режим — текст. Я могу вести диалог, разобрать смысл запроса, составить план, помочь переписать текст и искать актуальные источники. Режимы видео и изображений временно поставлены на паузу для тестирования понимания."
    if any(word in lower for word in ("привет", "здравствуй", "добрый день", "hello", "hi")):
        return "Привет. Я Covalt. Давай проверим именно текст: задай вопрос, попроси план, объяснение или редактуру. Если нужны свежие факты — включи поиск."
    if any(word in lower for word in ("спасибо", "благодарю", "thanks")):
        return "Пожалуйста. Если мой ответ неточный, напиши, что именно нужно исправить — это тоже тест."
    if previous_user and any(word in lower for word in ("подробнее", "продолжи", "а теперь", "как я сказал", "дальше")):
        return f"Я помню предыдущую тему: «{previous_user[:180]}». Продолжаю её, а не начинаю новый разговор. Уточни, какую часть раскрыть дальше."
    if results:
        return f"Я нашёл {len(results)} релевантных источника по запросу. Ниже оставил краткие выдержки и ссылки — проверьте дату публикации перед важными решениями."
    if searched:
        return "Я включил поиск на сервере, но сеть не вернула источники. Я не буду выдумывать результат — ниже можно открыть тот же запрос напрямую в браузере."
    if any(word in lower for word in ("объясни", "что такое", "расскажи про", "explain", "what is")):
        topic = re.sub(r"^(объясни|расскажи про|что такое|explain|what is)\s*", "", message, flags=re.IGNORECASE).strip(" ?") or "эту тему"
        return f"Я понял, что нужно объяснить тему «{topic}». В локальном текстовом режиме у меня нет надёжной базы фактов, поэтому я не стану придумывать объяснение. Включи «Искать в интернете» — тогда я сначала соберу источники."
    if re.search(r"(составь|сделай)\s+.{0,40}\bплан\b|план действий|make a plan", lower):
        topic = re.sub(r"(составь|сделай)(?:\s+короткий|\s+подробный)?\s+план|план действий|make a plan", "", message, flags=re.IGNORECASE).strip(" :,.!?…") or "задачи"
        return f"Для темы «{topic}» предлагаю начать так:\n1. Уточнить цель и критерий готовности.\n2. Разбить работу на маленькие шаги.\n3. Проверить результат на отдельном тесте.\n4. Зафиксировать, что нужно улучшить."
    if any(word in lower for word in ("перепиши", "улучши текст", "исправь", "rewrite", "edit")):
        source = message.split(":", 1)[1].strip() if ":" in message else message
        return f"Я могу отредактировать этот текст. Сейчас вижу исходник: «{source[:220]}». Уточни желаемый тон — деловой, короткий, дружелюбный или рекламный."
    if "?" in message or lower.startswith(("почему", "как ", "когда ", "где ", "зачем ", "можно ли")):
        return "Я вижу вопрос, но пока не знаю на него проверенного ответа. Чтобы не выдумывать факты, включи поиск в интернете или добавь контекст, на который мне можно опереться."
    if intent["scene"] != "text":
        return f"Я понял текстовый запрос и выделил сцену «{intent['scene']}», действие «{intent['action']}», настроение «{intent['mood']}» и палитру «{intent['palette']}». Если это не то, уточни объект или цель словами."
    return "Я прочитал сообщение, но пока не понял, какой текстовый результат нужен. Напиши действие: «объясни», «составь план», «перепиши», «сравни» или задай конкретный вопрос."


def chat(message: str, history: list[dict[str, str]] | None = None, use_web: bool = False) -> dict[str, Any]:
    message = re.sub(r"\s+", " ", str(message or "")).strip()
    if not message:
        raise ValueError("Напишите сообщение")
    if len(message) > MAX_CHAT_LENGTH:
        raise ValueError(f"Сообщение слишком длинное (максимум {MAX_CHAT_LENGTH} символов)")
    started = time.perf_counter()
    intent = understand_prompt(message)
    should_search = use_web or any(word in message.lower() for word in ("найди", "источники", "новости", "сейчас", "find", "latest", "search"))
    results: list[SearchResult] = []
    system = (
        f"You are {COVALT_NAME}, a concise assistant created by {COVALT_CREATOR}. "
        "Answer in the user's language. Do not invent facts or sources."
    )
    messages = [{"role": "system", "content": system}]
    if history:
        for item in history[-8:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": str(item["content"])[:2000]})
    messages.append({"role": "user", "content": message})
    answer = None
    model_name = "local-intent-fallback"
    provider_error = ""
    if os.environ.get("COVALT_LLM_URL"):
        answer = _remote_answer(messages)
        if answer:
            model_name = "openai-compatible"
        else:
            provider_error = "configured endpoint unavailable"
    if not answer:
        answer, tool_results, ollama_model = _ollama_answer(messages, message, use_web)
        if answer:
            results = tool_results
            model_name = ollama_model
        else:
            provider_error = "ollama unavailable; using transparent fallback"
    if should_search and not results:
        # The fallback still attempts the search once, so a server with
        # outbound internet starts working without changing the chat code.
        results = search_web(message)
    if not answer:
        answer = _local_answer(message, intent, results, should_search, history)
    # Give the text core a visible, bounded thinking window. This is not a
    # claim that a hidden model is running: it makes the compose step explicit
    # and leaves room for a future larger model without instant fake answers.
    target_seconds = min(2.4, 0.8 + len(message) / 700)
    elapsed = time.perf_counter() - started
    if elapsed < target_seconds:
        time.sleep(target_seconds - elapsed)
    thinking_ms = round((time.perf_counter() - started) * 1000)
    return {
        "answer": answer,
        "sources": [result.__dict__ for result in results],
        "understanding": intent,
        "searched": should_search,
        "search_status": "sources-found" if results else ("server-network-unavailable" if should_search else "not-requested"),
        "search_links": browser_search_links(message) if should_search and not results else [],
        "thinking_ms": thinking_ms,
        "thinking_stages": ["prompt parsed", "context checked", "answer composed"],
        "memory_messages": len(history or []),
        "model": model_name,
        "provider_error": provider_error,
    }


_IMAGE_NETWORK: TinyMotionNetwork | None = None


def generate_image(prompt: str, image_dir: Path) -> dict[str, Any]:
    """Create a local PNG using the same trained Covalt motion renderer."""
    global _IMAGE_NETWORK
    if _IMAGE_NETWORK is None:
        _IMAGE_NETWORK = TinyMotionNetwork(seed=2048)
    image_id = uuid.uuid4().hex[:12]
    image_dir.mkdir(parents=True, exist_ok=True)
    scene = parse_prompt(prompt)
    frame = render_frame(scene, _IMAGE_NETWORK, 0.18, width=768, height=512)
    filename = f"{image_id}.png"
    with (image_dir / filename).open("wb") as handle:
        handle.write(png_bytes(frame))
    return {
        "id": image_id,
        "filename": filename,
        "url": f"/generated-images/{filename}",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "prompt": prompt,
        "scene": scene.kind,
        "understanding": understand_prompt(prompt),
    }
