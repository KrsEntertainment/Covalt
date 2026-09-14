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
import urllib.parse
import urllib.request
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from html.parser import HTMLParser
from pathlib import Path
from typing import Any

import numpy as np

from video_generator import TinyMotionNetwork, parse_prompt, png_bytes, render_frame

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
    action_words = {
        "летит": "flight", "лететь": "flight", "движется": "motion", "двигается": "motion",
        "вращается": "rotation", "вращаться": "rotation", "пульсирует": "pulse", "сияет": "glow",
        "waves": "motion", "вращается": "rotation", "flies": "flight", "moves": "motion",
        "spins": "rotation", "glows": "glow", "плывёт": "motion", "плывет": "motion",
    }
    action = next((value for word, value in action_words.items() if word in lower), "motion")
    moods = {
        "спокой": "calm", "мяг": "soft", "ярк": "bright", "неон": "neon", "ноч": "night",
        "тём": "dark", "темн": "dark", "warm": "warm", "calm": "calm", "neon": "neon",
    }
    mood = next((value for word, value in moods.items() if word in lower), "cinematic")
    camera = "close-up" if any(word in lower for word in ("крупный план", "close-up", "портрет")) else "wide"
    language = "ru" if re.search(r"[а-яё]", lower) else "en"
    return {
        "raw": text,
        "language": language,
        "scene": scene.kind,
        "palette": scene.palette,
        "action": action,
        "mood": mood,
        "camera": camera,
        "entities": [scene.kind, scene.palette],
        "confidence": 0.82 if scene.kind != "orb" or scene.palette != "violet" else 0.61,
    }


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


def _local_answer(message: str, intent: dict[str, Any], results: list[SearchResult]) -> str:
    lower = message.lower()
    if any(word in lower for word in ("кто ты", "твоё имя", "твое имя", "как тебя зовут", "who are you", "your name")):
        return f"Я {COVALT_NAME} — локальный AI-движок, созданный {COVALT_CREATOR}. Я умею разбирать промпты, собирать 2D-видео, изображения и искать свежую информацию в интернете."
    if any(word in lower for word in ("создатель", "кто тебя создал", "creator", "made you")):
        return f"Мой создатель — {COVALT_CREATOR} Моё имя — {COVALT_NAME}."
    if any(word in lower for word in ("что ты умеешь", "возможности", "what can you do")):
        return "Я могу: разобрать идею по смыслу, собрать локальную 2D-сцену в MP4, сделать изображение, ответить в чате и найти актуальные источники в интернете, если включить поиск."
    if results:
        return f"Я нашёл {len(results)} релевантных источника по запросу. Ниже оставил краткие выдержки и ссылки — проверьте дату публикации перед важными решениями."
    return (
        f"Я понял запрос как сцену «{intent['scene']}» с действием «{intent['action']}», "
        f"настроением «{intent['mood']}» и палитрой «{intent['palette']}». "
        "Можно попросить меня уточнить промпт, включить поиск или перейти в режим изображения/видео."
    )


def chat(message: str, history: list[dict[str, str]] | None = None, use_web: bool = False) -> dict[str, Any]:
    message = re.sub(r"\s+", " ", str(message or "")).strip()
    if not message:
        raise ValueError("Напишите сообщение")
    if len(message) > MAX_CHAT_LENGTH:
        raise ValueError(f"Сообщение слишком длинное (максимум {MAX_CHAT_LENGTH} символов)")
    intent = understand_prompt(message)
    should_search = use_web or any(word in message.lower() for word in ("найди", "источники", "новости", "сейчас", "find", "latest", "search"))
    results = search_web(message) if should_search else []
    system = (
        f"You are {COVALT_NAME}, a concise assistant created by {COVALT_CREATOR}. "
        "Be transparent: do not claim to have a large pretrained model when using local fallback. "
        "Answer in the user's language."
    )
    messages = [{"role": "system", "content": system}]
    if history:
        for item in history[-8:]:
            if item.get("role") in {"user", "assistant"} and item.get("content"):
                messages.append({"role": item["role"], "content": str(item["content"])[:2000]})
    messages.append({"role": "user", "content": message})
    answer = _remote_answer(messages) or _local_answer(message, intent, results)
    return {
        "answer": answer,
        "sources": [result.__dict__ for result in results],
        "understanding": intent,
        "searched": should_search,
        "model": "remote-compatible" if os.environ.get("COVALT_LLM_URL") else "covalt-local-intent",
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
