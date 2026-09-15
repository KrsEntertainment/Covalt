"""Covalt web application: prompt -> neural 2D MP4 -> moderated library."""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import re
import threading
import uuid
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path

from flask import Flask, abort, jsonify, render_template, request, send_from_directory, session

from covalt_brain import COVALT_CREATOR, COVALT_NAME, chat as covalt_chat, generate_image as covalt_generate_image, ollama_status
from covalt_tests import run_tests
from video_generator import MAX_SECONDS, encode_video


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
VIDEO_DIR = DATA_DIR / "videos"
THUMB_DIR = DATA_DIR / "thumbs"
IMAGE_DIR = DATA_DIR / "images"
CATALOG_PATH = DATA_DIR / "catalog.json"
NEWS_PATH = DATA_DIR / "news.json"
ADMIN_PASSWORD = os.environ.get("COVALT_ADMIN_PASSWORD", "9086")
MAX_PROMPT_LENGTH = 400

for directory in (DATA_DIR, VIDEO_DIR, THUMB_DIR, IMAGE_DIR):
    directory.mkdir(parents=True, exist_ok=True)

app = Flask(__name__, static_folder="static", template_folder="templates")
app.secret_key = os.environ.get("COVALT_SECRET_KEY", "covalt-local-development-key-change-me")
# Static assets are versioned in the template too, so the studio updates without
# requiring users to clear a cached JavaScript bundle after a deployment.
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0
app.config["MAX_CONTENT_LENGTH"] = 1 * 1024 * 1024

_catalog_lock = threading.RLock()
_jobs_lock = threading.RLock()
_jobs: dict[str, dict] = {}
_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="covalt-render")


def _read_catalog() -> list[dict]:
    with _catalog_lock:
        if not CATALOG_PATH.exists():
            return []
        try:
            with CATALOG_PATH.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []


def _write_catalog(items: list[dict]) -> None:
    with _catalog_lock:
        temporary = CATALOG_PATH.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
        temporary.replace(CATALOG_PATH)


def _read_news() -> list[dict]:
    with _catalog_lock:
        if not NEWS_PATH.exists():
            return []
        try:
            with NEWS_PATH.open("r", encoding="utf-8") as handle:
                value = json.load(handle)
            return value if isinstance(value, list) else []
        except (OSError, json.JSONDecodeError):
            return []


def _write_news(items: list[dict]) -> None:
    with _catalog_lock:
        temporary = NEWS_PATH.with_suffix(".tmp")
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(items, handle, ensure_ascii=False, indent=2)
        temporary.replace(NEWS_PATH)


def _public_news(item: dict) -> dict:
    return {
        "id": item.get("id"),
        "title": item.get("title", ""),
        "body": item.get("body", ""),
        "created_at": item.get("created_at"),
        "updated_at": item.get("updated_at"),
        "published": bool(item.get("published")),
    }


def _visible_news() -> list[dict]:
    items = _read_news()
    if not _is_admin():
        items = [item for item in items if item.get("published")]
    return sorted(items, key=lambda item: item.get("updated_at") or item.get("created_at", ""), reverse=True)


def _signed_token(value: str) -> str:
    secret = str(app.secret_key).encode("utf-8")
    return hmac.new(secret, value.encode("utf-8"), hashlib.sha256).hexdigest()


def _admin_token() -> str:
    # This token is returned only after the password is accepted. It makes the
    # preview resilient in iframe/proxy environments that do not retain a
    # Flask cookie, while the normal session remains the primary auth method.
    return _signed_token("covalt-admin")


def _has_admin_access() -> bool:
    supplied = request.headers.get("X-Covalt-Admin", "")
    return bool(session.get("admin")) or hmac.compare_digest(supplied, _admin_token())


def _is_admin() -> bool:
    return _has_admin_access()


def _preview_token(video_id: str) -> str:
    return _signed_token(f"covalt-preview:{video_id}")


def _has_preview_access(video_id: str) -> bool:
    supplied = request.args.get("preview", "")
    return hmac.compare_digest(supplied, _preview_token(video_id))


def _admin_required(function):
    @wraps(function)
    def wrapped(*args, **kwargs):
        if not _has_admin_access():
            return jsonify({"error": "Нужен вход администратора"}), 401
        return function(*args, **kwargs)

    return wrapped


def _safe_title(title: str, prompt: str) -> str:
    title = re.sub(r"\s+", " ", str(title or "")).strip()
    return title[:90] or prompt[:70].strip() or "Новая сцена Covalt"


def _find_video(video_id: str) -> dict | None:
    return next((item for item in _read_catalog() if item.get("id") == video_id), None)


def _public_item(item: dict) -> dict:
    # Never expose server paths or internal job details to the browser. Draft
    # URLs carry a signed preview ticket because some hosted previews do not
    # forward Flask cookies to <video> and <img> requests.
    item_id = item.get("id")
    preview = "" if item.get("published") else f"?preview={_preview_token(item_id)}"
    return {
        "id": item_id,
        "title": item.get("title"),
        "prompt": item.get("prompt"),
        "duration": item.get("duration"),
        "created_at": item.get("created_at"),
        "published": bool(item.get("published")),
        "scene": item.get("scene"),
        "palette": item.get("palette"),
        "video_url": f"/media/{item_id}{preview}" if item.get("filename") else None,
        "download_url": f"/download/{item_id}{preview}",
        "thumbnail_url": f"/thumbs/{item_id}{preview}" if item.get("thumbnail") else None,
    }


def _visible_catalog() -> list[dict]:
    items = _read_catalog()
    if not _is_admin():
        items = [item for item in items if item.get("published")]
    return sorted(items, key=lambda item: item.get("created_at", ""), reverse=True)


def _set_job(job_id: str, **changes) -> None:
    with _jobs_lock:
        if job_id in _jobs:
            _jobs[job_id].update(changes)


def _run_generation(job_id: str, prompt: str, title: str, duration: float) -> None:
    video_id = uuid.uuid4().hex[:12]
    filename = f"{video_id}.mp4"
    thumbnail = f"{video_id}.png"
    video_path = VIDEO_DIR / filename
    thumb_path = THUMB_DIR / thumbnail
    _set_job(job_id, status="rendering", progress=0.02, message="Нейросеть строит движение…")
    try:
        details = encode_video(
            prompt,
            duration,
            str(video_path),
            str(thumb_path),
            progress=lambda value: _set_job(job_id, progress=0.02 + value * 0.94, message="Собираем кадры и MP4…"),
        )
        now = datetime.now(timezone.utc).isoformat()
        item = {
            "id": video_id,
            "title": title,
            "prompt": prompt,
            "duration": round(duration, 2),
            "created_at": now,
            "filename": filename,
            "thumbnail": thumbnail,
            "published": False,
            **details,
        }
        with _catalog_lock:
            catalog = _read_catalog()
            catalog.append(item)
            _write_catalog(catalog)
        _set_job(job_id, status="done", progress=1.0, message="Видео готово. Администратор может опубликовать его.", video=_public_item(item))
    except Exception as error:  # surface a friendly error in the UI, keep server alive
        for path in (video_path, thumb_path):
            try:
                path.unlink(missing_ok=True)
            except OSError:
                pass
        _set_job(job_id, status="error", progress=0, message=str(error))


@app.get("/tests")
def tests_page():
    return render_template("tests.html", is_admin=_is_admin(), creator_name=COVALT_CREATOR, covalt_name=COVALT_NAME)


@app.get("/api/tests")
def tests_api():
    return jsonify(run_tests())


@app.get("/")
def index():
    visible = [_public_item(item) for item in _visible_catalog()]
    drafts = [item for item in visible if not item.get("published")]
    return render_template(
        "index.html",
        videos=visible,
        draft_videos=drafts,
        news=[_public_news(item) for item in _visible_news()],
        creator_name=COVALT_CREATOR,
        covalt_name=COVALT_NAME,
        is_admin=_is_admin(),
    )


@app.get("/api/videos")
def videos():
    return jsonify({"videos": [_public_item(item) for item in _visible_catalog()], "admin": _is_admin()})


@app.get("/api/news")
def news_list():
    return jsonify({"news": [_public_news(item) for item in _visible_news()], "admin": _is_admin()})


@app.post("/api/news")
@_admin_required
def create_news():
    payload = request.get_json(silent=True) or {}
    title = re.sub(r"\s+", " ", str(payload.get("title", ""))).strip()
    body = str(payload.get("body", "")).strip()
    if not title or not body:
        return jsonify({"error": "Укажите заголовок и текст новости"}), 400
    if len(title) > 120 or len(body) > 5000:
        return jsonify({"error": "Заголовок или текст новости слишком длинные"}), 400
    now = datetime.now(timezone.utc).isoformat()
    item = {
        "id": uuid.uuid4().hex[:12],
        "title": title,
        "body": body,
        "created_at": now,
        "updated_at": now,
        "published": bool(payload.get("published", True)),
    }
    with _catalog_lock:
        items = _read_news()
        items.append(item)
        _write_news(items)
    return jsonify({"ok": True, "news": _public_news(item)})


@app.put("/api/news/<news_id>")
@_admin_required
def update_news(news_id: str):
    payload = request.get_json(silent=True) or {}
    title = re.sub(r"\s+", " ", str(payload.get("title", ""))).strip()
    body = str(payload.get("body", "")).strip()
    if not title or not body:
        return jsonify({"error": "Укажите заголовок и текст новости"}), 400
    with _catalog_lock:
        items = _read_news()
        item = next((entry for entry in items if entry.get("id") == news_id), None)
        if item is None:
            return jsonify({"error": "Новость не найдена"}), 404
        item.update({
            "title": title[:120],
            "body": body[:5000],
            "published": bool(payload.get("published", item.get("published", True))),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        })
        _write_news(items)
    return jsonify({"ok": True, "news": _public_news(item)})


@app.delete("/api/news/<news_id>")
@_admin_required
def delete_news(news_id: str):
    with _catalog_lock:
        items = _read_news()
        if not any(entry.get("id") == news_id for entry in items):
            return jsonify({"error": "Новость не найдена"}), 404
        _write_news([entry for entry in items if entry.get("id") != news_id])
    return jsonify({"ok": True})


@app.get("/api/model/status")
def model_status_api():
    return jsonify(ollama_status())


@app.post("/api/chat")
def chat_api():
    payload = request.get_json(silent=True) or {}
    try:
        result = covalt_chat(payload.get("message", ""), payload.get("history", []), bool(payload.get("use_web")))
    except ValueError as error:
        return jsonify({"error": str(error)}), 400
    except Exception:
        return jsonify({"error": "Covalt не смог обработать запрос. Попробуйте ещё раз."}), 502
    return jsonify(result)


@app.post("/api/prompt/understand")
def prompt_understanding():
    from covalt_brain import understand_prompt
    payload = request.get_json(silent=True) or {}
    return jsonify(understand_prompt(str(payload.get("prompt", ""))))


@app.post("/api/image")
def image_generate():
    payload = request.get_json(silent=True) or {}
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", ""))).strip()
    if not prompt:
        return jsonify({"error": "Опишите изображение"}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Описание слишком длинное (максимум {MAX_PROMPT_LENGTH} символов)"}), 400
    try:
        return jsonify(covalt_generate_image(prompt, IMAGE_DIR))
    except Exception as error:
        return jsonify({"error": f"Не удалось создать изображение: {error}"}), 500


@app.post("/api/login")
def login():
    payload = request.get_json(silent=True) or {}
    password = str(payload.get("password", ""))
    if hmac.compare_digest(password, ADMIN_PASSWORD):
        session["admin"] = True
        # Return the queue together with the login response so the admin screen
        # can render immediately, even when a preview browser has a stale page.
        return jsonify({
            "ok": True,
            "admin": True,
            "admin_token": _admin_token(),
            "videos": [_public_item(item) for item in _visible_catalog()],
        })
    return jsonify({"ok": False, "error": "Неверный пароль"}), 401


@app.post("/api/logout")
def logout():
    session.pop("admin", None)
    return jsonify({"ok": True})


@app.post("/api/generate")
def generate():
    payload = request.get_json(silent=True) or {}
    prompt = re.sub(r"\s+", " ", str(payload.get("prompt", ""))).strip()
    if not prompt:
        return jsonify({"error": "Напишите описание сцены"}), 400
    if len(prompt) > MAX_PROMPT_LENGTH:
        return jsonify({"error": f"Описание слишком длинное (максимум {MAX_PROMPT_LENGTH} символов)"}), 400
    try:
        duration = float(payload.get("duration", 8))
    except (TypeError, ValueError):
        return jsonify({"error": "Длительность должна быть числом"}), 400
    if not 1 <= duration <= MAX_SECONDS:
        return jsonify({"error": "Длительность должна быть от 1 до 300 секунд"}), 400
    title = _safe_title(payload.get("title", ""), prompt)
    job_id = uuid.uuid4().hex
    with _jobs_lock:
        _jobs[job_id] = {"id": job_id, "status": "queued", "progress": 0, "message": "Задача в очереди…"}
    _executor.submit(_run_generation, job_id, prompt, title, duration)
    return jsonify({"job_id": job_id})


@app.get("/api/jobs/<job_id>")
def job_status(job_id: str):
    with _jobs_lock:
        job = dict(_jobs.get(job_id, {}))
    if not job:
        return jsonify({"error": "Задача не найдена"}), 404
    return jsonify(job)


@app.post("/api/videos/<video_id>/publish")
@_admin_required
def publish(video_id: str):
    with _catalog_lock:
        catalog = _read_catalog()
        item = next((item for item in catalog if item.get("id") == video_id), None)
        if item is None:
            return jsonify({"error": "Видео не найдено"}), 404
        item["published"] = True
        _write_catalog(catalog)
    return jsonify({"ok": True, "video": _public_item(item)})


@app.delete("/api/videos/<video_id>")
@_admin_required
def delete_video(video_id: str):
    with _catalog_lock:
        catalog = _read_catalog()
        item = next((item for item in catalog if item.get("id") == video_id), None)
        if item is None:
            return jsonify({"error": "Видео не найдено"}), 404
        _write_catalog([entry for entry in catalog if entry.get("id") != video_id])
    for directory, filename in ((VIDEO_DIR, item.get("filename")), (THUMB_DIR, item.get("thumbnail"))):
        if filename:
            try:
                (directory / filename).unlink(missing_ok=True)
            except OSError:
                pass
    return jsonify({"ok": True})


def _catalog_file(video_id: str, field: str, directory: Path):
    item = _find_video(video_id)
    if not item or (not item.get("published") and not (_has_admin_access() or _has_preview_access(video_id))):
        abort(404)
    filename = item.get(field)
    if not filename or Path(filename).name != filename:
        abort(404)
    return send_from_directory(directory, filename, conditional=True)


@app.get("/media/<video_id>")
def media(video_id: str):
    # The URL uses an id rather than accepting arbitrary filesystem paths.
    item = _find_video(video_id)
    if not item or (not item.get("published") and not (_has_admin_access() or _has_preview_access(video_id))):
        abort(404)
    return send_from_directory(VIDEO_DIR, item["filename"], conditional=True)


@app.get("/download/<video_id>")
def download(video_id: str):
    item = _find_video(video_id)
    if not item or (not item.get("published") and not (_has_admin_access() or _has_preview_access(video_id))):
        abort(404)
    return send_from_directory(VIDEO_DIR, item["filename"], as_attachment=True, download_name=f"{item.get('title', 'covalt-video')}.mp4", conditional=True)


@app.get("/thumbs/<video_id>")
def thumbs(video_id: str):
    return _catalog_file(video_id, "thumbnail", THUMB_DIR)


@app.get("/generated-images/<filename>")
def generated_image(filename: str):
    if Path(filename).name != filename or not filename.endswith(".png"):
        abort(404)
    return send_from_directory(IMAGE_DIR, filename, conditional=True)


@app.get("/health")
def health():
    return jsonify({"ok": True, "service": COVALT_NAME, "creator": COVALT_CREATOR, "admin": _is_admin()})


if __name__ == "__main__":
    app.run(host=os.environ.get("COVALT_HOST", "0.0.0.0"), port=int(os.environ.get("PORT", "5000")), debug=False)
