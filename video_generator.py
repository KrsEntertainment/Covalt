"""Covalt's small, self-contained neural 2D video engine.

This module deliberately does not use a text-to-video API or downloaded model.
A tiny MLP is trained in NumPy when the process starts.  It learns smooth
trajectories for a handful of 2D scene primitives.  A prompt selects a scene,
the network predicts the position/scale/rotation for every frame, and a
software rasterizer turns the result into RGB frames.  FFmpeg then packages
those frames into a normal H.264 MP4.

It is not pretending to be a large cinematic diffusion model.  It is a real,
local neural generator for short, stylised 2D motion graphics and can render
up to five minutes without a cloud service.
"""

from __future__ import annotations

import hashlib
import math
import os
import re
import struct
import subprocess
import zlib
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np

try:
    import imageio_ffmpeg
except ImportError:  # A clear error is better than a mysterious import failure.
    imageio_ffmpeg = None


FPS = 12
WIDTH = 480
HEIGHT = 270
MAX_SECONDS = 300
SCENE_NAMES = ("orb", "square", "triangle", "star", "sunset", "waves", "city", "rings")


class TinyMotionNetwork:
    """A tiny MLP trained from scratch on analytic motion examples.

    The targets are generated at runtime.  The scene one-hot vector and time
    Fourier features go through two dense ReLU layers.  The five outputs are
    x, y, scale, angle and pulse.  Keeping this network small makes first run
    friendly on a CPU while still making the neural part observable and real.
    """

    def __init__(self, seed: int = 2026):
        self.rng = np.random.default_rng(seed)
        self.input_dim = len(SCENE_NAMES) + 6
        self.hidden1 = 48
        self.hidden2 = 32
        self.output_dim = 5
        self.w1 = self.rng.normal(0, math.sqrt(2 / self.input_dim), (self.input_dim, self.hidden1)).astype(np.float32)
        self.b1 = np.zeros(self.hidden1, dtype=np.float32)
        self.w2 = self.rng.normal(0, math.sqrt(2 / self.hidden1), (self.hidden1, self.hidden2)).astype(np.float32)
        self.b2 = np.zeros(self.hidden2, dtype=np.float32)
        self.w3 = self.rng.normal(0, math.sqrt(2 / self.hidden2), (self.hidden2, self.output_dim)).astype(np.float32)
        self.b3 = np.zeros(self.output_dim, dtype=np.float32)
        self._train()

    @staticmethod
    def _features(scene_ids: np.ndarray, t: np.ndarray) -> np.ndarray:
        scene = np.zeros((len(t), len(SCENE_NAMES)), dtype=np.float32)
        scene[np.arange(len(t)), scene_ids] = 1.0
        t = t.astype(np.float32)
        time = np.column_stack(
            [
                t,
                np.sin(2 * np.pi * t),
                np.cos(2 * np.pi * t),
                np.sin(4 * np.pi * t),
                np.cos(4 * np.pi * t),
                np.ones_like(t),
            ]
        ).astype(np.float32)
        return np.concatenate([scene, time], axis=1)

    @staticmethod
    def _target(scene_ids: np.ndarray, t: np.ndarray) -> np.ndarray:
        """Smooth target trajectories, one slightly different per scene."""
        phase = scene_ids.astype(np.float32) * 0.37
        speed = 0.72 + (scene_ids % 3) * 0.12
        x = 0.5 + 0.29 * np.sin(2 * np.pi * (speed * t) + phase)
        y = 0.5 + 0.21 * np.cos(2 * np.pi * (0.55 * t) + phase * 1.7)
        scale = 0.82 + 0.16 * np.sin(2 * np.pi * (1.35 * t) + phase)
        angle = 0.5 + 0.5 * np.sin(2 * np.pi * (0.65 * t) + phase)
        pulse = 0.5 + 0.5 * np.sin(2 * np.pi * (2.0 * t) + phase)
        return np.column_stack([x, y, scale, angle, pulse]).astype(np.float32)

    def _forward(self, x: np.ndarray):
        z1 = x @ self.w1 + self.b1
        a1 = np.maximum(z1, 0)
        z2 = a1 @ self.w2 + self.b2
        a2 = np.maximum(z2, 0)
        out = 1 / (1 + np.exp(-(a2 @ self.w3 + self.b3)))
        return out, (x, z1, a1, z2, a2)

    def _train(self) -> None:
        # A full-batch Adam fit is deterministic and takes well under a second
        # on an ordinary laptop.  This is intentionally not a pre-trained file.
        n = 1024
        scene_ids = self.rng.integers(0, len(SCENE_NAMES), n)
        t = self.rng.random(n).astype(np.float32)
        x = self._features(scene_ids, t)
        y = self._target(scene_ids, t)
        m = [np.zeros_like(p) for p in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)]
        v = [np.zeros_like(p) for p in (self.w1, self.b1, self.w2, self.b2, self.w3, self.b3)]
        params = [self.w1, self.b1, self.w2, self.b2, self.w3, self.b3]
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        for step in range(1, 181):
            pred, (xx, z1, a1, z2, a2) = self._forward(x)
            d = (pred - y) * pred * (1 - pred) * (2 / n)
            gw3 = a2.T @ d
            gb3 = d.sum(axis=0)
            da2 = d @ self.w3.T
            dz2 = da2 * (z2 > 0)
            gw2 = a1.T @ dz2
            gb2 = dz2.sum(axis=0)
            da1 = dz2 @ self.w2.T
            dz1 = da1 * (z1 > 0)
            gw1 = xx.T @ dz1
            gb1 = dz1.sum(axis=0)
            grads = [gw1, gb1, gw2, gb2, gw3, gb3]
            lr = 0.018 * (1 - step / 230) + 0.001
            for i, (p, g) in enumerate(zip(params, grads)):
                m[i] = beta1 * m[i] + (1 - beta1) * g
                v[i] = beta2 * v[i] + (1 - beta2) * (g * g)
                mh = m[i] / (1 - beta1**step)
                vh = v[i] / (1 - beta2**step)
                p -= lr * mh / (np.sqrt(vh) + eps)

    def predict(self, scene: str, t: float) -> np.ndarray:
        scene_id = SCENE_NAMES.index(scene)
        x = self._features(np.array([scene_id]), np.array([t], dtype=np.float32))
        return self._forward(x)[0][0]


@dataclass
class Scene:
    prompt: str
    kind: str
    palette: str
    seed: int


# Russian and English words intentionally share the same compact scene parser.
SCENE_WORDS = {
    "orb": ("шар мяч круг orb ball planet пузырь bubble", "orb"),
    "square": ("квадрат square cube блок block", "square"),
    "triangle": ("треугольник triangle", "triangle"),
    "star": ("звезда star sparkle искра", "star"),
    "sunset": ("закат sunset солнце sun horizon горы mountains", "sunset"),
    "waves": ("волны wave ocean море sea вода water", "waves"),
    "city": ("город city дома buildings skyline", "city"),
    "rings": ("кольца ring rings портал portal", "rings"),
}
PALETTE_WORDS = {
    "violet": "фиолетовый purple violet космос space neon неон",
    "sunset": "оранжевый orange закат sunset теплый warm",
    "ocean": "синий blue ocean море морской cyan бирюзовый",
    "forest": "зеленый green лес forest",
    "rose": "розовый pink rose красный red",
}


def parse_prompt(prompt: str) -> Scene:
    text = prompt.strip().lower()
    scores = {name: 0 for name in SCENE_NAMES}
    for name, (words, _) in SCENE_WORDS.items():
        scores[name] = sum(1 for word in words.split() if word in text)
    # Default to an orb: even an abstract prompt immediately produces a nice scene.
    kind = max(scores, key=scores.get) if max(scores.values()) else "orb"
    palettes = {name: sum(1 for word in words.split() if word in text) for name, words in PALETTE_WORDS.items()}
    palette = max(palettes, key=palettes.get) if max(palettes.values()) else {
        "sunset": "sunset", "waves": "ocean", "city": "violet", "rings": "violet"
    }.get(kind, "violet")
    seed = int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)
    return Scene(prompt=prompt, kind=kind, palette=palette, seed=seed)


def _blend(frame: np.ndarray, mask: np.ndarray, color: tuple[float, float, float], alpha=1.0) -> None:
    a = np.clip(mask.astype(np.float32) * alpha, 0, 1)[..., None]
    c = np.asarray(color, dtype=np.float32)[None, None, :]
    frame[:] = frame * (1 - a) + c * a


def _circle(frame, cx, cy, radius, color, alpha=1.0):
    h, w, _ = frame.shape
    yy, xx = np.ogrid[:h, :w]
    mask = ((xx - cx) ** 2 + (yy - cy) ** 2) <= radius * radius
    _blend(frame, mask, color, alpha)


def _rotated_box(frame, cx, cy, radius, angle, color):
    h, w, _ = frame.shape
    yy, xx = np.ogrid[:h, :w]
    ca, sa = math.cos(angle), math.sin(angle)
    dx, dy = xx - cx, yy - cy
    local_x, local_y = dx * ca + dy * sa, -dx * sa + dy * ca
    mask = (np.abs(local_x) <= radius) & (np.abs(local_y) <= radius)
    _blend(frame, mask, color)


def _polygon_mask(h, w, points):
    yy, xx = np.mgrid[:h, :w]
    inside = np.zeros((h, w), dtype=bool)
    # Even-odd polygon rasterization, adequate for the simple large shapes here.
    x0, y0 = points[-1]
    for x1, y1 in points:
        crosses = ((y1 > yy) != (y0 > yy)) & (xx < (x0 - x1) * (yy - y1) / ((y0 - y1) + 1e-6) + x1)
        inside ^= crosses
        x0, y0 = x1, y1
    return inside


def _triangle(frame, cx, cy, radius, angle, color):
    pts = []
    for i in range(3):
        a = angle - math.pi / 2 + i * 2 * math.pi / 3
        pts.append((cx + math.cos(a) * radius, cy + math.sin(a) * radius))
    _blend(frame, _polygon_mask(frame.shape[0], frame.shape[1], pts), color)


def _star(frame, cx, cy, radius, angle, color):
    pts = []
    for i in range(10):
        a = angle - math.pi / 2 + i * math.pi / 5
        r = radius if i % 2 == 0 else radius * 0.43
        pts.append((cx + math.cos(a) * r, cy + math.sin(a) * r))
    _blend(frame, _polygon_mask(frame.shape[0], frame.shape[1], pts), color)


def _line(frame, x, y, color, width=2):
    h, w, _ = frame.shape
    for offset in range(-width, width + 1):
        xi = np.clip(np.rint(x + offset).astype(int), 0, w - 1)
        yi = np.clip(np.rint(y + offset).astype(int), 0, h - 1)
        frame[yi, xi] = np.asarray(color, dtype=np.float32)


def _background(scene: Scene, width: int, height: int, t: float) -> np.ndarray:
    yy, xx = np.mgrid[:height, :width]
    v = yy / max(1, height - 1)
    if scene.palette == "sunset":
        top, bottom = np.array([0.08, 0.04, 0.20]), np.array([0.96, 0.25, 0.12])
    elif scene.palette == "ocean":
        top, bottom = np.array([0.02, 0.08, 0.22]), np.array([0.02, 0.55, 0.62])
    elif scene.palette == "forest":
        top, bottom = np.array([0.02, 0.08, 0.08]), np.array([0.10, 0.42, 0.24])
    elif scene.palette == "rose":
        top, bottom = np.array([0.18, 0.02, 0.15]), np.array([0.74, 0.08, 0.32])
    else:
        top, bottom = np.array([0.025, 0.02, 0.10]), np.array([0.18, 0.04, 0.32])
    frame = top[None, None, :] * (1 - v[..., None]) + bottom[None, None, :] * v[..., None]
    # A very subtle moving light makes even a still scene feel alive.
    light = np.exp(-(((xx / width - (0.5 + 0.12 * math.sin(t * 2 * math.pi))) ** 2) + (v - 0.45) ** 2) / 0.16)
    frame = np.clip(frame + light[..., None] * np.array([0.08, 0.03, 0.11]), 0, 1)
    return (frame * 255).astype(np.float32) / 255.0


def render_frame(scene: Scene, network: TinyMotionNetwork, t: float, width=WIDTH, height=HEIGHT) -> np.ndarray:
    """Render one RGB uint8 frame from the neural trajectory."""
    frame = _background(scene, width, height, t)
    rng = np.random.default_rng(scene.seed)
    # Stable stars/particles (the seed prevents them from flickering between frames).
    stars = rng.random((42, 3))
    if scene.kind not in ("sunset", "waves", "city"):
        for sx, sy, brightness in stars:
            x, y = int(sx * width), int(sy * height * 0.72)
            r = 1 if brightness < 0.85 else 2
            frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] = np.clip(
                frame[max(0, y - r):y + r + 1, max(0, x - r):x + r + 1] + brightness * 0.35, 0, 1
            )

    motion = network.predict(scene.kind, t)
    cx, cy = float(motion[0] * width), float(motion[1] * height)
    radius = max(10.0, float(motion[2]) * min(width, height) * 0.23)
    angle = float(motion[3] * math.pi * 2)
    pulse = float(motion[4])
    if scene.palette == "ocean":
        color = (0.16, 0.92, 1.0)
    elif scene.palette == "sunset":
        color = (1.0, 0.82, 0.22)
    elif scene.palette == "forest":
        color = (0.30, 1.0, 0.52)
    elif scene.palette == "rose":
        color = (1.0, 0.25, 0.60)
    else:
        color = (0.55, 0.35 + pulse * 0.25, 1.0)

    if scene.kind == "sunset":
        _circle(frame, width * 0.72, height * 0.38, height * 0.18 + pulse * 3, (1.0, 0.62, 0.18))
        # Two simple mountain silhouettes.
        _blend(frame, _polygon_mask(height, width, [(0, height), (width * .15, height * .54), (width * .30, height * .78), (width * .50, height * .45), (width * .76, height * .76), (width, height * .55), (width, height)]), (0.12, 0.04, 0.18))
    elif scene.kind == "waves":
        for row in range(4):
            base = height * (0.50 + row * 0.12)
            xs = np.arange(width)
            ys = base + np.sin(xs / 30 + t * 7 + row) * (5 + row * 2)
            _line(frame, xs, ys, (0.42, 0.96, 1.0), 1)
    elif scene.kind == "city":
        for i in range(9):
            bx = i * width / 8 - 10
            bh = height * (0.19 + ((scene.seed >> (i % 16)) & 7) / 28)
            frame[int(height - bh):height, max(0, int(bx)):min(width, int(bx + width / 10))] = (0.055, 0.035, 0.14)
            for wy in np.arange(height - bh + 10, height - 5, 14):
                wx = int(bx + 8 + ((i * 11) % 12))
                if 0 <= wx < width:
                    frame[int(wy):int(wy + 3), wx:min(width, wx + 4)] = (1.0, 0.55, 0.22)

    # Soft glow under every main object.
    _circle(frame, cx, cy, radius * (1.7 + pulse * 0.25), color, 0.10)
    if scene.kind == "square":
        _rotated_box(frame, cx, cy, radius, angle, color)
    elif scene.kind == "triangle":
        _triangle(frame, cx, cy, radius * 1.22, angle, color)
    elif scene.kind == "star":
        _star(frame, cx, cy, radius * 1.25, angle, color)
    elif scene.kind == "rings":
        for ring in (0.65, 0.88, 1.1):
            _circle(frame, cx, cy, radius * ring, color, 0.18)
            _circle(frame, cx, cy, radius * ring * 0.72, (0.04, 0.02, 0.12), 0.9)
    else:
        _circle(frame, cx, cy, radius, color)
        _circle(frame, cx - radius * .28, cy - radius * .30, radius * .18, (1, 1, 1), 0.65)
    # A thin highlight that moves with the network's rotation.
    hx = cx + math.cos(angle - 0.7) * radius * 0.75
    hy = cy + math.sin(angle - 0.7) * radius * 0.75
    _circle(frame, hx, hy, max(2, radius * 0.08), (1, 1, 1), 0.55)
    return np.clip(frame * 255, 0, 255).astype(np.uint8)


def png_bytes(rgb: np.ndarray) -> bytes:
    """Encode an RGB array as a PNG without Pillow."""
    h, w, _ = rgb.shape
    raw = b"".join(b"\x00" + rgb[y].tobytes() for y in range(h))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)

    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 2, 0, 0, 0)) + chunk(b"IDAT", zlib.compress(raw, 6)) + chunk(b"IEND", b"")


def encode_video(
    prompt: str,
    duration: float,
    video_path: str,
    thumbnail_path: str,
    progress: Optional[Callable[[float], None]] = None,
    width: int = WIDTH,
    height: int = HEIGHT,
) -> dict:
    """Generate and encode an MP4, returning renderer metadata."""
    if imageio_ffmpeg is None:
        raise RuntimeError("Не найден imageio-ffmpeg. Установите зависимости из requirements.txt")
    duration = max(1.0, min(float(duration), MAX_SECONDS))
    scene = parse_prompt(prompt)
    network = TinyMotionNetwork()
    total = max(1, int(math.ceil(duration * FPS)))
    os.makedirs(os.path.dirname(video_path), exist_ok=True)
    os.makedirs(os.path.dirname(thumbnail_path), exist_ok=True)
    ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
    command = [
        ffmpeg, "-y", "-loglevel", "error", "-f", "rawvideo", "-vcodec", "rawvideo",
        "-pix_fmt", "rgb24", "-s", f"{width}x{height}", "-r", str(FPS), "-i", "-",
        "-an", "-c:v", "libx264", "-preset", "veryfast", "-crf", "27", "-pix_fmt", "yuv420p",
        "-movflags", "+faststart", video_path,
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    first = None
    try:
        for frame_no in range(total):
            frame = render_frame(scene, network, frame_no / FPS, width, height)
            if first is None:
                first = frame.copy()
            process.stdin.write(frame.tobytes())
            if progress and (frame_no % max(1, FPS // 2) == 0 or frame_no == total - 1):
                progress((frame_no + 1) / total)
        process.stdin.close()
        error = process.stderr.read().decode("utf-8", "replace")
        exit_code = process.wait()
    except Exception:
        if process.stdin and not process.stdin.closed:
            process.stdin.close()
        process.kill()
        process.wait()
        raise
    if exit_code != 0:
        raise RuntimeError(f"Не удалось собрать MP4: {error[-500:]}")
    with open(thumbnail_path, "wb") as handle:
        handle.write(png_bytes(first if first is not None else render_frame(scene, network, 0)))
    return {"scene": scene.kind, "palette": scene.palette, "fps": FPS, "width": width, "height": height}
