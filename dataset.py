"""Синтетический мультимодальный датасет: «фигура + подпись».

Никаких внешних данных и API — изображения рисуем сами через numpy,
подписи генерируем из словаря. Задача: по паре (картинка, текст)
предсказать, описывает ли текст картинку (1) или нет (0).
"""

import numpy as np
import torch
from torch.utils.data import Dataset

# 3 фигуры и 3 цвета — полностью самодостаточный «мир».
SHAPES = ["circle", "square", "triangle"]
COLORS = {
    "red":   (1.0, 0.0, 0.0),
    "green": (0.0, 1.0, 0.0),
    "blue":  (0.0, 0.0, 1.0),
}

# Словарь подписей: всегда формат "<цвет> <фигура>", например "red circle".
VOCAB = ["<pad>", "<unk>", "red", "green", "blue",
         "circle", "square", "triangle"]

IMG_SIZE = 32
MAX_LEN = 4  # макс. число токенов в подписи ("red" + "triangle" = 2 + паддинг)


def encode_caption(caption: str) -> torch.Tensor:
    """Строку -> список индексов словаря -> тензор фикс. длины."""
    ids = [VOCAB.index(tok) for tok in caption.split()]
    ids += [0] * (MAX_LEN - len(ids))  # pad до фикс. длины
    return torch.tensor(ids, dtype=torch.long)


# --- Рисование фигур в numpy (без PIL/matplotlib) ---------------------------

def _edge(a, b, x, y):
    """Ориентированное расстояние от точки (x,y) до ребра (a->b)."""
    return (x - a[0]) * (b[1] - a[1]) - (y - a[1]) * (b[0] - a[0])


def _point_in_triangle(xx, yy, v1, v2, v3):
    e1 = _edge(v1, v2, xx, yy)
    e2 = _edge(v2, v3, xx, yy)
    e3 = _edge(v3, v1, xx, yy)
    inside = ((e1 >= 0) & (e2 >= 0) & (e3 >= 0)) | \
             ((e1 <= 0) & (e2 <= 0) & (e3 <= 0))
    return inside


def render_image(shape: str, color: str, size: int = IMG_SIZE) -> np.ndarray:
    """Рисует залитую фигуру на чёрном фоне. Возвращает (H, W, 3) float32."""
    img = np.zeros((size, size, 3), dtype=np.float32)
    c = size // 2
    r = size * 0.34
    yy, xx = np.mgrid[0:size, 0:size]
    dx, dy = xx - c, yy - c

    if shape == "circle":
        mask = dx * dx + dy * dy <= r * r
    elif shape == "square":
        h = r * 0.85
        mask = (np.abs(dx) <= h) & (np.abs(dy) <= h)
    elif shape == "triangle":
        v1 = (c, c - r)
        v2 = (c - r * 0.866, c + r * 0.5)
        v3 = (c + r * 0.866, c + r * 0.5)
        mask = _point_in_triangle(xx, yy, v1, v2, v3)
    else:
        raise ValueError(f"Unknown shape: {shape}")

    img[mask] = COLORS[color]
    return img


# --- Датасет ----------------------------------------------------------------

class MultimodalDataset(Dataset):
    """Генерирует пары (картинка, подпись, метка) детерминированно (seed)."""

    def __init__(self, n: int = 2048, seed: int = 0):
        self.n = n
        self.seed = seed
        rng = np.random.default_rng(seed)
        self.samples = []

        for _ in range(n):
            shape = rng.choice(SHAPES)
            color = rng.choice(list(COLORS))
            img = render_image(shape, color)

            match = bool(rng.integers(0, 2))
            if match:
                caption = f"{color} {shape}"
            else:
                # Несовпадение: меняем либо цвет, либо фигуру.
                if rng.integers(0, 2) == 0:
                    wrong_shape = rng.choice([s for s in SHAPES if s != shape])
                    caption = f"{color} {wrong_shape}"
                else:
                    wrong_color = rng.choice([c for c in COLORS if c != color])
                    caption = f"{wrong_color} {shape}"

            self.samples.append((img, encode_caption(caption), match,
                                 shape, color, caption))

    def __len__(self):
        return self.n

    def __getitem__(self, idx):
        img, ids, match, shape, color, caption = self.samples[idx]
        # (H, W, 3) -> (3, H, W) как принято в torch
        img_t = torch.from_numpy(img).permute(2, 0, 1)
        label = torch.tensor(float(match), dtype=torch.float32)
        return img_t, ids, label


def collate(batch):
    """Собирает батч: картинки, токены, метки (+ метаданные для демо)."""
    imgs = torch.stack([b[0] for b in batch])
    ids = torch.stack([b[1] for b in batch])
    labels = torch.stack([b[2] for b in batch])
    return imgs, ids, labels
