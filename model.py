"""Мультимодальная модель, написанная с нуля на PyTorch.

Два энкодера:
  - ImageEncoder: маленькая CNN (свёртки + пулинг) -> вектор фигуры.
  - TextEncoder:  Embedding + GRU -> вектор подписи.

Затем Fusion: склейка двух векторов -> MLP -> логит (совпадает/нет).

Никаких предобученных весов, huggingface, готовых бэкбонов и внешних API.
"""

import torch
from torch import nn


class ImageEncoder(nn.Module):
    """CNN: 3x32x32 -> вектор (out_dim)."""

    def __init__(self, out_dim: int = 64):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                # 16x16
            nn.Conv2d(16, 32, kernel_size=3, padding=1), nn.ReLU(),
            nn.MaxPool2d(2),                                # 8x8
        )
        self.head = nn.Linear(32 * 8 * 8, out_dim)

    def forward(self, x):
        x = self.features(x)          # (B, 32, 8, 8)
        x = x.flatten(1)              # (B, 2048)
        return self.head(x)           # (B, out_dim)


class TextEncoder(nn.Module):
    """Embedding + GRU -> вектор подписи (out_dim)."""

    def __init__(self, vocab_size: int, emb_dim: int = 32,
                 hidden: int = 64, out_dim: int = 64):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.gru = nn.GRU(emb_dim, hidden, batch_first=True)
        self.head = nn.Linear(hidden, out_dim)

    def forward(self, ids):
        emb = self.embedding(ids)     # (B, L, emb_dim)
        _, h = self.gru(emb)          # h: (1, B, hidden) — последнее скрытое
        h = h.squeeze(0)              # (B, hidden)
        return self.head(h)           # (B, out_dim)


class MultimodalModel(nn.Module):
    """Склейка модальностей + классификация (бинарная)."""

    def __init__(self, vocab_size: int):
        super().__init__()
        self.img_enc = ImageEncoder(out_dim=64)
        self.txt_enc = TextEncoder(vocab_size, out_dim=64)
        self.fusion = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, img, ids):
        vi = self.img_enc(img)       # (B, 64)
        vt = self.txt_enc(ids)       # (B, 64)
        z = torch.cat([vi, vt], dim=1)  # (B, 128)
        return self.fusion(z).squeeze(1)  # (B,) — логиты
