"""Обучение мультимодальной модели на синтетических данных.

Запуск:
    python train.py [--epochs 10] [--samples 2048] [--lr 1e-3]

Сохраняет веса и словарь в covalt_model.pt.
"""

import argparse
import random

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from dataset import VOCAB, MultimodalDataset, collate
from model import MultimodalModel


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    correct, total, loss_sum = 0, 0, 0.0
    loss_fn = nn.BCEWithLogitsLoss()
    for imgs, ids, labels in loader:
        imgs, ids, labels = imgs.to(device), ids.to(device), labels.to(device)
        logits = model(imgs, ids)
        loss_sum += loss_fn(logits, labels).item() * len(labels)
        preds = (torch.sigmoid(logits) >= 0.5).float()
        correct += (preds == labels).sum().item()
        total += len(labels)
    return loss_sum / total, correct / total


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--samples", type=int, default=2048)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Устройство: {device}")

    # 90% обучаем, 10% проверяем.
    n_train = int(args.samples * 0.9)
    train_ds = MultimodalDataset(n=n_train, seed=args.seed)
    val_ds = MultimodalDataset(n=args.samples - n_train, seed=args.seed + 1)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True,
                              collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False,
                            collate_fn=collate)

    model = MultimodalModel(len(VOCAB)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Параметров модели: {n_params:,}")

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    loss_fn = nn.BCEWithLogitsLoss()

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for imgs, ids, labels in train_loader:
            imgs, ids, labels = imgs.to(device), ids.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(imgs, ids)
            loss = loss_fn(logits, labels)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(labels)

        val_loss, val_acc = evaluate(model, val_loader, device)
        print(f"Эпоха {epoch:>2}/{args.epochs} | "
              f"train_loss {running/len(train_ds):.4f} | "
              f"val_loss {val_loss:.4f} | val_acc {val_acc*100:.1f}%")

    checkpoint = {
        "state_dict": model.state_dict(),
        "vocab": VOCAB,
        "config": {"out_dim": 64},
    }
    torch.save(checkpoint, "covalt_model.pt")
    print("Сохранено: covalt_model.pt")


if __name__ == "__main__":
    main()
