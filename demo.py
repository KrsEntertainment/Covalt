"""Инференс: показывает, как обученная модель работает на примерах.

Запуск (после train.py):
    python demo.py
"""

import torch

from dataset import VOCAB, encode_caption, render_image
from model import MultimodalModel

CHARS = " .:-=+*#%@"


def image_to_ascii(img_rgb, width=28):
    """(H, W, 3) float32 -> ASCII-строка (яркость по каналам RGB)."""
    h, w, _ = img_rgb.shape
    lum = img_rgb.mean(axis=2)  # (H, W)
    # грубое уменьшение без сторонних библиотек
    block_h = max(1, h * 2 // width)  # *2: символы выше, чем широкие
    block_w = max(1, w // width)
    lines = []
    for y in range(0, h, block_h):
        line = ""
        for x in range(0, w, block_w):
            val = lum[y:y + block_h, x:x + block_w].mean()
            line += CHARS[min(int(val * (len(CHARS) - 1)), len(CHARS) - 1)]
        lines.append(line)
    return "\n".join(lines)


@torch.no_grad()
def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load("covalt_model.pt", map_location=device, weights_only=False)
    model = MultimodalModel(len(ckpt["vocab"])).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    # Несколько примеров: часть «совпадает», часть «не совпадает».
    examples = [
        ("circle", "red", "red circle",   True),
        ("square", "blue", "blue square", True),
        ("triangle", "green", "green triangle", True),
        ("circle", "red", "green circle", False),
        ("square", "blue", "red square",  False),
        ("triangle", "green", "green circle", False),
    ]

    print("=" * 60)
    print("Мультимодальный демо: картинка + подпись -> 'совпадает?'")
    print("=" * 60)

    for shape, color, caption, expected in examples:
        img = render_image(shape, color)
        img_t = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0).to(device)
        ids = encode_caption(caption).unsqueeze(0).to(device)

        logit = model(img_t, ids).item()
        prob = torch.sigmoid(torch.tensor(logit)).item()
        pred = prob >= 0.5

        print(f"\nКартинка: {color} {shape}   Подпись: \"{caption}\"")
        print(image_to_ascii(img))
        tag = "OK " if pred == expected else "ERR"
        print(f"{tag} P(совпадает)={prob:.3f}  "
              f"(ожидалось {'да' if expected else 'нет'}, "
              f"модель говорит {'да' if pred else 'нет'})")


if __name__ == "__main__":
    main()
