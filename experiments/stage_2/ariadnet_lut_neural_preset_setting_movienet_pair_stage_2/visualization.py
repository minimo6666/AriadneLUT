from __future__ import annotations

from pathlib import Path
from typing import Sequence

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torchvision.transforms.functional import to_pil_image


def _to_pil(tensor: torch.Tensor) -> Image.Image:
    return to_pil_image(tensor.detach().float().cpu().clamp(0.0, 1.0)).convert("RGB")


def save_stage2_grid(
    tensors: Sequence[torch.Tensor],
    labels: Sequence[str],
    path: str | Path,
    max_items: int = 2,
    tile_size: int = 220,
    title: str = "",
) -> None:
    if len(tensors) != len(labels):
        raise ValueError("tensors and labels must have the same length")
    if not tensors:
        return

    batch = min(int(max_items), int(tensors[0].shape[0]))
    cols = len(tensors)
    label_h = 28
    title_h = 34 if title else 0
    pad = 6
    width = cols * tile_size + (cols + 1) * pad
    height = title_h + batch * (tile_size + label_h + pad) + pad
    canvas = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(canvas)
    font = ImageFont.load_default()
    if title:
        draw.text((pad, 9), title, fill="black", font=font)

    for row in range(batch):
        y0 = title_h + pad + row * (tile_size + label_h + pad)
        for col, (tensor, label) in enumerate(zip(tensors, labels)):
            image = _to_pil(tensor[row])
            image = ImageOps.pad(
                image,
                (tile_size, tile_size),
                method=Image.Resampling.LANCZOS,
                color=(245, 245, 245),
            )
            x0 = pad + col * (tile_size + pad)
            canvas.paste(image, (x0, y0))
            draw.text((x0 + 2, y0 + tile_size + 6), label, fill="black", font=font)

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
