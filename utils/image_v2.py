from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence
import textwrap

from PIL import Image, ImageDraw, ImageFont
import torch
from torchvision.transforms.functional import to_pil_image
from torchvision.utils import make_grid


def _font_v2(size: int):
    try:
        return ImageFont.load_default(size=int(size))
    except TypeError:
        return ImageFont.load_default()


def _wrap_label_v2(
    label: str,
    approximate_characters: int,
) -> str:
    words = str(label).replace("_", " ")
    return "\n".join(
        textwrap.wrap(
            words,
            width=max(
                int(approximate_characters),
                8,
            ),
        )
    )


def save_labeled_grid_v2(
    tensors: Iterable[torch.Tensor],
    path,
    labels: Sequence[str],
    title: str,
    max_items: int = 2,
    padding: int = 4,
) -> None:
    """
    Save aligned tensor columns with readable wrapped labels.

    Each tensor must have shape [B, 3, H, W]. The function lays out one batch
    sample per row and one tensor source per column.
    """
    tensors = list(tensors)

    if len(tensors) != len(labels):
        raise ValueError(
            "labels must match tensor columns"
        )

    if not tensors:
        raise ValueError(
            "at least one tensor is required"
        )

    for tensor in tensors:
        if tensor.ndim != 4:
            raise ValueError(
                "every tensor must have shape [B, C, H, W]"
            )

    batch_size = min(
        int(max_items),
        *(
            int(tensor.shape[0])
            for tensor in tensors
        ),
    )

    if batch_size <= 0:
        raise ValueError(
            "visualization tensors contain no samples"
        )

    images = []

    for row_index in range(batch_size):
        for tensor in tensors:
            images.append(
                tensor[row_index]
                .detach()
                .float()
                .cpu()
                .clamp(0.0, 1.0)
            )

    number_of_columns = len(tensors)

    grid = make_grid(
        images,
        nrow=number_of_columns,
        padding=int(padding),
        pad_value=1.0,
    )

    grid_image = to_pil_image(grid).convert("RGB")

    cell_width = int(images[0].shape[-1]) + int(padding)
    approximate_characters = max(
        cell_width // 8,
        10,
    )

    wrapped_labels = [
        _wrap_label_v2(
            label,
            approximate_characters,
        )
        for label in labels
    ]

    maximum_label_lines = max(
        label.count("\n") + 1
        for label in wrapped_labels
    )

    title_height = 25
    label_line_height = 14
    header_height = (
        title_height
        + maximum_label_lines * label_line_height
        + 12
    )

    canvas = Image.new(
        "RGB",
        (
            grid_image.width,
            grid_image.height + header_height,
        ),
        "white",
    )
    canvas.paste(
        grid_image,
        (0, header_height),
    )

    draw = ImageDraw.Draw(canvas)
    title_font = _font_v2(16)
    label_font = _font_v2(12)

    draw.text(
        (8, 5),
        str(title),
        fill="black",
        font=title_font,
    )

    for column_index, label in enumerate(
        wrapped_labels
    ):
        x_position = (
            int(padding)
            + column_index * cell_width
            + 2
        )
        draw.multiline_text(
            (
                x_position,
                title_height,
            ),
            label,
            fill="black",
            font=label_font,
            spacing=1,
        )

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(output_path)
