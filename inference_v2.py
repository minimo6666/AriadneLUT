from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageDraw, ImageFont, ImageOps
import torch
from torch.amp import autocast
from torchvision.transforms.functional import (
    pil_to_tensor,
    to_pil_image,
)

from datasets.coco_wds_v2 import get_coco_loader_v2
from models.ariadne_lut_v2 import AriadneLUTV2
from utils.checkpoint import load_checkpoint
from utils.config import load_config


SUPPORTED_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
    ".webp",
}


def _font_v2(size: int):
    try:
        return ImageFont.load_default(
            size=int(size)
        )
    except TypeError:
        return ImageFont.load_default()


def _safe_name_v2(path: Path) -> str:
    cleaned = "".join(
        character
        if (
            character.isalnum()
            or character in "-_"
        )
        else "_"
        for character in path.stem
    )
    return cleaned or "image"


def _collect_images_v2(
    explicit_paths: Iterable[str],
    directory: str,
    recursive: bool,
    maximum: int,
) -> list[Path]:
    candidates: list[Path] = []

    for raw_path in explicit_paths:
        path = Path(raw_path).expanduser()
        if not path.is_file():
            raise FileNotFoundError(
                f"Image file not found: {path}"
            )
        if path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(
                f"Unsupported image extension: {path}"
            )
        candidates.append(path.resolve())

    if directory:
        root = Path(directory).expanduser()
        if not root.is_dir():
            raise NotADirectoryError(
                f"Image directory not found: {root}"
            )

        iterator = (
            root.rglob("*")
            if recursive
            else root.glob("*")
        )

        directory_paths = sorted(
            path.resolve()
            for path in iterator
            if (
                path.is_file()
                and path.suffix.lower()
                in SUPPORTED_EXTENSIONS
            )
        )
        candidates.extend(directory_paths)

    unique: list[Path] = []
    seen = set()

    for path in candidates:
        key = str(path)
        if key not in seen:
            seen.add(key)
            unique.append(path)

    if maximum > 0:
        unique = unique[:maximum]

    return unique


def _load_rgb_pil_v2(path: Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")


def _pil_to_tensor_v2(
    image: Image.Image,
    device: torch.device,
) -> torch.Tensor:
    return (
        pil_to_tensor(image)
        .float()
        .div(255.0)
        .unsqueeze(0)
        .to(device)
    )


def _tensor_to_pil_v2(
    tensor: torch.Tensor,
) -> Image.Image:
    image = (
        tensor[0]
        .detach()
        .float()
        .cpu()
        .clamp(0.0, 1.0)
    )
    return to_pil_image(image).convert("RGB")


def _save_tensor_v2(
    tensor: torch.Tensor,
    path: Path,
) -> None:
    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    _tensor_to_pil_v2(tensor).save(path)


def _thumbnail_v2(
    image: Image.Image,
    size: int,
) -> Image.Image:
    return ImageOps.pad(
        image.convert("RGB"),
        (int(size), int(size)),
        method=Image.Resampling.LANCZOS,
        color=(245, 245, 245),
        centering=(0.5, 0.5),
    )


def _draw_centered_text_v2(
    draw: ImageDraw.ImageDraw,
    box: tuple[int, int, int, int],
    text: str,
    font,
) -> None:
    left, top, right, bottom = box
    text_box = draw.multiline_textbbox(
        (0, 0),
        text,
        font=font,
        align="center",
        spacing=2,
    )
    text_width = text_box[2] - text_box[0]
    text_height = text_box[3] - text_box[1]

    x = left + (right - left - text_width) / 2
    y = top + (bottom - top - text_height) / 2

    draw.multiline_text(
        (x, y),
        text,
        fill="black",
        font=font,
        align="center",
        spacing=2,
    )


def _build_matrix_grid_v2(
    content_records: list[dict],
    style_records: list[dict],
    output_lookup: dict[tuple[int, int], Path],
    path: Path,
    thumbnail_size: int,
) -> None:
    """
    Build one comparison matrix:

        first column: original content images
        top row:      style reference images
        cells:        stylized outputs

    This layout makes reusable Style LUT behavior easy to inspect.
    """
    cell_size = int(thumbnail_size)
    label_height = 54
    padding = 4

    columns = len(style_records) + 1
    rows = len(content_records) + 1

    width = (
        columns * cell_size
        + (columns + 1) * padding
    )
    height = (
        rows * (cell_size + label_height)
        + (rows + 1) * padding
    )

    canvas = Image.new(
        "RGB",
        (width, height),
        "white",
    )
    draw = ImageDraw.Draw(canvas)

    label_font = _font_v2(13)
    corner_font = _font_v2(15)

    def cell_origin(
        row: int,
        column: int,
    ) -> tuple[int, int]:
        x = padding + column * (
            cell_size + padding
        )
        y = padding + row * (
            cell_size + label_height + padding
        )
        return x, y

    # Corner cell.
    corner_x, corner_y = cell_origin(0, 0)
    _draw_centered_text_v2(
        draw,
        (
            corner_x,
            corner_y,
            corner_x + cell_size,
            corner_y + cell_size + label_height,
        ),
        "Content rows\n×\nStyle columns",
        corner_font,
    )

    # Style headers.
    for style_index, style_record in enumerate(
        style_records
    ):
        x, y = cell_origin(
            0,
            style_index + 1,
        )
        canvas.paste(
            _thumbnail_v2(
                style_record["pil"],
                cell_size,
            ),
            (x, y),
        )
        _draw_centered_text_v2(
            draw,
            (
                x,
                y + cell_size,
                x + cell_size,
                y + cell_size + label_height,
            ),
            f"style\n{style_record['name']}",
            label_font,
        )

    # Content rows and stylized cells.
    for content_index, content_record in enumerate(
        content_records
    ):
        row = content_index + 1
        x, y = cell_origin(row, 0)

        canvas.paste(
            _thumbnail_v2(
                content_record["pil"],
                cell_size,
            ),
            (x, y),
        )
        _draw_centered_text_v2(
            draw,
            (
                x,
                y + cell_size,
                x + cell_size,
                y + cell_size + label_height,
            ),
            f"content\n{content_record['name']}",
            label_font,
        )

        for style_index, style_record in enumerate(
            style_records
        ):
            output_path = output_lookup[
                (
                    content_index,
                    style_index,
                )
            ]
            output_image = _load_rgb_pil_v2(
                output_path
            )

            x, y = cell_origin(
                row,
                style_index + 1,
            )
            canvas.paste(
                _thumbnail_v2(
                    output_image,
                    cell_size,
                ),
                (x, y),
            )
            _draw_centered_text_v2(
                draw,
                (
                    x,
                    y + cell_size,
                    x + cell_size,
                    y + cell_size + label_height,
                ),
                "stylized",
                label_font,
            )

    path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    canvas.save(path)



def _sample_coco_tensors_v2(
    cfg,
    phase: str,
    number_of_samples: int,
    scan_batches: int,
    seed: int,
) -> tuple[list[torch.Tensor], int]:
    """
    Reservoir-sample images from the COCO WebDataset loader.

    Validation is the recommended phase because its image transform is
    deterministic. The train phase is also supported but uses the project's
    training transform.

    Returns:
        sampled_images:
            A list of CPU tensors with shape [3, H, W].
        number_of_seen_images:
            The number of stream images considered by reservoir sampling.
    """
    if number_of_samples <= 0:
        return [], 0

    if scan_batches <= 0:
        raise ValueError(
            "--coco-scan-batches must be positive"
        )

    if phase not in {"train", "val"}:
        raise ValueError(
            "--coco-phase must be 'train' or 'val'"
        )

    random_generator = random.Random(
        int(seed)
    )
    torch.manual_seed(int(seed))

    loader = get_coco_loader_v2(
        cfg,
        phase=phase,
    )

    reservoir: list[torch.Tensor] = []
    number_of_seen_images = 0

    for batch_index, batch in enumerate(loader):
        images = batch[0]

        if images.ndim != 4:
            raise RuntimeError(
                "COCO loader must return an image batch "
                "with shape [B, C, H, W]"
            )

        for image in images:
            number_of_seen_images += 1
            image_cpu = (
                image.detach()
                .float()
                .cpu()
                .clamp(0.0, 1.0)
            )

            if len(reservoir) < number_of_samples:
                reservoir.append(image_cpu)
            else:
                replacement_index = (
                    random_generator.randrange(
                        number_of_seen_images
                    )
                )

                if replacement_index < number_of_samples:
                    reservoir[
                        replacement_index
                    ] = image_cpu

        if batch_index + 1 >= scan_batches:
            break

    if len(reservoir) < number_of_samples:
        raise RuntimeError(
            "COCO sampling found only "
            f"{len(reservoir)} image(s), but "
            f"{number_of_samples} were requested. "
            "Increase --coco-scan-batches, reduce the requested "
            "counts, or check the COCO WebDataset configuration."
        )

    # Shuffle after reservoir sampling before splitting into content/style.
    random_generator.shuffle(reservoir)

    return reservoir, number_of_seen_images


def _save_sampled_coco_inputs_v2(
    sampled_images: list[torch.Tensor],
    content_count: int,
    style_count: int,
    output_root: Path,
    phase: str,
    seed: int,
) -> tuple[list[Path], list[Path], dict]:
    expected_count = (
        int(content_count)
        + int(style_count)
    )

    if len(sampled_images) != expected_count:
        raise ValueError(
            "sampled_images count does not match "
            "content_count + style_count"
        )

    sampled_root = (
        output_root
        / "sampled_coco_inputs"
    )
    content_root = sampled_root / "contents"
    style_root = sampled_root / "styles"

    content_root.mkdir(
        parents=True,
        exist_ok=True,
    )
    style_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    content_paths: list[Path] = []
    style_paths: list[Path] = []

    for index, tensor in enumerate(
        sampled_images[:content_count]
    ):
        path = (
            content_root
            / f"coco_content_{index:03d}.png"
        )
        to_pil_image(
            tensor.clamp(0.0, 1.0)
        ).save(path)
        content_paths.append(path.resolve())

    for index, tensor in enumerate(
        sampled_images[content_count:]
    ):
        path = (
            style_root
            / f"coco_style_{index:03d}.png"
        )
        to_pil_image(
            tensor.clamp(0.0, 1.0)
        ).save(path)
        style_paths.append(path.resolve())

    sampling_information = {
        "enabled": True,
        "phase": str(phase),
        "seed": int(seed),
        "content_count": int(content_count),
        "style_count": int(style_count),
        "sampled_input_directory": str(
            sampled_root.resolve()
        ),
    }

    return (
        content_paths,
        style_paths,
        sampling_information,
    )

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "AriadneLUT V2 multi-image inference. "
            "Each content is normalized once, and each Style LUT "
            "is extracted once and reused across all contents."
        )
    )

    parser.add_argument(
        "--config",
        required=True,
    )
    parser.add_argument(
        "--checkpoint",
        required=True,
    )

    parser.add_argument(
        "--content",
        nargs="*",
        default=[],
        help=(
            "One or more explicit content image paths."
        ),
    )
    parser.add_argument(
        "--content-dir",
        default="",
        help=(
            "Directory containing content images."
        ),
    )

    parser.add_argument(
        "--style",
        nargs="*",
        default=[],
        help=(
            "One or more explicit style image paths."
        ),
    )
    parser.add_argument(
        "--style-dir",
        default="",
        help=(
            "Directory containing style images."
        ),
    )

    parser.add_argument(
        "--output-dir",
        required=True,
    )
    parser.add_argument(
        "--device",
        default="cuda",
    )
    parser.add_argument(
        "--recursive",
        action="store_true",
    )
    parser.add_argument(
        "--max-contents",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--max-styles",
        type=int,
        default=4,
    )
    parser.add_argument(
        "--thumbnail-size",
        type=int,
        default=256,
    )
    parser.add_argument(
        "--grid-name",
        default="multi_style_comparison.png",
    )
    parser.add_argument(
        "--save-canonical",
        action="store_true",
    )

    parser.add_argument(
        "--sample-coco",
        action="store_true",
        help=(
            "Randomly sample additional content and style images "
            "from the configured COCO WebDataset."
        ),
    )
    parser.add_argument(
        "--coco-content-count",
        type=int,
        default=8,
        help=(
            "Number of random COCO content images when "
            "--sample-coco is enabled."
        ),
    )
    parser.add_argument(
        "--coco-style-count",
        type=int,
        default=4,
        help=(
            "Number of random COCO style images when "
            "--sample-coco is enabled."
        ),
    )
    parser.add_argument(
        "--coco-phase",
        choices=["train", "val"],
        default="val",
        help=(
            "COCO split used for random sampling. "
            "The validation split is recommended."
        ),
    )
    parser.add_argument(
        "--coco-scan-batches",
        type=int,
        default=200,
        help=(
            "Maximum number of COCO loader batches considered "
            "by reservoir sampling."
        ),
    )
    parser.add_argument(
        "--coco-seed",
        type=int,
        default=12345,
        help=(
            "Random seed for reproducible COCO sampling."
        ),
    )
    parser.add_argument(
        "--amp",
        action="store_true",
        help=(
            "Use CUDA autocast during inference."
        ),
    )

    arguments = parser.parse_args()

    cfg = load_config(arguments.config)

    output_root = Path(
        arguments.output_dir
    ).expanduser()
    output_root.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Existing manual file/directory mode remains unchanged.
    content_paths = _collect_images_v2(
        explicit_paths=arguments.content,
        directory=arguments.content_dir,
        recursive=arguments.recursive,
        maximum=int(arguments.max_contents),
    )
    style_paths = _collect_images_v2(
        explicit_paths=arguments.style,
        directory=arguments.style_dir,
        recursive=arguments.recursive,
        maximum=int(arguments.max_styles),
    )

    coco_sampling_information = {
        "enabled": False,
    }

    # Optional COCO random-sampling mode. It can be used alone or combined
    # with explicitly supplied files/directories.
    if arguments.sample_coco:
        coco_content_count = int(
            arguments.coco_content_count
        )
        coco_style_count = int(
            arguments.coco_style_count
        )

        if coco_content_count < 0:
            raise ValueError(
                "--coco-content-count cannot be negative"
            )

        if coco_style_count < 0:
            raise ValueError(
                "--coco-style-count cannot be negative"
            )

        total_coco_samples = (
            coco_content_count
            + coco_style_count
        )

        if total_coco_samples <= 0:
            raise ValueError(
                "--sample-coco requires a positive total of "
                "--coco-content-count and --coco-style-count"
            )

        (
            sampled_coco_images,
            number_of_seen_coco_images,
        ) = _sample_coco_tensors_v2(
            cfg=cfg,
            phase=str(arguments.coco_phase),
            number_of_samples=total_coco_samples,
            scan_batches=int(
                arguments.coco_scan_batches
            ),
            seed=int(arguments.coco_seed),
        )

        (
            sampled_content_paths,
            sampled_style_paths,
            coco_sampling_information,
        ) = _save_sampled_coco_inputs_v2(
            sampled_images=sampled_coco_images,
            content_count=coco_content_count,
            style_count=coco_style_count,
            output_root=output_root,
            phase=str(arguments.coco_phase),
            seed=int(arguments.coco_seed),
        )

        coco_sampling_information[
            "scan_batches"
        ] = int(arguments.coco_scan_batches)
        coco_sampling_information[
            "seen_images"
        ] = int(number_of_seen_coco_images)

        content_paths.extend(
            sampled_content_paths
        )
        style_paths.extend(
            sampled_style_paths
        )

    if not content_paths:
        raise ValueError(
            "No content images were found. Use --content, "
            "--content-dir, or --sample-coco."
        )

    if not style_paths:
        raise ValueError(
            "No style images were found. Use --style, "
            "--style-dir, or --sample-coco."
        )

    requested_device = str(arguments.device)
    device = torch.device(
        requested_device
        if (
            not requested_device.startswith("cuda")
            or torch.cuda.is_available()
        )
        else "cpu"
    )

    model = AriadneLUTV2(
        cfg.model
    ).to(device)

    load_checkpoint(
        arguments.checkpoint,
        model,
        strict=True,
        map_location=device,
    )

    model.eval()

    amp_enabled = (
        bool(arguments.amp)
        and device.type == "cuda"
    )

    style_records = []

    # Extract every reusable Style LUT exactly once.
    with torch.inference_mode():
        for style_path in style_paths:
            style_pil = _load_rgb_pil_v2(
                style_path
            )
            style_tensor = _pil_to_tensor_v2(
                style_pil,
                device,
            )

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                style_state = (
                    model.extract_style(
                        style_tensor
                    )
                )

            style_records.append(
                {
                    "path": style_path,
                    "name": _safe_name_v2(
                        style_path
                    ),
                    "pil": style_pil,
                    "style_lut": (
                        style_state[
                            "style_lut"
                        ].detach()
                    ),
                }
            )

    content_records = []
    output_lookup = {}
    manifest_outputs = []

    # Normalize every content once, then apply all cached Style LUTs.
    with torch.inference_mode():
        for content_index, content_path in enumerate(
            content_paths
        ):
            content_pil = _load_rgb_pil_v2(
                content_path
            )
            content_tensor = _pil_to_tensor_v2(
                content_pil,
                device,
            )

            with autocast(
                device_type=device.type,
                enabled=amp_enabled,
            ):
                normalization_state = (
                    model.normalize(
                        content_tensor
                    )
                )

            content_name = _safe_name_v2(
                content_path
            )

            content_records.append(
                {
                    "path": content_path,
                    "name": content_name,
                    "pil": content_pil,
                }
            )

            if arguments.save_canonical:
                canonical_path = (
                    output_root
                    / "canonical"
                    / f"{content_name}.png"
                )
                _save_tensor_v2(
                    normalization_state[
                        "canonical"
                    ],
                    canonical_path,
                )

            for style_index, style_record in enumerate(
                style_records
            ):
                with autocast(
                    device_type=device.type,
                    enabled=amp_enabled,
                ):
                    stylized = model.apply_style(
                        normalization_state[
                            "canonical"
                        ],
                        style_record[
                            "style_lut"
                        ],
                    )

                style_name = style_record["name"]

                output_path = (
                    output_root
                    / "stylized"
                    / style_name
                    / (
                        f"{content_name}"
                        f"__to__{style_name}.png"
                    )
                )

                _save_tensor_v2(
                    stylized,
                    output_path,
                )

                output_lookup[
                    (
                        content_index,
                        style_index,
                    )
                ] = output_path

                manifest_outputs.append(
                    {
                        "content": str(
                            content_path
                        ),
                        "style": str(
                            style_record["path"]
                        ),
                        "output": str(
                            output_path
                        ),
                    }
                )

    grid_path = (
        output_root
        / arguments.grid_name
    )

    _build_matrix_grid_v2(
        content_records=content_records,
        style_records=style_records,
        output_lookup=output_lookup,
        path=grid_path,
        thumbnail_size=int(
            arguments.thumbnail_size
        ),
    )

    manifest = {
        "config": str(arguments.config),
        "checkpoint": str(
            arguments.checkpoint
        ),
        "device": str(device),
        "content_count": len(
            content_records
        ),
        "style_count": len(
            style_records
        ),
        "style_lut_extractions": len(
            style_records
        ),
        "content_normalizations": len(
            content_records
        ),
        "outputs": manifest_outputs,
        "comparison_grid": str(
            grid_path
        ),
        "coco_sampling": (
            coco_sampling_information
        ),
    }

    manifest_path = (
        output_root
        / "manifest.json"
    )
    manifest_path.write_text(
        json.dumps(
            manifest,
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    print(
        f"Loaded checkpoint: {arguments.checkpoint}"
    )
    if coco_sampling_information.get(
        "enabled",
        False,
    ):
        print(
            "COCO random sampling: "
            f"phase={coco_sampling_information['phase']}, "
            f"seed={coco_sampling_information['seed']}, "
            f"seen={coco_sampling_information['seen_images']}, "
            f"sampled contents="
            f"{coco_sampling_information['content_count']}, "
            f"sampled styles="
            f"{coco_sampling_information['style_count']}"
        )

    print(
        f"Contents: {len(content_records)}"
    )
    print(
        f"Styles: {len(style_records)}"
    )
    print(
        f"Stylized outputs: {len(manifest_outputs)}"
    )
    print(
        "Each Style LUT was extracted once and reused "
        "for every content."
    )
    print(
        f"Comparison grid: {grid_path}"
    )
    print(
        f"Manifest: {manifest_path}"
    )


if __name__ == "__main__":
    main()
