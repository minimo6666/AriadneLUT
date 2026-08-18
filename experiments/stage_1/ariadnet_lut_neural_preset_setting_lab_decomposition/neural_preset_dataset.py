from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image
from pillow_lut import load_cube_file
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms.functional import to_tensor


class NeuralPresetCOCOLUTPairs(Dataset):
    """COCO + random .cube pairing used by Neural Preset.

    For every COCO image, two LUTs are sampled independently with replacement
    and applied through Pillow. This intentionally mirrors
    Neural-Preset/datasets/coco.py rather than Ariadne's FilmGradeAugmentor.
    """

    def __init__(
        self,
        coco_root: str,
        split: str,
        lut_root: str,
        image_size: int,
        deterministic: bool = False,
        seed: int = 160122,
    ) -> None:
        self.image_size = int(image_size)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)

        split_root = Path(coco_root) / str(split)
        self.image_paths = sorted(split_root.glob("*.jpg"))
        self.lut_paths = sorted(Path(lut_root).rglob("*.cube"))

        if not self.image_paths:
            raise RuntimeError(f"No COCO .jpg files found in {split_root}")
        if not self.lut_paths:
            raise RuntimeError(f"No .cube LUT files found below {lut_root}")

        # Neural Preset eagerly parses all LUTs before DataLoader workers fork.
        self.luts = [load_cube_file(str(path)) for path in self.lut_paths]

    def __len__(self) -> int:
        return len(self.image_paths)

    def _sample_lut_indices(self, index: int) -> tuple[int, int]:
        if self.deterministic:
            generator = np.random.default_rng(self.seed + int(index))
            indices = generator.integers(0, len(self.luts), size=2)
        else:
            indices = np.random.randint(0, len(self.luts), size=2)
        return int(indices[0]), int(indices[1])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor | str]:
        image_path = self.image_paths[int(index)]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )

        lut_index_a, lut_index_b = self._sample_lut_indices(index)
        image_a = image.filter(self.luts[lut_index_a])
        image_b = image.filter(self.luts[lut_index_b])

        return {
            "content": to_tensor(image),
            "image_a": to_tensor(image_a),
            "image_b": to_tensor(image_b),
            "image_path": str(image_path),
            "lut_a": str(self.lut_paths[lut_index_a]),
            "lut_b": str(self.lut_paths[lut_index_b]),
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = torch.initial_seed() % (2**32)
    np.random.seed(worker_seed)


def build_loader(cfg, split: str, train: bool) -> tuple[DataLoader, NeuralPresetCOCOLUTPairs]:
    dataset = NeuralPresetCOCOLUTPairs(
        coco_root=str(cfg.data.coco_root),
        split=split,
        lut_root=str(cfg.data.lut_root),
        image_size=int(cfg.data.image_size),
        deterministic=(
            bool(cfg.data.deterministic_validation) if not train else False
        ),
        seed=int(cfg.data.validation_seed),
    )

    workers = int(cfg.data.num_workers)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size),
        shuffle=bool(train),
        num_workers=workers,
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=(
            workers > 0 and bool(cfg.data.persistent_workers)
        ),
        worker_init_fn=_seed_worker,
        drop_last=False,
    )
    return loader, dataset
