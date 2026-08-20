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

    SANITY_FILTER_THRESHOLDS = {
        "minimum_luminance_correlation": 0.70,
        "minimum_luminance_std_ratio": 0.30,
        "maximum_luminance_std_ratio": 3.00,
        "maximum_mean_luminance_shift": 0.35,
        "maximum_added_clipped_channel_fraction": 0.25,
    }

    def __init__(
        self,
        coco_root: str,
        split: str,
        lut_root: str,
        image_size: int,
        deterministic: bool = False,
        seed: int = 160122,
        sanity_filter: bool = False,
        sanity_max_attempts: int = 8,
    ) -> None:
        self.image_size = int(image_size)
        self.deterministic = bool(deterministic)
        self.seed = int(seed)
        self.sanity_filter = bool(sanity_filter)
        self.sanity_max_attempts = int(sanity_max_attempts)
        if self.sanity_max_attempts <= 0:
            raise ValueError("sanity_max_attempts must be positive")

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

    @staticmethod
    def _is_sane_rec709_pair(base: np.ndarray, styled: np.ndarray) -> bool:
        """Reject only catastrophic Rec.709/LUT incompatibilities.

        Thresholds are deliberately lenient: creative color casts remain
        valid, while severe clipping, luminance collapse/inversion, or extreme
        exposure shifts are rejected.
        """
        base = base.astype(np.float32, copy=False)
        styled = styled.astype(np.float32, copy=False)
        luma_weights = np.asarray([0.2126, 0.7152, 0.0722], dtype=np.float32)
        base_luma = base @ luma_weights
        styled_luma = styled @ luma_weights
        base_centered = base_luma - float(base_luma.mean())
        styled_centered = styled_luma - float(styled_luma.mean())
        denom = float(
            np.sqrt(np.square(base_centered).sum() * np.square(styled_centered).sum())
        )
        correlation = (
            float((base_centered * styled_centered).sum()) / max(denom, 1e-8)
        )
        std_ratio = float(styled_luma.std()) / max(float(base_luma.std()), 1e-6)
        mean_shift = abs(float(styled_luma.mean()) - float(base_luma.mean()))
        base_clip = float(((base <= 1.0 / 255.0) | (base >= 254.0 / 255.0)).mean())
        styled_clip = float(
            ((styled <= 1.0 / 255.0) | (styled >= 254.0 / 255.0)).mean()
        )
        added_clip = max(0.0, styled_clip - base_clip)
        limits = NeuralPresetCOCOLUTPairs.SANITY_FILTER_THRESHOLDS
        return bool(
            correlation >= limits["minimum_luminance_correlation"]
            and limits["minimum_luminance_std_ratio"]
            <= std_ratio
            <= limits["maximum_luminance_std_ratio"]
            and mean_shift <= limits["maximum_mean_luminance_shift"]
            and added_clip
            <= limits["maximum_added_clipped_channel_fraction"]
        )

    def __getitem__(
        self, index: int
    ) -> dict[str, torch.Tensor | str | int | float]:
        image_path = self.image_paths[int(index)]
        image = Image.open(image_path).convert("RGB")
        image = image.resize(
            (self.image_size, self.image_size),
            Image.Resampling.BICUBIC,
        )

        if not self.sanity_filter:
            lut_index_a, lut_index_b = self._sample_lut_indices(index)
            image_a = image.filter(self.luts[lut_index_a])
            image_b = image.filter(self.luts[lut_index_b])
            attempts_a = attempts_b = 1
            fallback_a = fallback_b = 0
        else:
            base_array = np.asarray(image, dtype=np.float32) / 255.0
            rng = (
                np.random.default_rng(self.seed + int(index))
                if self.deterministic
                else None
            )

            def draw_index() -> int:
                if rng is not None:
                    return int(rng.integers(0, len(self.luts)))
                return int(np.random.randint(0, len(self.luts)))

            def draw_sane_style() -> tuple[int, Image.Image, int, int]:
                chosen_index = 0
                chosen_image = image
                for attempt in range(1, self.sanity_max_attempts + 1):
                    chosen_index = draw_index()
                    chosen_image = image.filter(self.luts[chosen_index])
                    styled_array = np.asarray(chosen_image, dtype=np.float32) / 255.0
                    if self._is_sane_rec709_pair(base_array, styled_array):
                        return chosen_index, chosen_image, attempt, 0
                return chosen_index, chosen_image, self.sanity_max_attempts, 1

            lut_index_a, image_a, attempts_a, fallback_a = draw_sane_style()
            lut_index_b, image_b, attempts_b, fallback_b = draw_sane_style()

        return {
            "content": to_tensor(image),
            "image_a": to_tensor(image_a),
            "image_b": to_tensor(image_b),
            "image_path": str(image_path),
            "lut_a": str(self.lut_paths[lut_index_a]),
            "lut_b": str(self.lut_paths[lut_index_b]),
            "sanity_attempts_mean": 0.5 * (attempts_a + attempts_b),
            "sanity_fallback_ratio": 0.5 * (fallback_a + fallback_b),
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
        sanity_filter=(
            bool(getattr(cfg.data, "sanity_filter", False)) and bool(train)
        ),
        sanity_max_attempts=int(getattr(cfg.data, "sanity_max_attempts", 8)),
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
