from __future__ import annotations

from array import array
import json
import random
from pathlib import Path
from typing import Any, BinaryIO

import numpy as np
from PIL import Image
from pillow_lut import load_cube_file
import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from torchvision.transforms.functional import pil_to_tensor


class ResumableDistributedSampler(DistributedSampler):
    """Distributed sampler that can resume at a per-rank sample offset."""

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.start_index = 0

    def set_start_index(self, start_index: int) -> None:
        value = int(start_index)
        if value < 0 or value > self.num_samples:
            raise ValueError(
                f"start_index must be within [0, {self.num_samples}], got {value}"
            )
        self.start_index = value

    def __iter__(self):
        indices = list(super().__iter__())
        return iter(indices[self.start_index :])

    def __len__(self) -> int:
        return max(super().__len__() - self.start_index, 0)

class MovieNetPairDataset(Dataset):
    """MovieNet same-style/different-content pairs with LUT corruption.

    The original movie frames remain exact pixel targets. Each access applies
    independently sampled Stage-1 Neural Preset LUTs to the two frames. For
    validation, the samples are deterministic for a given pair index.
    """

    def __init__(
        self,
        manifest_path: str | Path,
        lut_root: str | Path,
        image_size: int,
        train: bool,
        validation_seed: int,
        max_corruption_clipped_fraction: float,
        max_corruption_resample_attempts: int,
    ) -> None:
        self.manifest_path = Path(manifest_path).expanduser().resolve()
        self.lut_root = Path(lut_root).expanduser().resolve()
        self.image_size = int(image_size)
        self.train = bool(train)
        self.validation_seed = int(validation_seed)
        self.max_clipped_fraction = float(max_corruption_clipped_fraction)
        self.max_resample_attempts = max(int(max_corruption_resample_attempts), 1)

        if not self.manifest_path.is_file():
            raise FileNotFoundError(f"MovieNet pair manifest not found: {self.manifest_path}")
        if self.image_size <= 0:
            raise ValueError("data.image_size must be positive")

        self.record_offsets = self._index_manifest(self.manifest_path)
        self._manifest_handle: BinaryIO | None = None
        if not self.record_offsets:
            raise RuntimeError(f"No pairs found in {self.manifest_path}")

        self.lut_paths = sorted(self.lut_root.rglob("*.cube"))
        if not self.lut_paths:
            raise RuntimeError(f"No .cube LUT files found below {self.lut_root}")

        # Spawn workers receive these pickle-safe LUTs without inheriting
        # any CUDA/NCCL process state from the training rank.
        self.luts = [load_cube_file(str(path)) for path in self.lut_paths]

    @staticmethod
    def _index_manifest(path: Path) -> array:
        """Store compact byte offsets instead of 800k Python dictionaries."""
        offsets = array("Q")
        with path.open("rb") as handle:
            while True:
                offset = handle.tell()
                line = handle.readline()
                if not line:
                    break
                if line.strip():
                    offsets.append(offset)
        return offsets

    def __getstate__(self) -> dict[str, Any]:
        state = self.__dict__.copy()
        state["_manifest_handle"] = None
        return state

    def _record_at(self, index: int) -> dict[str, Any]:
        if self._manifest_handle is None or self._manifest_handle.closed:
            self._manifest_handle = self.manifest_path.open("rb")
        self._manifest_handle.seek(int(self.record_offsets[index]))
        line = self._manifest_handle.readline()
        try:
            source = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"Invalid JSON at pair index {index} in {self.manifest_path}"
            ) from exc
        missing = [key for key in ("image_a", "image_b") if not source.get(key)]
        if missing:
            raise ValueError(
                f"Missing {missing} at pair index {index} in {self.manifest_path}"
            )
        return {
            "pair_id": str(source.get("pair_id", index)),
            "movie_id": str(source.get("movie_id", "")),
            "image_a": str(source["image_a"]),
            "image_b": str(source["image_b"]),
            "proxy_style_score": float(source.get("proxy_style_score", 0.0)),
            "csd_style_similarity": float(source.get("csd_style_similarity", 0.0)),
            "content_dissimilarity": float(source.get("content_dissimilarity", 0.0)),
        }

    def __len__(self) -> int:
        return len(self.record_offsets)

    def _load_frame(self, path: str) -> Image.Image:
        image_path = Path(path)
        if not image_path.is_file():
            raise FileNotFoundError(f"MovieNet frame not found: {image_path}")
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            image = image.resize(
                (self.image_size, self.image_size),
                Image.Resampling.BICUBIC,
            )
        return image

    @staticmethod
    def _newly_clipped_fraction(original: Image.Image, corrupted: Image.Image) -> float:
        """Fraction of channels newly driven to an 8-bit endpoint by a LUT."""
        source = np.asarray(original, dtype=np.uint8)
        result = np.asarray(corrupted, dtype=np.uint8)
        newly_low = (result <= 1) & (source > 1)
        newly_high = (result >= 254) & (source < 254)
        return float(np.logical_or(newly_low, newly_high).mean())

    def _validation_rng(self, index: int, direction: int) -> np.random.Generator:
        # Use widely separated deterministic streams for A and B.
        seed = (
            self.validation_seed
            + int(index) * 2_000_003
            + int(direction) * 1_000_003
        ) % (2**63 - 1)
        return np.random.default_rng(seed)

    def _corrupt(
        self,
        image: Image.Image,
        index: int,
        direction: int,
    ) -> tuple[Image.Image, str, float]:
        rng = None if self.train else self._validation_rng(index, direction)
        best_image: Image.Image | None = None
        best_lut_index = -1
        best_fraction = float("inf")

        for _ in range(self.max_resample_attempts):
            if rng is None:
                lut_index = int(np.random.randint(0, len(self.luts)))
            else:
                lut_index = int(rng.integers(0, len(self.luts)))
            candidate = image.filter(self.luts[lut_index])
            clipped_fraction = self._newly_clipped_fraction(image, candidate)
            if clipped_fraction < best_fraction:
                best_image = candidate
                best_lut_index = lut_index
                best_fraction = clipped_fraction
            if clipped_fraction <= self.max_clipped_fraction:
                break

        if best_image is None or best_lut_index < 0:
            raise RuntimeError("Failed to sample a corruption LUT")
        return best_image, str(self.lut_paths[best_lut_index]), best_fraction

    @staticmethod
    def _to_tensor(image: Image.Image) -> torch.Tensor:
        return pil_to_tensor(image).float().div_(255.0)

    def __getitem__(self, index: int) -> dict[str, Any]:
        index = int(index)
        record = self._record_at(index)
        frame_a_image = self._load_frame(record["image_a"])
        frame_b_image = self._load_frame(record["image_b"])
        corrupted_a_image, lut_a, clipped_a = self._corrupt(
            frame_a_image, index=int(index), direction=0
        )
        corrupted_b_image, lut_b, clipped_b = self._corrupt(
            frame_b_image, index=int(index), direction=1
        )

        return {
            "frame_a": self._to_tensor(frame_a_image),
            "frame_b": self._to_tensor(frame_b_image),
            "corrupted_a": self._to_tensor(corrupted_a_image),
            "corrupted_b": self._to_tensor(corrupted_b_image),
            "pair_id": record["pair_id"],
            "movie_id": record["movie_id"],
            "image_a": record["image_a"],
            "image_b": record["image_b"],
            "corruption_lut_a": lut_a,
            "corruption_lut_b": lut_b,
            "corruption_clipped_fraction_a": clipped_a,
            "corruption_clipped_fraction_b": clipped_b,
            "proxy_style_score": record["proxy_style_score"],
            "csd_style_similarity": record["csd_style_similarity"],
            "content_dissimilarity": record["content_dissimilarity"],
        }


def _seed_worker(worker_id: int) -> None:
    del worker_id
    worker_seed = int(torch.initial_seed() % (2**32))
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _manifest_for_split(cfg: Any, split: str) -> Path:
    attribute = {
        "train": "train_manifest",
        "valid": "val_manifest",
        "val": "val_manifest",
        "test": "test_manifest",
    }.get(str(split))
    if attribute is None:
        raise ValueError(f"Unsupported MovieNet pair split: {split}")
    pair_root = Path(str(cfg.data.pair_root)).expanduser()
    manifest = Path(str(getattr(cfg.data, attribute))).expanduser()
    return manifest if manifest.is_absolute() else pair_root / manifest


def build_loader(
    cfg: Any,
    split: str,
    train: bool,
    rank: int = 0,
    world_size: int = 1,
) -> tuple[DataLoader, MovieNetPairDataset]:
    dataset = MovieNetPairDataset(
        manifest_path=_manifest_for_split(cfg, split),
        lut_root=str(cfg.data.lut_root),
        image_size=int(cfg.data.image_size),
        train=bool(train),
        validation_seed=int(cfg.data.validation_seed),
        max_corruption_clipped_fraction=float(
            cfg.data.max_corruption_clipped_fraction
        ),
        max_corruption_resample_attempts=int(
            cfg.data.max_corruption_resample_attempts
        ),
    )
    workers = int(cfg.data.num_workers)
    distributed = int(world_size) > 1
    sampler = None
    if distributed:
        sampler = ResumableDistributedSampler(
            dataset,
            num_replicas=int(world_size),
            rank=int(rank),
            shuffle=bool(train),
            seed=int(cfg.experiment.seed),
            drop_last=False,
        )
    generator = torch.Generator()
    generator.manual_seed(
        int(cfg.experiment.seed)
        + int(rank) * 100_003
        + (0 if train else 1)
    )
    batch_size = int(
        getattr(cfg.data, "batch_size_per_gpu", cfg.data.batch_size)
        if distributed
        else cfg.data.batch_size
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=bool(train) and sampler is None,
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=workers > 0 and bool(cfg.data.persistent_workers),
        worker_init_fn=_seed_worker,
        generator=generator,
        multiprocessing_context="spawn" if workers > 0 else None,
        drop_last=False,
    )
    return loader, dataset
