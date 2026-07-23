from __future__ import annotations

from typing import Any

from torch.utils.data import DataLoader
import webdataset as wds

from .common import find_tars
from .transforms import make_transform


def _resolve_num_workers_v2(
    requested_workers: int,
    number_of_shards: int,
    phase: str,
) -> int:
    """
    Prevent workers from receiving an empty shard list.

    This is important for the user's current dataset layout, where the
    validation split has fewer shards than the training split.
    """
    requested_workers = max(
        int(requested_workers),
        0,
    )
    number_of_shards = max(
        int(number_of_shards),
        1,
    )

    if requested_workers == 0:
        return 0

    resolved_workers = min(
        requested_workers,
        number_of_shards,
    )

    if resolved_workers != requested_workers:
        print(
            f"[COCO WebDataset V2:{phase}] "
            f"Reducing num_workers from "
            f"{requested_workers} to "
            f"{resolved_workers}, because only "
            f"{number_of_shards} shard(s) were found."
        )

    return resolved_workers


def _build_webdataset_v2(
    tar_files: list[str],
    train: bool,
    shuffle_buffer: int,
) -> Any:
    common_arguments = dict(
        resampled=bool(train),
        shardshuffle=False,
        nodesplitter=wds.split_by_node,
        workersplitter=wds.split_by_worker,
        handler=wds.warn_and_continue,
    )

    # Newer WebDataset versions support empty_check=False. The worker limit
    # above already prevents empty workers, while this remains an additional
    # safeguard when available.
    try:
        dataset = wds.WebDataset(
            tar_files,
            empty_check=False,
            **common_arguments,
        )
    except TypeError:
        dataset = wds.WebDataset(
            tar_files,
            **common_arguments,
        )

    if train and int(shuffle_buffer) > 0:
        dataset = dataset.shuffle(
            int(shuffle_buffer)
        )

    return dataset


def get_coco_loader_v2(
    cfg,
    phase: str = "train",
) -> DataLoader:
    train = phase == "train"

    split = (
        cfg.data.train_split
        if train
        else cfg.data.val_split
    )

    tar_files = find_tars(
        cfg.data.coco_root,
        split,
    )

    number_of_workers = _resolve_num_workers_v2(
        requested_workers=int(
            cfg.data.num_workers
        ),
        number_of_shards=len(tar_files),
        phase=phase,
    )

    transform = make_transform(
        image_size=int(cfg.data.image_size),
        train=train,
    )

    dataset = _build_webdataset_v2(
        tar_files=tar_files,
        train=train,
        shuffle_buffer=int(
            cfg.data.shuffle_buffer
        ),
    )

    dataset = (
        dataset
        .decode("pil")
        .to_tuple("jpg;png")
        .map_tuple(transform)
        .batched(
            int(cfg.data.batch_size),
            partial=not train,
        )
    )

    return DataLoader(
        dataset,
        batch_size=None,
        num_workers=number_of_workers,
        pin_memory=True,
        persistent_workers=(
            number_of_workers > 0
        ),
    )
