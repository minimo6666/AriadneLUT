from __future__ import annotations

import argparse
import copy
from contextlib import nullcontext
from datetime import timedelta
from itertools import islice
import json
import math
import os
from pathlib import Path
import shutil
import sys

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler, Sampler
from tqdm import tqdm


SOURCE_DIR = Path(__file__).resolve().parent
EXPERIMENT_ROOT = SOURCE_DIR.parents[2]
REPO_ROOT = EXPERIMENT_ROOT.parents[3]
EXPERIMENT_DIR = SOURCE_DIR
LUT_BASELINE_DIR = (
    REPO_ROOT / "experiments/stage_1/ariadnet_lut_neural_preset_setting"
)
for path in (REPO_ROOT, LUT_BASELINE_DIR, EXPERIMENT_DIR):
    value = str(path)
    while value in sys.path:
        sys.path.remove(value)
    sys.path.insert(0, value)

from metrics import psnr  # noqa: E402
from neural_preset_dataset import (  # noqa: E402
    NeuralPresetCOCOLUTPairs,
    _seed_worker,
)
from neural_preset_loss_tonal_chromatic import TonalChromaticStage1Objective  # noqa: E402
from ariadne_tonal_chromatic import AriadneTonalChromaticStage1  # noqa: E402
from utils.config import load_config  # noqa: E402
from utils.image_v2 import save_labeled_grid_v2  # noqa: E402
from utils.logger import ScalarLogger  # noqa: E402
from utils.seed import seed_everything  # noqa: E402


BASE_CONFIG = (
    REPO_ROOT
    / "experiments/stage_1/ariadnet_lut_neural_preset_setting/config.yaml"
)
DEFAULT_OUTPUT = str(EXPERIMENT_ROOT / "_scratch/baseline")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Tonal-Chromatic Stage-1 with DDP."
    )
    parser.add_argument("--config", default=str(BASE_CONFIG))
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--global-batch-size", type=int, default=0)
    parser.add_argument("--batch-size-per-gpu", type=int, default=0)
    parser.add_argument("--num-workers-per-gpu", type=int, default=-1)
    parser.add_argument("--learning-rate", type=float, default=0.0)
    parser.add_argument("--lambda-tonal", type=float, default=0.5)
    parser.add_argument("--lambda-chromatic", type=float, default=1.0)
    parser.add_argument("--lambda-leakage", type=float, default=0.5)
    parser.add_argument("--lambda-slider", type=float, default=0.5)
    parser.add_argument("--lambda-perceptual", type=float, default=0.03)
    parser.add_argument("--perceptual-size", type=int, default=128)
    parser.add_argument("--lambda-lut-smooth", type=float, default=0.02)
    parser.add_argument("--lambda-gamut", type=float, default=0.02)
    parser.add_argument("--chromatic-scale-multiplier", type=float, default=0.5)
    parser.add_argument(
        "--ablation",
        choices=(
            "baseline",
            "vivid_tail_loss",
            "chroma_aware_pool",
            "sanity_filtered_data",
        ),
        default="baseline",
        help="Exactly one controlled intervention; baseline preserves the original run.",
    )
    parser.add_argument("--lambda-vivid-tail", type=float, default=1.0)
    parser.add_argument("--vivid-tail-quantile", type=float, default=0.90)
    parser.add_argument("--chroma-pool-blend", type=float, default=0.50)
    parser.add_argument("--sanity-max-attempts", type=int, default=8)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def unwrap_model(model: nn.Module) -> AriadneTonalChromaticStage1:
    unwrapped = model.module if isinstance(model, DistributedDataParallel) else model
    if not isinstance(unwrapped, AriadneTonalChromaticStage1):
        raise TypeError(f"Unexpected model type: {type(unwrapped)!r}")
    return unwrapped


class DistributedEvalSampler(Sampler[int]):
    """Shard validation without DistributedSampler's duplicate padding."""

    def __init__(self, dataset, rank: int, world_size: int) -> None:
        self.dataset = dataset
        self.rank = int(rank)
        self.world_size = int(world_size)

    def __iter__(self):
        return iter(range(self.rank, len(self.dataset), self.world_size))

    def __len__(self) -> int:
        remaining = len(self.dataset) - self.rank
        return max((remaining + self.world_size - 1) // self.world_size, 0)


def prepare_paths(cfg, config_path: str, is_main: bool) -> dict[str, Path]:
    root = resolve_path(str(cfg.experiment.output_dir))
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    if is_main:
        for path in paths.values():
            path.mkdir(parents=True, exist_ok=True)
        shutil.copy2(config_path, root / "config.yaml")
        unexpected = sorted(
            path.name
            for path in paths["checkpoints"].glob("*.pth")
            if path.name not in {"best.pth", "last.pth"}
        )
        if unexpected:
            raise RuntimeError(
                "Checkpoint directory contains forbidden per-epoch files: "
                f"{unexpected}"
            )
    return paths


def build_loader(
    cfg,
    split: str,
    train: bool,
    rank: int,
    world_size: int,
    shared_lut_dataset: NeuralPresetCOCOLUTPairs | None = None,
) -> tuple[DataLoader, NeuralPresetCOCOLUTPairs]:
    if shared_lut_dataset is not None:
        dataset = copy.copy(shared_lut_dataset)
        dataset.image_paths = sorted(
            (Path(str(cfg.data.coco_root)) / split).glob("*.jpg")
        )
        dataset.image_size = int(cfg.data.image_size)
        dataset.deterministic = (
            bool(cfg.data.deterministic_validation) if not train else False
        )
        dataset.seed = int(cfg.data.validation_seed)
        dataset.sanity_filter = (
            bool(getattr(cfg.data, "sanity_filter", False)) and bool(train)
        )
        dataset.sanity_max_attempts = int(
            getattr(cfg.data, "sanity_max_attempts", 8)
        )
    else:
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
    distributed = world_size > 1
    if train and distributed:
        sampler: Sampler[int] | None = DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            seed=int(cfg.experiment.seed),
            drop_last=False,
        )
    elif not train and distributed:
        sampler = DistributedEvalSampler(dataset, rank, world_size)
    else:
        sampler = None

    generator = torch.Generator()
    generator.manual_seed(int(cfg.experiment.seed) + int(rank))
    workers = int(cfg.data.num_workers_per_gpu)
    loader = DataLoader(
        dataset,
        batch_size=int(cfg.data.batch_size_per_gpu),
        shuffle=bool(train and sampler is None),
        sampler=sampler,
        num_workers=workers,
        pin_memory=bool(cfg.data.pin_memory),
        persistent_workers=(
            workers > 0 and train and bool(cfg.data.persistent_workers)
        ),
        prefetch_factor=(int(cfg.data.prefetch_factor) if workers > 0 else None),
        worker_init_fn=_seed_worker,
        generator=generator,
        drop_last=False,
    )
    return loader, dataset


def move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=True)


def scalar_dict(values: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in values.items()
    }


def add_dataset_diagnostics(values: dict[str, float], batch: dict) -> None:
    for key in ("sanity_attempts_mean", "sanity_fallback_ratio"):
        if key not in batch:
            continue
        value = batch[key]
        if torch.is_tensor(value):
            values[key] = float(value.float().mean())
        else:
            values[key] = float(value)


def add_metrics(
    values: dict[str, float],
    pair_result: dict,
    image_a: torch.Tensor,
    image_b: torch.Tensor,
) -> None:
    output_ab = pair_result["output_ab"]
    output_ba = pair_result["output_ba"]
    canonical_a = pair_result["state_a"]["canonical"]
    canonical_b = pair_result["state_b"]["canonical"]
    values.update(
        {
            "psnr_a_to_b": float(psnr(output_ab.clamp(0, 1), image_b)),
            "psnr_b_to_a": float(psnr(output_ba.clamp(0, 1), image_a)),
            "canonical_psnr": float(
                psnr(canonical_a.clamp(0, 1), canonical_b.clamp(0, 1))
            ),
            "tonal_only_psnr_a_to_b": float(
                psnr(pair_result["output_ab_tonal_only"].clamp(0, 1), image_b)
            ),
            "tonal_only_psnr_b_to_a": float(
                psnr(pair_result["output_ba_tonal_only"].clamp(0, 1), image_a)
            ),
            "chroma_only_psnr_a_to_b": float(
                psnr(pair_result["output_ab_chroma_only"].clamp(0, 1), image_b)
            ),
            "chroma_only_psnr_b_to_a": float(
                psnr(pair_result["output_ba_chroma_only"].clamp(0, 1), image_a)
            ),
            "output_out_of_range_ratio": float(
                torch.cat([output_ab, output_ba])
                .sub(0.5).abs().gt(0.5).float().mean()
            ),
        }
    )
    values["mean_bidirectional_psnr"] = 0.5 * (
        values["psnr_a_to_b"] + values["psnr_b_to_a"]
    )


def accumulate(
    totals: dict[str, float], values: dict[str, float], weight: int
) -> None:
    for key, value in values.items():
        if math.isfinite(float(value)):
            totals[key] = totals.get(key, 0.0) + float(value) * int(weight)


def distributed_average(
    totals: dict[str, float],
    count: int,
    device: torch.device,
    distributed: bool,
) -> dict[str, float]:
    keys = sorted(totals)
    payload = torch.tensor(
        [float(count), *[float(totals[key]) for key in keys]],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    global_count = float(payload[0])
    if global_count <= 0:
        raise RuntimeError("Cannot average zero samples")
    return {
        key: float(payload[index + 1]) / global_count
        for index, key in enumerate(keys)
    }


def save_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_metric: float,
    validation: dict[str, float],
    args: argparse.Namespace,
) -> None:
    payload = {
        "model": unwrap_model(model).state_dict(),
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_metric": float(best_metric),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "protocol": "tonal_chromatic_factorized_stage1_ddp",
        "validation_total": float(validation["total"]),
        "validation_mean_bidirectional_psnr": float(
            validation["mean_bidirectional_psnr"]
        ),
        "chromatic_scale_multiplier": float(args.chromatic_scale_multiplier),
        "lambda_tonal": float(args.lambda_tonal),
        "lambda_chromatic": float(args.lambda_chromatic),
        "lambda_leakage": float(args.lambda_leakage),
        "lambda_slider": float(args.lambda_slider),
        "lambda_perceptual": float(args.lambda_perceptual),
        "lambda_lut_smooth": float(args.lambda_lut_smooth),
        "lambda_gamut": float(args.lambda_gamut),
        "ablation": str(args.ablation),
        "lambda_vivid_tail": (
            float(args.lambda_vivid_tail)
            if args.ablation == "vivid_tail_loss"
            else 0.0
        ),
        "vivid_tail_quantile": float(args.vivid_tail_quantile),
        "style_pooling": (
            "chroma_aware" if args.ablation == "chroma_aware_pool" else "mean"
        ),
        "chroma_pool_blend": float(args.chroma_pool_blend),
        "sanity_filter": args.ablation == "sanity_filtered_data",
        "sanity_filter_train_only": True,
        "sanity_max_attempts": int(args.sanity_max_attempts),
        "sanity_filter_thresholds": dict(
            NeuralPresetCOCOLUTPairs.SANITY_FILTER_THRESHOLDS
        ),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    os.replace(temporary, path)


def load_training_checkpoint(
    path: Path,
    model: AriadneTonalChromaticStage1,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    device: torch.device,
) -> dict:
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"], strict=True)
    optimizer.load_state_dict(checkpoint["optimizer"])
    scheduler.load_state_dict(checkpoint["scheduler"])
    scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def write_manifest(
    cfg,
    paths: dict[str, Path],
    train_dataset: NeuralPresetCOCOLUTPairs,
    val_dataset: NeuralPresetCOCOLUTPairs,
    world_size: int,
    accumulation: int,
    args: argparse.Namespace,
) -> None:
    payload = {
        "protocol": "Tonal-Chromatic factorized Stage-1 DDP",
        "controlled_baseline": str(LUT_BASELINE_DIR),
        "base_config": str(BASE_CONFIG),
        "coco_root": str(cfg.data.coco_root),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "lut_root": str(cfg.data.lut_root),
        "lut_count": len(train_dataset.lut_paths),
        "seed": int(cfg.experiment.seed),
        "world_size": int(world_size),
        "batch_size_per_gpu": int(cfg.data.batch_size_per_gpu),
        "gradient_accumulation_steps": int(accumulation),
        "global_batch_size": int(cfg.data.global_batch_size),
        "learning_rate": float(cfg.train.learning_rate),
        "lambda_tonal": float(args.lambda_tonal),
        "lambda_chromatic": float(args.lambda_chromatic),
        "lambda_leakage": float(args.lambda_leakage),
        "lambda_slider": float(args.lambda_slider),
        "lambda_perceptual": float(args.lambda_perceptual),
        "perceptual_size": int(args.perceptual_size),
        "lambda_lut_smooth": float(args.lambda_lut_smooth),
        "lambda_gamut": float(args.lambda_gamut),
        "chromatic_scale_multiplier": float(args.chromatic_scale_multiplier),
        "ablation": str(args.ablation),
        "lambda_vivid_tail": (
            float(args.lambda_vivid_tail)
            if args.ablation == "vivid_tail_loss"
            else 0.0
        ),
        "vivid_tail_quantile": float(args.vivid_tail_quantile),
        "style_pooling": (
            "chroma_aware" if args.ablation == "chroma_aware_pool" else "mean"
        ),
        "chroma_pool_blend": float(args.chroma_pool_blend),
        "sanity_filter": args.ablation == "sanity_filtered_data",
        "sanity_filter_train_only": True,
        "sanity_max_attempts": int(args.sanity_max_attempts),
        "sanity_filter_thresholds": dict(
            NeuralPresetCOCOLUTPairs.SANITY_FILTER_THRESHOLDS
        ),
        "checkpoint_policy": ["best.pth", "last.pth"],
        "best_criterion": "max validation mean bidirectional PSNR",
    }
    (paths["root"] / "data_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


@torch.no_grad()
def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: TonalChromaticStage1Objective,
    cfg,
    device: torch.device,
    epoch: int,
    paths: dict[str, Path],
    max_batches: int,
    distributed: bool,
    is_main: bool,
) -> dict[str, float]:
    base_model = unwrap_model(model)
    base_model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    first_visual = None
    iterator = loader if max_batches <= 0 else islice(loader, max_batches)
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"

    for batch in tqdm(
        iterator,
        desc=f"ValTC-DDP {epoch:03d}",
        total=(max_batches or len(loader)),
        disable=not is_main,
    ):
        content = move_tensor(batch["content"], device)
        image_a = move_tensor(batch["image_a"], device)
        image_b = move_tensor(batch["image_b"], device)
        alpha_t = torch.full(
            (image_a.shape[0],), 0.5, device=device, dtype=image_a.dtype
        )
        alpha_c = torch.full(
            (image_a.shape[0],), 0.5, device=device, dtype=image_a.dtype
        )
        with autocast(device_type=device.type, enabled=amp_enabled):
            pair_result = base_model.forward_pair(
                image_a,
                image_b,
                controlled_tonal_strength=alpha_t,
                controlled_chromatic_strength=alpha_c,
            )
            losses = criterion(pair_result, image_a, image_b)
        values = scalar_dict(losses)
        add_dataset_diagnostics(values, batch)
        add_metrics(values, pair_result, image_a, image_b)
        batch_size = int(image_a.shape[0])
        accumulate(totals, values, batch_size)
        sample_count += batch_size
        if is_main and first_visual is None:
            first_visual = (content, image_a, image_b, pair_result)

    metrics = distributed_average(totals, sample_count, device, distributed)
    if is_main and first_visual is not None:
        content, image_a, image_b, pair_result = first_visual
        save_labeled_grid_v2(
            tensors=(
                content,
                image_a,
                image_b,
                pair_result["output_ab"],
                image_b,
                pair_result["output_ab_tonal_only"],
                pair_result["output_ab_chroma_only"],
                pair_result["output_ba"],
                image_a,
            ),
            path=paths["images"] / f"val_epoch_{epoch:04d}.png",
            labels=(
                "original_COCO",
                "LUT_A",
                "LUT_B",
                "A_to_B_full",
                "GT_B",
                "A_to_B_tonal_only",
                "A_to_B_chroma_only",
                "B_to_A_full",
                "GT_A",
            ),
            title=f"Tonal-Chromatic DDP - epoch {epoch}",
            max_items=int(cfg.validation.max_visual_items),
        )
    return metrics


def main() -> None:
    args = parse_args()
    config_path = str(resolve_path(args.config))
    cfg = load_config(config_path)
    cfg.experiment.output_dir = args.output_dir
    cfg.model.chromatic_scale_multiplier = float(args.chromatic_scale_multiplier)
    cfg.model.style_pooling = (
        "chroma_aware" if args.ablation == "chroma_aware_pool" else "mean"
    )
    cfg.model.style_chroma_pool_blend = float(args.chroma_pool_blend)
    cfg.data.sanity_filter = args.ablation == "sanity_filtered_data"
    cfg.data.sanity_max_attempts = int(args.sanity_max_attempts)
    if args.resume:
        cfg.train.resume = args.resume
    if args.epochs > 0:
        cfg.train.epochs = args.epochs
    if args.learning_rate > 0.0:
        cfg.train.learning_rate = args.learning_rate

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    is_main = rank == 0
    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            timeout=timedelta(minutes=10),
        )
    else:
        if str(cfg.device).startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(str(cfg.device))

    local_batch = (
        int(args.batch_size_per_gpu)
        if args.batch_size_per_gpu > 0
        else int(cfg.data.batch_size)
    )
    global_batch = (
        int(args.global_batch_size)
        if args.global_batch_size > 0
        else local_batch * world_size
    )
    workers_per_gpu = (
        int(args.num_workers_per_gpu)
        if args.num_workers_per_gpu >= 0
        else max(1, int(cfg.data.num_workers) // max(world_size, 1))
    )
    cfg.data.batch_size_per_gpu = local_batch
    cfg.data.global_batch_size = global_batch
    cfg.data.num_workers_per_gpu = workers_per_gpu
    cfg.data.prefetch_factor = 2
    cfg.data.persistent_workers = workers_per_gpu > 0

    seed_everything(int(cfg.experiment.seed) + rank)
    paths = prepare_paths(cfg, config_path, is_main)
    if distributed:
        dist.barrier(device_ids=[local_rank])

    samples_per_micro_step = local_batch * world_size
    if global_batch % samples_per_micro_step != 0:
        raise ValueError(
            f"global_batch_size={global_batch} must be divisible by "
            f"batch_size_per_gpu*world_size={samples_per_micro_step}"
        )
    accumulation = global_batch // samples_per_micro_step
    if accumulation <= 0:
        raise ValueError("Gradient accumulation must be positive")

    resume_value = str(cfg.train.resume)
    last_path = paths["checkpoints"] / "last.pth"
    if last_path.exists() and not resume_value:
        raise RuntimeError(
            f"Refusing to overwrite {last_path}; pass --resume {last_path}"
        )

    rank_print = print if is_main else lambda *values, **kwargs: None
    rank_print("Loading matched COCO/LUT training split...", flush=True)
    train_loader, train_dataset = build_loader(
        cfg, str(cfg.data.train_split), True, rank, world_size
    )
    rank_print("Loading deterministic validation split...", flush=True)
    val_loader, val_dataset = build_loader(
        cfg,
        str(cfg.data.val_split),
        False,
        rank,
        world_size,
        shared_lut_dataset=train_dataset,
    )

    base_model = AriadneTonalChromaticStage1(cfg.model).to(device)
    criterion = TonalChromaticStage1Objective(
        lambda_consistency=float(cfg.loss.lambda_consistency),
        lambda_tonal=args.lambda_tonal,
        lambda_chromatic=args.lambda_chromatic,
        lambda_leakage=args.lambda_leakage,
        lambda_slider=args.lambda_slider,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lut_smooth=args.lambda_lut_smooth,
        lambda_gamut=args.lambda_gamut,
        lambda_vivid_tail=(
            args.lambda_vivid_tail
            if args.ablation == "vivid_tail_loss"
            else 0.0
        ),
        vivid_tail_quantile=args.vivid_tail_quantile,
        perceptual_size=args.perceptual_size,
    ).to(device)
    optimizer = torch.optim.Adam(
        base_model.parameters(),
        lr=float(cfg.train.learning_rate),
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(cfg.train.scheduler_step_size),
        gamma=float(cfg.train.scheduler_gamma),
    )
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    global_step = 0
    best_metric = float("-inf")
    if resume_value:
        checkpoint = load_training_checkpoint(
            resolve_path(resume_value),
            base_model,
            optimizer,
            scheduler,
            scaler,
            device,
        )
        checkpoint_ablation = str(checkpoint.get("ablation", "baseline"))
        if checkpoint_ablation != args.ablation:
            raise ValueError(
                "Refusing to resume a different intervention: "
                f"checkpoint={checkpoint_ablation!r}, requested={args.ablation!r}"
            )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))

    model: nn.Module = base_model
    if distributed:
        model = DistributedDataParallel(
            base_model,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
            gradient_as_bucket_view=True,
        )

    if is_main:
        write_manifest(
            cfg,
            paths,
            train_dataset,
            val_dataset,
            world_size,
            accumulation,
            args,
        )
    if distributed:
        dist.barrier(device_ids=[local_rank])

    total_parameters = sum(p.numel() for p in base_model.parameters())
    rank_print(f"Parameters={total_parameters:,}")
    rank_print(
        f"DDP world_size={world_size}; per_gpu_batch={local_batch}; "
        f"accumulation={accumulation}; global_batch={global_batch}; "
        f"learning_rate={float(cfg.train.learning_rate):.8f}"
    )
    rank_print(
        f"train_images={len(train_dataset):,}; val_images={len(val_dataset):,}; "
        f"LUTs={len(train_dataset.lut_paths)}; workers_per_gpu={workers_per_gpu}"
    )
    rank_print(
        f"ablation={args.ablation}; "
        f"lambda_vivid_tail={criterion.lambda_vivid_tail}; "
        f"style_pooling={cfg.model.style_pooling}; "
        f"sanity_filter={cfg.data.sanity_filter}"
    )

    logger = ScalarLogger(paths["logs"]) if is_main else None
    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            if device.type == "cuda":
                torch.cuda.reset_peak_memory_stats(device)
            sampler = train_loader.sampler
            if isinstance(sampler, DistributedSampler):
                sampler.set_epoch(epoch)
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_totals: dict[str, float] = {}
            train_samples = 0
            total_batches = len(train_loader)
            if args.max_train_batches > 0:
                total_batches = min(total_batches, args.max_train_batches)
            iterator = train_loader if args.max_train_batches <= 0 else islice(
                train_loader, args.max_train_batches
            )
            progress = tqdm(
                iterator,
                total=total_batches,
                desc=f"TrainTC-DDP {epoch:03d}",
                disable=not is_main,
                miniters=250,
                mininterval=60.0,
            )
            current_target = accumulation

            for batch_index, batch in enumerate(progress):
                if batch_index % accumulation == 0:
                    current_target = min(accumulation, total_batches - batch_index)
                should_step = (
                    (batch_index + 1) % accumulation == 0
                    or batch_index + 1 == total_batches
                )
                sync_context = (
                    model.no_sync()
                    if distributed and not should_step
                    else nullcontext()
                )
                content = move_tensor(batch["content"], device)
                image_a = move_tensor(batch["image_a"], device)
                image_b = move_tensor(batch["image_b"], device)
                alpha_t = torch.rand(
                    image_a.shape[0], device=device, dtype=image_a.dtype
                )
                alpha_c = torch.rand(
                    image_a.shape[0], device=device, dtype=image_a.dtype
                )
                with sync_context:
                    with autocast(device_type=device.type, enabled=amp_enabled):
                        pair_result = model(
                            image_a,
                            image_b,
                            _forward_pair=True,
                            controlled_tonal_strength=alpha_t,
                            controlled_chromatic_strength=alpha_c,
                        )
                        losses = criterion(pair_result, image_a, image_b)
                        scaled_loss = losses["total"] / current_target
                    scaler.scale(scaled_loss).backward()

                if should_step:
                    scaler.unscale_(optimizer)
                    if float(cfg.train.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            base_model.parameters(), float(cfg.train.grad_clip)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                values = scalar_dict(losses)
                add_dataset_diagnostics(values, batch)
                add_metrics(values, pair_result, image_a, image_b)
                batch_size = int(image_a.shape[0])
                accumulate(train_totals, values, batch_size)
                train_samples += batch_size
                postfix = {
                    "total": f"{values['total']:.4f}",
                    "psnr": f"{values['mean_bidirectional_psnr']:.2f}",
                    "tone": f"{values['tonal_match_loss']:.3f}",
                    "chroma": f"{values['chromatic_match_loss']:.3f}",
                    "slider": f"{values['slider_supervision_loss']:.3f}",
                    "step": global_step,
                }
                if args.ablation == "vivid_tail_loss":
                    postfix["vivid"] = f"{values['vivid_tail_loss']:.3f}"
                if args.ablation == "sanity_filtered_data":
                    postfix["attempts"] = (
                        f"{values['sanity_attempts_mean']:.2f}"
                    )
                    postfix["fallback"] = (
                        f"{values['sanity_fallback_ratio']:.3f}"
                    )
                progress.set_postfix(
                    postfix,
                    refresh=False,
                )
                if is_main and should_step and global_step % 250 == 0:
                    progress.write(
                        f"Progress epoch={epoch:03d} batch={batch_index + 1}/{total_batches} "
                        f"step={global_step} total={values['total']:.4f} "
                        f"psnr={values['mean_bidirectional_psnr']:.2f}"
                    )

                if (
                    is_main
                    and should_step
                    and global_step % int(cfg.train.log_every_optimizer_steps) == 0
                ):
                    logger.log(
                        {**values, "lr": optimizer.param_groups[0]["lr"]},
                        global_step,
                        "train",
                    )
                if (
                    is_main
                    and should_step
                    and global_step % int(cfg.train.image_every_optimizer_steps) == 0
                ):
                    save_labeled_grid_v2(
                        tensors=(
                            content,
                            image_a,
                            image_b,
                            pair_result["output_ab"],
                            image_b,
                            pair_result["output_ab_tonal_only"],
                            pair_result["output_ab_chroma_only"],
                        ),
                        path=paths["images"] / f"train_step_{global_step:08d}.png",
                        labels=(
                            "original_COCO",
                            "LUT_A",
                            "LUT_B",
                            "A_to_B_full",
                            "GT_B",
                            "A_to_B_tonal_only",
                            "A_to_B_chroma_only",
                        ),
                        title=f"Tonal-Chromatic DDP step {global_step}",
                        max_items=int(cfg.validation.max_visual_items),
                    )

            train_values = distributed_average(
                train_totals, train_samples, device, distributed
            )
            if is_main:
                logger.log(train_values, global_step, "train_epoch")

            val_values = validate(
                model,
                val_loader,
                criterion,
                cfg,
                device,
                epoch,
                paths,
                args.max_val_batches,
                distributed,
                is_main,
            )
            if device.type == "cuda":
                peak_memory = torch.tensor(
                    torch.cuda.max_memory_allocated(device) / (1024**3),
                    dtype=torch.float64,
                    device=device,
                )
                if distributed:
                    dist.all_reduce(peak_memory, op=dist.ReduceOp.MAX)
                val_values["peak_gpu_memory_gib"] = float(peak_memory)
            scheduler.step()
            if is_main:
                logger.log(val_values, global_step, "val")
                rank_print(
                    "Validation:",
                    " ".join(
                        f"{key}={value:.6f}"
                        for key, value in val_values.items()
                    ),
                    flush=True,
                )
                current_metric = float(val_values["mean_bidirectional_psnr"])
                if current_metric > best_metric:
                    best_metric = current_metric
                    save_checkpoint(
                        paths["checkpoints"] / "best.pth",
                        model,
                        optimizer,
                        scheduler,
                        scaler,
                        epoch,
                        global_step,
                        best_metric,
                        val_values,
                        args,
                    )
                save_checkpoint(
                    last_path,
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_metric,
                    val_values,
                    args,
                )
                checkpoint_names = sorted(
                    p.name for p in paths["checkpoints"].glob("*.pth")
                )
                if any(
                    name not in {"best.pth", "last.pth"}
                    for name in checkpoint_names
                ):
                    raise RuntimeError(
                        f"Unexpected checkpoints: {checkpoint_names}"
                    )
                rank_print(
                    f"Checkpoint policy OK: {checkpoint_names}; "
                    f"best_validation_psnr={best_metric:.6f}",
                    flush=True,
                )
            if distributed:
                dist.barrier(device_ids=[local_rank])
    finally:
        if logger is not None:
            logger.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
