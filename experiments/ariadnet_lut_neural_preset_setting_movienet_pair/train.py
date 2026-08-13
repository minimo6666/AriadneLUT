from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import sys
from contextlib import nullcontext
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from tqdm import tqdm

EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from models.ariadne_lut_v2 import AriadneLUTV2
from utils.checkpoint import load_checkpoint
from utils.config import load_config

from loss_movienet_pair import MovieNetPairStage2Objective
from model_movienet_pair import AriadneMovieNetPairStage2
from visualization import save_stage2_grid


DEFAULT_CONFIG = EXPERIMENT_DIR / "config.yaml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MovieNet pair Stage-2 residual-LUT fine-tuning for AriadneLUT."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--resume", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


def seed_everything(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def move_tensor(value, device: torch.device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    return value


def batch_psnr(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    mse = (prediction.float().clamp(0.0, 1.0) - target.float()).square().flatten(1).mean(1)
    return (-10.0 * torch.log10(mse.clamp_min(1e-10))).mean()


def tensor_scalar(value: torch.Tensor | float) -> float:
    if torch.is_tensor(value):
        return float(value.detach().float().cpu().item())
    return float(value)


def require_batch_contract(batch: dict[str, Any]) -> None:
    required = ("frame_a", "frame_b", "corrupted_a", "corrupted_b")
    missing = [key for key in required if key not in batch]
    if missing:
        raise KeyError(
            "MovieNet pair dataset contract is incomplete. Missing keys: "
            f"{missing}. See DATASET_CONTRACT.md."
        )


def make_bidirectional_batch(
    batch: dict[str, Any],
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Build A<-B and B<-A directions in one batched forward.

    Direction 1: corrupted(A) + reference(B) -> GT A
    Direction 2: corrupted(B) + reference(A) -> GT B
    """
    require_batch_contract(batch)
    frame_a = move_tensor(batch["frame_a"], device)
    frame_b = move_tensor(batch["frame_b"], device)
    corrupted_a = move_tensor(batch["corrupted_a"], device)
    corrupted_b = move_tensor(batch["corrupted_b"], device)

    content = torch.cat([corrupted_a, corrupted_b], dim=0)
    style = torch.cat([frame_b, frame_a], dim=0)
    target = torch.cat([frame_a, frame_b], dim=0)
    views = {
        "frame_a": frame_a,
        "frame_b": frame_b,
        "corrupted_a": corrupted_a,
        "corrupted_b": corrupted_b,
    }
    return content, style, target, views


def prepare_paths(cfg: Any) -> dict[str, Path]:
    root = resolve_path(str(cfg.experiment.output_dir))
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    return paths


def residual_strength(global_step: int, ramp_steps: int) -> float:
    ramp = max(int(ramp_steps), 0)
    if ramp == 0:
        return 1.0
    return min(max(float(global_step + 1) / float(ramp), 0.0), 1.0)


def build_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    warmup_steps: int,
    min_lr_ratio: float,
):
    total = max(int(total_steps), 1)
    warmup = min(max(int(warmup_steps), 0), total - 1)
    minimum = float(min_lr_ratio)

    def scale(step: int) -> float:
        if warmup > 0 and step < warmup:
            return max(float(step + 1) / float(warmup), 1e-6)
        progress = (step - warmup) / max(total - warmup, 1)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return minimum + (1.0 - minimum) * cosine

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scale)


def unwrap_stage2_model(model: nn.Module) -> AriadneMovieNetPairStage2:
    wrapped_types = (nn.DataParallel, DistributedDataParallel)
    unwrapped = model.module if isinstance(model, wrapped_types) else model
    if not isinstance(unwrapped, AriadneMovieNetPairStage2):
        raise TypeError(f"Unexpected Stage-2 model type: {type(unwrapped)!r}")
    return unwrapped


def save_stage2_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler,
    scaler: GradScaler,
    epoch: int,
    global_step: int,
    best_psnr: float,
    cfg: Any,
    next_batch_index: int = 0,
    epoch_complete: bool = True,
) -> None:
    if dist.is_available() and dist.is_initialized() and dist.get_rank() != 0:
        return

    payload = {
        "stage": "ariadnet_lut_neural_preset_setting_movienet_pair",
        "epoch": int(epoch),
        "global_step": int(global_step),
        "best_psnr": float(best_psnr),
        "next_batch_index": int(next_batch_index),
        "epoch_complete": bool(epoch_complete),
        "stage1_config": str(cfg.stage1.config),
        "stage1_checkpoint": str(cfg.stage1.checkpoint),
        "stage2_state_dict": unwrap_stage2_model(model).stage2_state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary_path)
    os.replace(temporary_path, path)


def load_stage2_checkpoint(
    path: Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None = None,
    scheduler=None,
    scaler: GradScaler | None = None,
    map_location: str | torch.device = "cpu",
) -> dict[str, Any]:
    checkpoint = torch.load(path, map_location=map_location)
    unwrap_stage2_model(model).load_stage2_state_dict(
        checkpoint["stage2_state_dict"], strict=True
    )
    if optimizer is not None and "optimizer" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer"])
    if scheduler is not None and "scheduler" in checkpoint:
        scheduler.load_state_dict(checkpoint["scheduler"])
    if scaler is not None and "scaler" in checkpoint:
        scaler.load_state_dict(checkpoint["scaler"])
    return checkpoint


def build_loaders_from_dataset_module(
    cfg: Any,
    rank: int = 0,
    world_size: int = 1,
):
    """Build the MovieNet JSONL pair loaders implemented beside this script."""
    try:
        from movienet_pair_dataset import build_loader
    except ImportError as exc:
        raise ImportError(
            "Unable to import movienet_pair_dataset.py."
        ) from exc

    train_loader, train_dataset = build_loader(
        cfg, split="train", train=True, rank=rank, world_size=world_size
    )
    val_loader, val_dataset = build_loader(
        cfg, split="valid", train=False, rank=rank, world_size=world_size
    )
    return train_loader, train_dataset, val_loader, val_dataset


def validation_pass(
    model: nn.Module,
    loader,
    criterion: MovieNetPairStage2Objective,
    cfg: Any,
    device: torch.device,
    epoch: int,
    paths: dict[str, Path],
    max_batches: int,
    distributed: bool,
    is_main_process: bool,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    count = 0
    last_visual = None
    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"

    with torch.no_grad():
        for batch_index, batch in enumerate(loader):
            if max_batches > 0 and batch_index >= max_batches:
                break
            content, style, target, views = make_bidirectional_batch(batch, device)
            with autocast(device_type=device.type, enabled=amp_enabled):
                result = model(content, style, residual_strength=1.0)
                losses = criterion(result, target)

            baseline_psnr = batch_psnr(result["baseline_output"], target)
            refined_psnr = batch_psnr(result["output"], target)
            values = {key: tensor_scalar(value) for key, value in losses.items()}
            values.update(
                {
                    "baseline_psnr": tensor_scalar(baseline_psnr),
                    "refined_psnr": tensor_scalar(refined_psnr),
                    "psnr_gain": tensor_scalar(refined_psnr - baseline_psnr),
                    "delta_abs_mean": tensor_scalar(result["delta_lut"].abs().mean()),
                    "final_lut_oob": tensor_scalar(
                        ((result["final_lut"] < 0.0) | (result["final_lut"] > 1.0))
                        .float()
                        .mean()
                    ),
                }
            )
            batch_size = int(target.shape[0])
            for key, value in values.items():
                totals[key] = totals.get(key, 0.0) + float(value) * batch_size
            count += batch_size
            if is_main_process:
                last_visual = (content, style, target, result)

    if count == 0:
        raise RuntimeError("Validation loader produced no batches")
    metric_keys = sorted(totals)
    payload = torch.tensor(
        [float(count), *[totals[key] for key in metric_keys]],
        dtype=torch.float64,
        device=device,
    )
    if distributed:
        dist.all_reduce(payload, op=dist.ReduceOp.SUM)
    global_count = float(payload[0].item())
    means = {
        key: float(payload[index + 1].item()) / global_count
        for index, key in enumerate(metric_keys)
    }
    if is_main_process and last_visual is not None:
        content, style, target, result = last_visual
        save_stage2_grid(
            tensors=(
                content,
                style,
                result["canonical"],
                result["baseline_output"],
                result["output"],
                target,
            ),
            labels=(
                "corrupted content",
                "movie reference",
                "frozen Z_C",
                "Stage1 baseline",
                "Stage2 refined",
                "movie GT",
            ),
            path=paths["images"] / f"val_epoch_{epoch:04d}.png",
            max_items=int(cfg.validation.max_visual_items),
            title=(
                f"epoch={epoch} baseline={means['baseline_psnr']:.2f}dB "
                f"refined={means['refined_psnr']:.2f}dB gain={means['psnr_gain']:+.2f}dB"
            ),
        )
    return means


def main() -> None:
    args = parse_args()
    cfg = load_config(str(resolve_path(args.config)))
    if args.output_dir:
        cfg.experiment.output_dir = args.output_dir

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    distributed = world_size > 1
    rank = int(os.environ.get("RANK", "0")) if distributed else 0
    local_rank = int(os.environ.get("LOCAL_RANK", "0")) if distributed else 0
    is_main_process = rank == 0

    if distributed:
        if not torch.cuda.is_available():
            raise RuntimeError("DDP with NCCL requires CUDA")
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
        dist.init_process_group(
            backend="nccl",
            device_id=device,
            timeout=timedelta(minutes=3),
        )
    else:
        device_name = str(cfg.device)
        if device_name.startswith("cuda") and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        device = torch.device(device_name)

    seed_everything(int(cfg.experiment.seed) + rank)
    paths = prepare_paths(cfg)
    if distributed:
        dist.barrier()

    stage1_config_path = resolve_path(str(cfg.stage1.config))
    stage1_checkpoint_path = resolve_path(str(cfg.stage1.checkpoint))
    if not stage1_checkpoint_path.is_file():
        raise FileNotFoundError(stage1_checkpoint_path)

    stage1_cfg = load_config(str(stage1_config_path))
    stage1 = AriadneLUTV2(stage1_cfg.model).to(device)
    load_checkpoint(
        str(stage1_checkpoint_path),
        stage1,
        optimizer=None,
        scheduler=None,
        scaler=None,
        strict=True,
        map_location=device,
    )
    stage1.eval()
    for parameter in stage1.parameters():
        parameter.requires_grad_(False)

    stage2_model = AriadneMovieNetPairStage2(stage1=stage1, cfg=cfg.model).to(device)
    criterion = MovieNetPairStage2Objective(cfg.loss).to(device)

    device_ids = [] if distributed else [
        int(value)
        for value in getattr(cfg.train, "data_parallel_device_ids", [])
    ]
    if device.type == "cuda" and len(device_ids) > 1:
        visible_devices = torch.cuda.device_count()
        invalid = [value for value in device_ids if value < 0 or value >= visible_devices]
        if invalid:
            raise ValueError(
                f"Invalid data_parallel_device_ids={invalid}; "
                f"only {visible_devices} CUDA devices are visible"
            )
        model: nn.Module = nn.DataParallel(
            stage2_model, device_ids=device_ids, output_device=device_ids[0]
        )
    else:
        model = stage2_model

    if distributed:
        model = DistributedDataParallel(
            stage2_model,
            device_ids=[local_rank],
            bucket_cap_mb=10,
            gradient_as_bucket_view=True,
            static_graph=True,
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    train_loader, train_dataset, val_loader, val_dataset = build_loaders_from_dataset_module(
        cfg, rank=rank, world_size=world_size
    )

    trainable = list(stage2_model.trainable_parameters())
    optimizer = torch.optim.AdamW(
        trainable,
        lr=float(cfg.train.learning_rate),
        betas=(float(cfg.train.beta1), float(cfg.train.beta2)),
        weight_decay=float(cfg.train.weight_decay),
    )

    accumulation = max(int(cfg.data.gradient_accumulation_steps), 1)
    latest_every_steps = max(
        int(getattr(cfg.train, "latest_every_steps", 0)), 0
    )
    batches_per_epoch = len(train_loader)
    if args.max_train_batches > 0:
        batches_per_epoch = min(batches_per_epoch, int(args.max_train_batches))
    optimizer_steps_per_epoch = math.ceil(batches_per_epoch / accumulation)
    total_optimizer_steps = optimizer_steps_per_epoch * int(cfg.train.epochs)
    scheduler = build_scheduler(
        optimizer,
        total_steps=total_optimizer_steps,
        warmup_steps=int(cfg.train.lr_warmup_steps),
        min_lr_ratio=float(cfg.train.min_lr_ratio),
    )

    amp_enabled = bool(cfg.train.amp) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)

    start_epoch = 0
    resume_batch_index = 0
    global_step = 0
    best_psnr = float("-inf")
    resume_value = args.resume or str(getattr(cfg.train, "resume", ""))
    if resume_value:
        checkpoint = load_stage2_checkpoint(
            resolve_path(resume_value),
            model,
            optimizer=optimizer,
            scheduler=scheduler,
            scaler=scaler,
            map_location=device,
        )
        checkpoint_epoch = int(checkpoint.get("epoch", -1))
        if bool(checkpoint.get("epoch_complete", True)):
            start_epoch = checkpoint_epoch + 1
        else:
            start_epoch = checkpoint_epoch
            resume_batch_index = int(checkpoint.get("next_batch_index", 0))
        global_step = int(checkpoint.get("global_step", 0))
        best_psnr = float(checkpoint.get("best_psnr", best_psnr))

    trainable_count = sum(p.numel() for p in trainable if p.requires_grad)
    frozen_count = sum(p.numel() for p in stage2_model.stage1.parameters())
    rank_print = print if is_main_process else lambda *args, **kwargs: None
    local_batch = int(
        getattr(cfg.data, "batch_size_per_gpu", cfg.data.batch_size)
        if distributed else cfg.data.batch_size
    )

    rank_print(f"Frozen Stage-1 parameters: {frozen_count:,}")
    rank_print(f"Trainable Stage-2 parameters: {trainable_count:,}")
    if distributed:
        rank_print(f"DistributedDataParallel backend=NCCL, world_size={world_size}")
    if isinstance(model, nn.DataParallel):
        rank_print(f"DataParallel CUDA devices: {model.device_ids}")
    rank_print(f"Train pairs: {len(train_dataset):,}; val pairs: {len(val_dataset):,}")
    rank_print(
        f"per_gpu_pair_batch={local_batch}, accumulation={accumulation}, "
        f"global_pair_batch={local_batch * world_size * accumulation}; "
        "each pair contributes two transfer directions"
    )

    log_path = paths["logs"] / "train.csv" if is_main_process else Path(os.devnull)
    log_exists = is_main_process and log_path.exists() and resume_value
    log_handle = log_path.open("a" if log_exists else "w", newline="", buffering=1)
    fieldnames = [
        "epoch", "global_step", "split", "total", "charbonnier", "lpips",
        "low_frequency", "color_moments", "delta_l2", "delta_smooth",
        "final_lut_curvature", "lut_range", "baseline_psnr", "refined_psnr",
        "psnr_gain", "delta_abs_mean", "final_lut_oob", "residual_strength", "lr",
    ]
    writer = csv.DictWriter(log_handle, fieldnames=fieldnames)
    if is_main_process and not log_exists:
        writer.writeheader()

    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            sampler = getattr(train_loader, "sampler", None)
            if distributed and hasattr(sampler, "set_epoch"):
                sampler.set_epoch(epoch)
            epoch_resume_batch = resume_batch_index if epoch == start_epoch else 0
            if epoch_resume_batch > 0:
                if not hasattr(sampler, "set_start_index"):
                    raise RuntimeError(
                        "Mid-epoch resume requires ResumableDistributedSampler"
                    )
                sampler.set_start_index(epoch_resume_batch * local_batch)
                rank_print(
                    f"Resuming epoch {epoch} at batch {epoch_resume_batch:,}"
                )
            elif hasattr(sampler, "set_start_index"):
                sampler.set_start_index(0)
            epoch_batches = len(train_loader)
            if args.max_train_batches > 0:
                epoch_batches = min(epoch_batches, int(args.max_train_batches))
            model.train()
            optimizer.zero_grad(set_to_none=True)
            progress = tqdm(train_loader, desc=f"MoviePairS2 {epoch:03d}", disable=not is_main_process)
            seen_batches = 0

            for batch_index, batch in enumerate(progress):
                if args.max_train_batches > 0 and batch_index >= args.max_train_batches:
                    break
                seen_batches += 1
                absolute_batch_index = epoch_resume_batch + batch_index
                content, style, target, views = make_bidirectional_batch(batch, device)
                alpha = residual_strength(global_step, int(cfg.train.residual_ramp_steps))
                should_step = (seen_batches % accumulation == 0)
                is_last = batch_index + 1 >= epoch_batches
                sync_context = (
                    model.no_sync()
                    if distributed and not (should_step or is_last)
                    else nullcontext()
                )
                with sync_context:
                    with autocast(device_type=device.type, enabled=amp_enabled):
                        result = model(content, style, residual_strength=alpha)
                        losses = criterion(result, target)
                        scaled_loss = losses["total"] / float(accumulation)

                    scaler.scale(scaled_loss).backward()

                if should_step or is_last:
                    scaler.unscale_(optimizer)
                    grad_norm = torch.nn.utils.clip_grad_norm_(
                        trainable,
                        float(cfg.train.grad_clip),
                    )
                    scale_before_step = scaler.get_scale()
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer_step_was_skipped = scaler.get_scale() < scale_before_step
                    optimizer.zero_grad(set_to_none=True)
                    if not optimizer_step_was_skipped:
                        scheduler.step()
                        global_step += 1

                    baseline_psnr = batch_psnr(result["baseline_output"], target)
                    refined_psnr = batch_psnr(result["output"], target)
                    values = {key: tensor_scalar(value) for key, value in losses.items()}
                    values.update(
                        {
                            "baseline_psnr": tensor_scalar(baseline_psnr),
                            "refined_psnr": tensor_scalar(refined_psnr),
                            "psnr_gain": tensor_scalar(refined_psnr - baseline_psnr),
                            "delta_abs_mean": tensor_scalar(result["delta_lut"].abs().mean()),
                            "final_lut_oob": tensor_scalar(
                                ((result["final_lut"] < 0.0) | (result["final_lut"] > 1.0))
                                .float()
                                .mean()
                            ),
                            "residual_strength": float(alpha),
                            "lr": float(optimizer.param_groups[0]["lr"]),
                        }
                    )
                    progress.set_postfix(
                        loss=f"{values['total']:.4f}",
                        base=f"{values['baseline_psnr']:.2f}",
                        out=f"{values['refined_psnr']:.2f}",
                        gain=f"{values['psnr_gain']:+.2f}",
                        alpha=f"{alpha:.2f}",
                    )

                    if global_step % int(cfg.train.log_every) == 0:
                        writer.writerow(
                            {
                                "epoch": epoch,
                                "global_step": global_step,
                                "split": "train",
                                **{key: values.get(key, "") for key in fieldnames if key not in {"epoch", "global_step", "split"}},
                            }
                        )

                    if is_main_process and global_step % int(cfg.train.image_every) == 0:
                        save_stage2_grid(
                            tensors=(
                                content,
                                style,
                                result["canonical"],
                                result["baseline_output"],
                                result["output"],
                                target,
                            ),
                            labels=(
                                "corrupted content",
                                "movie reference",
                                "frozen Z_C",
                                "Stage1 baseline",
                                "Stage2 refined",
                                "movie GT",
                            ),
                            path=paths["images"] / f"train_step_{global_step:08d}.png",
                            max_items=int(cfg.validation.max_visual_items),
                            title=f"step={global_step} alpha={alpha:.3f}",
                        )
                    should_save_latest = (
                        latest_every_steps > 0
                        and not optimizer_step_was_skipped
                        and global_step > 0
                        and global_step % latest_every_steps == 0
                    )
                    if should_save_latest:
                        save_stage2_checkpoint(
                            paths["checkpoints"] / "latest.pth",
                            model,
                            optimizer,
                            scheduler,
                            scaler,
                            epoch,
                            global_step,
                            best_psnr,
                            cfg,
                            next_batch_index=absolute_batch_index + 1,
                            epoch_complete=False,
                        )
                        if distributed:
                            dist.barrier()

            if seen_batches == 0:
                raise RuntimeError("Training loader produced no batches")

            val = validation_pass(
                model=model,
                loader=val_loader,
                criterion=criterion,
                cfg=cfg,
                device=device,
                epoch=epoch,
                paths=paths,
                max_batches=int(args.max_val_batches),
                distributed=distributed,
                is_main_process=is_main_process,
            )
            writer.writerow(
                {
                    "epoch": epoch,
                    "global_step": global_step,
                    "split": "val",
                    **{key: val.get(key, "") for key in fieldnames if key not in {"epoch", "global_step", "split", "residual_strength", "lr"}},
                    "residual_strength": 1.0,
                    "lr": float(optimizer.param_groups[0]["lr"]),
                }
            )
            rank_print("Validation:", " ".join(f"{k}={v:.6f}" for k, v in val.items()))

            is_best = float(val["refined_psnr"]) > best_psnr
            if is_best:
                best_psnr = float(val["refined_psnr"])
                save_stage2_checkpoint(
                    paths["checkpoints"] / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_psnr,
                    cfg,
                )

            # latest.pth is overwritten every epoch and is the final checkpoint
            # when training completes. Save no per-epoch stack by default.
            save_stage2_checkpoint(
                paths["checkpoints"] / "latest.pth",
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_psnr,
                cfg,
            )

            milestones = {
                int(value)
                for value in getattr(cfg.train, "checkpoint_milestone_epochs", [])
            }
            if epoch + 1 in milestones:
                save_stage2_checkpoint(
                    paths["checkpoints"] / f"epoch_{epoch + 1:04d}.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_psnr,
                    cfg,
                )
            if distributed:
                dist.barrier()
    finally:
        log_handle.close()
        if distributed and dist.is_initialized():
            dist.destroy_process_group()


if __name__ == "__main__":
    main()
