from __future__ import annotations

import argparse
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import sys

from PIL import Image
import torch
from torch.amp import GradScaler, autocast
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import psnr
from models.ariadne_lut_v2 import AriadneLUTV2
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.image_v2 import save_labeled_grid_v2
from utils.logger import ScalarLogger
from utils.seed import seed_everything

from neural_preset_dataset import build_loader
from neural_preset_loss import NeuralPresetObjectiveForAriadne


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train Ariadne predicted LUTs in Neural Preset's setting."
    )
    parser.add_argument("--config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else REPO_ROOT / path


def prepare_run(cfg, config_path: str) -> dict[str, Path]:
    root = resolve_path(str(cfg.experiment.output_dir))
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(config_path, root / "config.yaml")
    return paths


def move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=True)


def scalar_dict(values: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in values.items()
    }


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
            "output_out_of_range_ratio": float(
                torch.cat([output_ab, output_ba]).sub(0.5).abs().gt(0.5).float().mean()
            ),
            "canonical_out_of_range_ratio": float(
                torch.cat([canonical_a, canonical_b])
                .sub(0.5)
                .abs()
                .gt(0.5)
                .float()
                .mean()
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


def average(totals: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("Cannot average zero validation/training samples")
    return {key: value / count for key, value in totals.items()}


def load_fixed_tensor(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(resolve_path(path)).convert("RGB")
    image = image.resize(
        (int(image_size), int(image_size)), Image.Resampling.BICUBIC
    )
    return to_tensor(image).unsqueeze(0).to(device)


@torch.no_grad()
def save_fixed_pair(
    model: AriadneLUTV2,
    cfg,
    device: torch.device,
    epoch: int,
    paths: dict[str, Path],
) -> dict[str, float]:
    was_training = model.training
    model.eval()
    content = load_fixed_tensor(
        str(cfg.validation.fixed_content), int(cfg.data.image_size), device
    )
    style = load_fixed_tensor(
        str(cfg.validation.fixed_style), int(cfg.data.image_size), device
    )
    result = model(content, style)
    output = result["output"]
    canonical = result["canonical"]

    suffix = "init" if epoch < 0 else f"epoch_{epoch:04d}"
    save_labeled_grid_v2(
        tensors=(content, style, canonical, output),
        path=paths["images"] / f"fixed_blue_invasion_{suffix}.png",
        labels=("content_giraffe", "style_snow_blue_sky", "canonical", "stylized"),
        title=f"Fixed blue-invasion pair - {suffix}",
        max_items=1,
    )

    channel_delta = (output - content).mean(dim=(0, 2, 3))
    fixed_metrics = {
        "fixed_delta_red": float(channel_delta[0]),
        "fixed_delta_green": float(channel_delta[1]),
        "fixed_delta_blue": float(channel_delta[2]),
        "fixed_blue_dominance": float(
            channel_delta[2] - 0.5 * (channel_delta[0] + channel_delta[1])
        ),
        "fixed_output_out_of_range_ratio": float(
            output.sub(0.5).abs().gt(0.5).float().mean()
        ),
    }
    if was_training:
        model.train()
    return fixed_metrics


@torch.no_grad()
def validate(
    model: AriadneLUTV2,
    loader,
    criterion: NeuralPresetObjectiveForAriadne,
    cfg,
    device: torch.device,
    epoch: int,
    paths: dict[str, Path],
    max_batches: int,
) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    first_visual = None
    iterator = loader if max_batches <= 0 else islice(loader, max_batches)

    for batch in tqdm(iterator, desc=f"ValNP {epoch:03d}", total=(max_batches or len(loader))):
        content = move_tensor(batch["content"], device)
        image_a = move_tensor(batch["image_a"], device)
        image_b = move_tensor(batch["image_b"], device)
        pair_result = model.forward_pair(image_a, image_b)
        losses = criterion(pair_result, image_a, image_b)
        values = scalar_dict(losses)
        add_metrics(values, pair_result, image_a, image_b)
        batch_size = int(image_a.shape[0])
        accumulate(totals, values, batch_size)
        sample_count += batch_size

        if first_visual is None:
            first_visual = (
                content,
                image_a,
                image_b,
                pair_result["state_a"]["canonical"],
                pair_result["state_b"]["canonical"],
                pair_result["output_ab"],
                image_b,
                pair_result["output_ba"],
                image_a,
            )

    if first_visual is None:
        raise RuntimeError("Validation loader produced no batches")

    save_labeled_grid_v2(
        tensors=first_visual,
        path=paths["images"] / f"val_neural_preset_pairs_epoch_{epoch:04d}.png",
        labels=(
            "original_COCO",
            "LUT_A",
            "LUT_B",
            "canonical_A",
            "canonical_B",
            "A_to_B",
            "ground_truth_B",
            "B_to_A",
            "ground_truth_A",
        ),
        title=f"Ariadne LUT in Neural Preset setting - epoch {epoch}",
        max_items=int(cfg.validation.max_visual_items),
    )

    metrics = average(totals, sample_count)
    metrics.update(save_fixed_pair(model, cfg, device, epoch, paths))
    return metrics


def write_manifest(cfg, paths: dict[str, Path], train_dataset, val_dataset) -> None:
    payload = {
        "protocol": "Ariadne predicted 3D LUT model with Neural Preset data/objective",
        "coco_root": str(cfg.data.coco_root),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "lut_root": str(cfg.data.lut_root),
        "lut_count": len(train_dataset.lut_paths),
        "lut_files": [str(path) for path in train_dataset.lut_paths],
        "fixed_content": str(resolve_path(str(cfg.validation.fixed_content))),
        "fixed_style": str(resolve_path(str(cfg.validation.fixed_style))),
    }
    (paths["root"] / "data_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    if args.output_dir:
        cfg.experiment.output_dir = args.output_dir
    if args.resume:
        cfg.train.resume = args.resume
    if args.epochs > 0:
        cfg.train.epochs = args.epochs
    if args.num_workers >= 0:
        cfg.data.num_workers = args.num_workers
        cfg.data.persistent_workers = args.num_workers > 0
    if args.batch_size > 0:
        cfg.data.batch_size = args.batch_size

    seed_everything(int(cfg.experiment.seed))
    if not torch.cuda.is_available() and str(cfg.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(str(cfg.device))
    paths = prepare_run(cfg, args.config)

    latest_path = paths["checkpoints"] / "latest.pth"
    if latest_path.exists() and not str(cfg.train.resume):
        raise RuntimeError(
            f"Refusing to overwrite existing run at {paths['root']}; set train.resume"
        )

    print("Loading Neural Preset LUTs for train split...", flush=True)
    train_loader, train_dataset = build_loader(
        cfg, str(cfg.data.train_split), train=True
    )
    print("Loading Neural Preset LUTs for validation split...", flush=True)
    val_loader, val_dataset = build_loader(
        cfg, str(cfg.data.val_split), train=False
    )
    write_manifest(cfg, paths, train_dataset, val_dataset)

    model = AriadneLUTV2(cfg.model).to(device)
    criterion = NeuralPresetObjectiveForAriadne(
        lambda_consistency=float(cfg.loss.lambda_consistency)
    ).to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
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
    logger = ScalarLogger(paths["logs"])

    start_epoch = 0
    global_step = 0
    best_metric = float("inf")
    if str(cfg.train.resume):
        checkpoint = load_checkpoint(
            resolve_path(str(cfg.train.resume)),
            model,
            optimizer,
            scheduler,
            scaler,
            map_location=device,
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
    else:
        initial_metrics = save_fixed_pair(model, cfg, device, -1, paths)
        logger.log(initial_metrics, global_step, "fixed_init")

    parameter_count = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"Model parameters={parameter_count:,}; train_images={len(train_dataset):,}; "
        f"val_images={len(val_dataset):,}; LUTs={len(train_dataset.lut_paths)}; "
        f"micro_batch={cfg.data.batch_size}; accumulation={cfg.data.gradient_accumulation_steps}; "
        f"effective_batch={int(cfg.data.batch_size) * int(cfg.data.gradient_accumulation_steps)}",
        flush=True,
    )

    accumulation_steps = int(cfg.data.gradient_accumulation_steps)
    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_totals: dict[str, float] = {}
            train_samples = 0
            configured_total = len(train_loader)
            total_batches = (
                min(configured_total, args.max_train_batches)
                if args.max_train_batches > 0
                else configured_total
            )
            iterator = train_loader if args.max_train_batches <= 0 else islice(
                train_loader, args.max_train_batches
            )
            progress = tqdm(iterator, total=total_batches, desc=f"TrainNP {epoch:03d}")
            current_accumulation_target = accumulation_steps

            for batch_index, batch in enumerate(progress):
                if batch_index % accumulation_steps == 0:
                    current_accumulation_target = min(
                        accumulation_steps, total_batches - batch_index
                    )

                content = move_tensor(batch["content"], device)
                image_a = move_tensor(batch["image_a"], device)
                image_b = move_tensor(batch["image_b"], device)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    pair_result = model.forward_pair(image_a, image_b)
                    losses = criterion(pair_result, image_a, image_b)
                    scaled_loss = losses["total"] / current_accumulation_target

                scaler.scale(scaled_loss).backward()
                should_step = (
                    (batch_index + 1) % accumulation_steps == 0
                    or batch_index + 1 == total_batches
                )
                if should_step:
                    scaler.unscale_(optimizer)
                    if float(cfg.train.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(
                            model.parameters(), float(cfg.train.grad_clip)
                        )
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                values = scalar_dict(losses)
                add_metrics(values, pair_result, image_a, image_b)
                batch_size = int(image_a.shape[0])
                accumulate(train_totals, values, batch_size)
                train_samples += batch_size
                progress.set_postfix(
                    total=f"{values['total']:.4f}",
                    psnr=f"{values['mean_bidirectional_psnr']:.2f}",
                    canonical=f"{values['canonical_psnr']:.2f}",
                    step=global_step,
                )

                if should_step and global_step % int(
                    cfg.train.log_every_optimizer_steps
                ) == 0:
                    logger.log(
                        {**values, "lr": optimizer.param_groups[0]["lr"]},
                        global_step,
                        "train",
                    )

                if should_step and global_step % int(
                    cfg.train.image_every_optimizer_steps
                ) == 0:
                    save_labeled_grid_v2(
                        tensors=(
                            content,
                            image_a,
                            image_b,
                            pair_result["output_ab"],
                            image_b,
                            pair_result["output_ba"],
                            image_a,
                        ),
                        path=paths["images"] / f"train_step_{global_step:08d}.png",
                        labels=(
                            "original_COCO",
                            "LUT_A",
                            "LUT_B",
                            "A_to_B",
                            "ground_truth_B",
                            "B_to_A",
                            "ground_truth_A",
                        ),
                        title=f"Neural Preset setting - optimizer step {global_step}",
                        max_items=int(cfg.validation.max_visual_items),
                    )

            if train_samples == 0:
                raise RuntimeError("Training loader produced no batches")
            train_epoch_values = average(train_totals, train_samples)
            logger.log(train_epoch_values, global_step, "train_epoch")

            val_values = validate(
                model,
                val_loader,
                criterion,
                cfg,
                device,
                epoch,
                paths,
                args.max_val_batches,
            )
            logger.log(val_values, global_step, "val")
            scheduler.step()

            print(
                "Validation:",
                " ".join(f"{key}={value:.6f}" for key, value in val_values.items()),
                flush=True,
            )

            checkpoint_extra = {
                "protocol": "neural_preset_data_and_objective",
                "validation_total": val_values["total"],
                "validation_mean_bidirectional_psnr": val_values[
                    "mean_bidirectional_psnr"
                ],
                "fixed_blue_dominance": val_values["fixed_blue_dominance"],
            }
            if val_values["total"] < best_metric:
                best_metric = val_values["total"]
                save_checkpoint(
                    paths["checkpoints"] / "best.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_metric,
                    extra=checkpoint_extra,
                )

            save_checkpoint(
                latest_path,
                model,
                optimizer,
                scheduler,
                scaler,
                epoch,
                global_step,
                best_metric,
                extra=checkpoint_extra,
            )
            if (epoch + 1) % int(cfg.train.checkpoint_every_epochs) == 0:
                save_checkpoint(
                    paths["checkpoints"] / f"epoch_{epoch:04d}.pth",
                    model,
                    optimizer,
                    scheduler,
                    scaler,
                    epoch,
                    global_step,
                    best_metric,
                    extra=checkpoint_extra,
                )
    finally:
        logger.close()


if __name__ == "__main__":
    main()

