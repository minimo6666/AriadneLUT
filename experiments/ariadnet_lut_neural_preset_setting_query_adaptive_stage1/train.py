from __future__ import annotations

import argparse
import importlib.util
from itertools import islice
import json
import math
from pathlib import Path
import shutil
import sys
import time

from PIL import Image
import torch
from torch.amp import GradScaler, autocast
from torchvision.transforms.functional import to_tensor
from tqdm import tqdm
import yaml


EXPERIMENT_DIR = Path(__file__).resolve().parent
REPO_ROOT = EXPERIMENT_DIR.parents[1]
BASE_EXPERIMENT_DIR = REPO_ROOT / "experiments" / "ariadnet_lut_neural_preset_setting"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))
if str(EXPERIMENT_DIR) not in sys.path:
    sys.path.insert(0, str(EXPERIMENT_DIR))

from metrics import psnr
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.image_v2 import save_labeled_grid_v2
from utils.logger import ScalarLogger
from utils.seed import seed_everything

from ariadne_query_adaptive import AriadneLUTQueryAdaptiveStage1
from query_adaptive_loss import QueryAdaptiveNeuralPresetObjective


def _load_base_dataset_module():
    path = BASE_EXPERIMENT_DIR / "neural_preset_dataset.py"
    spec = importlib.util.spec_from_file_location("ariadne_base_np_dataset", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load dataset module: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BASE_DATASET = _load_base_dataset_module()
build_loader = BASE_DATASET.build_loader


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train query-adaptive hierarchical Ariadne Stage 1 in the exact Neural Preset setting."
    )
    parser.add_argument("--qa-config", default=str(EXPERIMENT_DIR / "config.yaml"))
    parser.add_argument("--base-config", default="")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--resume", default="")
    parser.add_argument("--init-dense-checkpoint", default="")
    parser.add_argument("--epochs", type=int, default=0)
    parser.add_argument("--max-train-batches", type=int, default=0)
    parser.add_argument("--max-val-batches", type=int, default=0)
    parser.add_argument("--num-workers", type=int, default=-1)
    parser.add_argument("--batch-size", type=int, default=0)
    return parser.parse_args()


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path


def load_qa_yaml(path: str) -> dict:
    payload = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("QA config must be a YAML mapping")
    return payload


def prepare_run(output_dir: str, base_config_path: Path, qa_config_path: Path) -> dict[str, Path]:
    root = resolve_path(output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(base_config_path, root / "base_config.yaml")
    shutil.copy2(qa_config_path, root / "query_adaptive_config.yaml")
    return paths


def move_tensor(value: torch.Tensor, device: torch.device) -> torch.Tensor:
    return value.to(device, non_blocking=True)


def scalar_dict(values: dict[str, torch.Tensor | float]) -> dict[str, float]:
    return {
        key: float(value.detach().float().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in values.items()
    }


def _level_ratios(pyramid: dict, prefix: str) -> dict[str, float]:
    values = {}
    for level_name in ("level_16", "level_32"):
        level = pyramid[level_name]
        suffix = level_name.split("_")[-1]
        values[f"{prefix}_active_{suffix}"] = float(level.support.active_ratio.mean().cpu())
        values[f"{prefix}_core_{suffix}"] = float(level.support.core_ratio.mean().cpu())
        values[f"{prefix}_tokens_{suffix}"] = float(level.token_count)
    return values


def add_metrics(values: dict[str, float], pair_result: dict, image_a: torch.Tensor, image_b: torch.Tensor) -> None:
    output_ab = pair_result["output_ab"]
    output_ba = pair_result["output_ba"]
    canonical_a = pair_result["state_a"]["canonical"]
    canonical_b = pair_result["state_b"]["canonical"]
    values.update(
        {
            "psnr_a_to_b": float(psnr(output_ab.clamp(0, 1), image_b)),
            "psnr_b_to_a": float(psnr(output_ba.clamp(0, 1), image_a)),
            "canonical_psnr": float(psnr(canonical_a.clamp(0, 1), canonical_b.clamp(0, 1))),
            "output_out_of_range_ratio": float(
                torch.cat([output_ab, output_ba]).sub(0.5).abs().gt(0.5).float().mean()
            ),
            "canonical_out_of_range_ratio": float(
                torch.cat([canonical_a, canonical_b]).sub(0.5).abs().gt(0.5).float().mean()
            ),
        }
    )
    values["mean_bidirectional_psnr"] = 0.5 * (values["psnr_a_to_b"] + values["psnr_b_to_a"])

    # Average A/B directions. Style A is queried by canonical B and vice versa.
    for branch, key in (("norm", "normalization_lut_pyramid"), ("style", "style_lut_pyramid")):
        a = _level_ratios(pair_result["state_a"][key], f"{branch}_a")
        b = _level_ratios(pair_result["state_b"][key], f"{branch}_b")
        for level in (16, 32):
            values[f"{branch}_active_{level}"] = 0.5 * (
                a[f"{branch}_a_active_{level}"] + b[f"{branch}_b_active_{level}"]
            )
            values[f"{branch}_core_{level}"] = 0.5 * (
                a[f"{branch}_a_core_{level}"] + b[f"{branch}_b_core_{level}"]
            )


def accumulate(totals: dict[str, float], values: dict[str, float], weight: int) -> None:
    for key, value in values.items():
        if math.isfinite(float(value)):
            totals[key] = totals.get(key, 0.0) + float(value) * int(weight)


def average(totals: dict[str, float], count: int) -> dict[str, float]:
    if count <= 0:
        raise RuntimeError("Cannot average zero samples")
    return {key: value / count for key, value in totals.items()}


def load_fixed_tensor(path: str, image_size: int, device: torch.device) -> torch.Tensor:
    image = Image.open(resolve_path(path)).convert("RGB")
    image = image.resize((int(image_size), int(image_size)), Image.Resampling.BICUBIC)
    return to_tensor(image).unsqueeze(0).to(device)


@torch.no_grad()
def save_fixed_pair(model, base_cfg, device, epoch: int, paths: dict[str, Path]) -> dict[str, float]:
    was_training = model.training
    model.eval()
    content = load_fixed_tensor(str(base_cfg.validation.fixed_content), int(base_cfg.data.image_size), device)
    style = load_fixed_tensor(str(base_cfg.validation.fixed_style), int(base_cfg.data.image_size), device)
    result = model(content, style)
    output = result["output"]
    canonical = result["canonical"]
    suffix = "init" if epoch < 0 else f"epoch_{epoch:04d}"
    save_labeled_grid_v2(
        tensors=(content, style, canonical, output),
        path=paths["images"] / f"fixed_blue_invasion_{suffix}.png",
        labels=("content_giraffe", "style_snow_blue_sky", "canonical", "stylized"),
        title=f"Query-adaptive fixed pair - {suffix}",
        max_items=1,
    )
    channel_delta = (output - content).mean(dim=(0, 2, 3))
    metrics = {
        "fixed_delta_red": float(channel_delta[0]),
        "fixed_delta_green": float(channel_delta[1]),
        "fixed_delta_blue": float(channel_delta[2]),
        "fixed_blue_dominance": float(channel_delta[2] - 0.5 * (channel_delta[0] + channel_delta[1])),
        "fixed_output_out_of_range_ratio": float(output.sub(0.5).abs().gt(0.5).float().mean()),
    }
    if was_training:
        model.train()
    return metrics


@torch.no_grad()
def validate(model, loader, criterion, base_cfg, device, epoch, paths, max_batches: int) -> dict[str, float]:
    model.eval()
    totals: dict[str, float] = {}
    sample_count = 0
    first_visual = None
    iterator = loader if max_batches <= 0 else islice(loader, max_batches)
    total = max_batches or len(loader)
    for batch in tqdm(iterator, desc=f"ValQA {epoch:03d}", total=total):
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
        path=paths["images"] / f"val_query_adaptive_pairs_epoch_{epoch:04d}.png",
        labels=(
            "original_COCO", "LUT_A", "LUT_B", "canonical_A", "canonical_B",
            "A_to_B", "ground_truth_B", "B_to_A", "ground_truth_A",
        ),
        title=f"Query-Adaptive Stage1 - epoch {epoch}",
        max_items=int(base_cfg.validation.max_visual_items),
    )
    metrics = average(totals, sample_count)
    metrics.update(save_fixed_pair(model, base_cfg, device, epoch, paths))
    return metrics


def _extract_state_dict(payload):
    if isinstance(payload, dict):
        for key in ("model", "state_dict", "model_state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if payload and all(torch.is_tensor(v) for v in payload.values()):
            return payload
    raise TypeError("Unsupported checkpoint structure for compatible initialization")


def compatible_dense_init(model: torch.nn.Module, checkpoint_path: Path) -> tuple[int, int]:
    """Optionally transplant encoders + identical 8^3 generator weights.

    High-resolution dense decoder weights intentionally have no counterpart in
    the query-adaptive model. Matching is name-and-shape based and therefore
    cannot silently load an incompatible tensor.
    """
    payload = torch.load(checkpoint_path, map_location="cpu")
    source = _extract_state_dict(payload)
    target = model.state_dict()
    matched = {k: v for k, v in source.items() if k in target and target[k].shape == v.shape}
    missing = len(target) - len(matched)
    model.load_state_dict(matched, strict=False)
    return len(matched), missing


def write_manifest(base_cfg, qa_cfg: dict, paths: dict[str, Path], train_dataset, val_dataset) -> None:
    payload = {
        "protocol": "Query-Adaptive Hierarchical LUT Stage1 with exact Neural Preset data pairing",
        "base_config": str(resolve_path(str(qa_cfg["base_config"]))),
        "coco_root": str(base_cfg.data.coco_root),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "lut_root": str(base_cfg.data.lut_root),
        "lut_count": len(train_dataset.lut_paths),
        "query_adaptive_model": qa_cfg.get("model", {}),
        "query_adaptive_loss": qa_cfg.get("loss", {}),
    }
    (paths["root"] / "data_manifest.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def main() -> None:
    args = parse_args()
    qa_cfg = load_qa_yaml(args.qa_config)
    base_config_path = resolve_path(args.base_config or str(qa_cfg["base_config"]))
    base_cfg = load_config(str(base_config_path))

    if args.num_workers >= 0:
        base_cfg.data.num_workers = args.num_workers
        base_cfg.data.persistent_workers = args.num_workers > 0
    if args.batch_size > 0:
        base_cfg.data.batch_size = args.batch_size
    epochs = int(args.epochs) if args.epochs > 0 else int(base_cfg.train.epochs)
    output_dir = args.output_dir or str(qa_cfg["experiment"]["output_dir"])

    seed_everything(int(base_cfg.experiment.seed))
    if not torch.cuda.is_available() and str(base_cfg.device).startswith("cuda"):
        raise RuntimeError("CUDA was requested but is not available")
    device = torch.device(str(base_cfg.device))
    paths = prepare_run(output_dir, base_config_path, Path(args.qa_config))
    latest_path = paths["checkpoints"] / "latest.pth"
    if latest_path.exists() and not args.resume:
        raise RuntimeError(f"Refusing to overwrite existing run at {paths['root']}; pass --resume")

    print("Loading the exact original Neural Preset Stage-1 dataset...", flush=True)
    train_loader, train_dataset = build_loader(base_cfg, str(base_cfg.data.train_split), train=True)
    val_loader, val_dataset = build_loader(base_cfg, str(base_cfg.data.val_split), train=False)
    write_manifest(base_cfg, qa_cfg, paths, train_dataset, val_dataset)

    model = AriadneLUTQueryAdaptiveStage1(base_cfg.model, qa_cfg["model"]).to(device)
    criterion = QueryAdaptiveNeuralPresetObjective(
        lambda_consistency=float(base_cfg.loss.lambda_consistency),
        qa_loss_cfg=qa_cfg.get("loss", {}),
    ).to(device)

    if args.init_dense_checkpoint and not args.resume:
        matched, missing = compatible_dense_init(model, resolve_path(args.init_dense_checkpoint))
        print(f"Compatible dense init: loaded {matched} tensors; {missing} QA tensors remain new.")

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=float(base_cfg.train.learning_rate),
        betas=(float(base_cfg.train.beta1), float(base_cfg.train.beta2)),
    )
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer,
        step_size=int(base_cfg.train.scheduler_step_size),
        gamma=float(base_cfg.train.scheduler_gamma),
    )
    amp_enabled = bool(base_cfg.train.amp) and device.type == "cuda"
    scaler = GradScaler("cuda", enabled=amp_enabled)
    logger = ScalarLogger(paths["logs"])

    start_epoch, global_step, best_metric = 0, 0, float("inf")
    if args.resume:
        checkpoint = load_checkpoint(
            resolve_path(args.resume), model, optimizer, scheduler, scaler, map_location=device
        )
        start_epoch = int(checkpoint.get("epoch", -1)) + 1
        global_step = int(checkpoint.get("global_step", 0))
        best_metric = float(checkpoint.get("best_metric", best_metric))
    else:
        logger.log(save_fixed_pair(model, base_cfg, device, -1, paths), global_step, "fixed_init")

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(
        f"QA model parameters={trainable:,}; train_images={len(train_dataset):,}; "
        f"val_images={len(val_dataset):,}; micro_batch={base_cfg.data.batch_size}; "
        f"accumulation={base_cfg.data.gradient_accumulation_steps}",
        flush=True,
    )

    accumulation_steps = int(base_cfg.data.gradient_accumulation_steps)
    try:
        for epoch in range(start_epoch, epochs):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            train_totals: dict[str, float] = {}
            train_samples = 0
            configured_total = len(train_loader)
            total_batches = min(configured_total, args.max_train_batches) if args.max_train_batches > 0 else configured_total
            iterator = train_loader if args.max_train_batches <= 0 else islice(train_loader, args.max_train_batches)
            progress = tqdm(iterator, total=total_batches, desc=f"TrainQA {epoch:03d}")
            current_accumulation_target = accumulation_steps
            tick = time.perf_counter()

            for batch_index, batch in enumerate(progress):
                if batch_index % accumulation_steps == 0:
                    current_accumulation_target = min(accumulation_steps, total_batches - batch_index)
                content = move_tensor(batch["content"], device)
                image_a = move_tensor(batch["image_a"], device)
                image_b = move_tensor(batch["image_b"], device)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    pair_result = model.forward_pair(image_a, image_b)
                    losses = criterion(pair_result, image_a, image_b)
                    scaled_loss = losses["total"] / current_accumulation_target
                scaler.scale(scaled_loss).backward()

                should_step = ((batch_index + 1) % accumulation_steps == 0) or (batch_index + 1 == total_batches)
                if should_step:
                    scaler.unscale_(optimizer)
                    if float(base_cfg.train.grad_clip) > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), float(base_cfg.train.grad_clip))
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    global_step += 1

                values = scalar_dict(losses)
                add_metrics(values, pair_result, image_a, image_b)
                values["seconds_per_microbatch"] = time.perf_counter() - tick
                tick = time.perf_counter()
                batch_size = int(image_a.shape[0])
                accumulate(train_totals, values, batch_size)
                train_samples += batch_size
                progress.set_postfix(
                    total=f"{values['total']:.4f}",
                    psnr=f"{values['mean_bidirectional_psnr']:.2f}",
                    n32=f"{100*values['norm_active_32']:.1f}%",
                    s32=f"{100*values['style_active_32']:.1f}%",
                    step=global_step,
                )

                if should_step and global_step % int(base_cfg.train.log_every_optimizer_steps) == 0:
                    logger.log({**values, "lr": optimizer.param_groups[0]["lr"]}, global_step, "train")
                if should_step and global_step % int(base_cfg.train.image_every_optimizer_steps) == 0:
                    save_labeled_grid_v2(
                        tensors=(
                            content, image_a, image_b,
                            pair_result["state_a"]["canonical"],
                            pair_result["output_ab"], image_b,
                            pair_result["output_ba"], image_a,
                        ),
                        path=paths["images"] / f"train_step_{global_step:08d}.png",
                        labels=(
                            "original_COCO", "LUT_A", "LUT_B", "canonical_A",
                            "A_to_B", "ground_truth_B", "B_to_A", "ground_truth_A",
                        ),
                        title=f"Query-Adaptive Stage1 - optimizer step {global_step}",
                        max_items=int(base_cfg.validation.max_visual_items),
                    )

            if train_samples == 0:
                raise RuntimeError("Training loader produced no batches")
            logger.log(average(train_totals, train_samples), global_step, "train_epoch")
            val_values = validate(
                model, val_loader, criterion, base_cfg, device, epoch, paths, args.max_val_batches
            )
            logger.log(val_values, global_step, "val")
            scheduler.step()
            print("Validation:", " ".join(f"{k}={v:.6f}" for k, v in val_values.items()), flush=True)

            # Match the dense baseline checkpoint-selection objective exactly.
            selection_metric = val_values["base_neural_preset_total"]
            extra = {
                "protocol": "query_adaptive_hierarchical_stage1",
                "validation_total": val_values["total"],
                "validation_base_neural_preset_total": selection_metric,
                "validation_mean_bidirectional_psnr": val_values["mean_bidirectional_psnr"],
                "normalization_active_32": val_values.get("norm_active_32", float("nan")),
                "style_active_32": val_values.get("style_active_32", float("nan")),
                "fixed_blue_dominance": val_values["fixed_blue_dominance"],
            }
            if selection_metric < best_metric:
                best_metric = selection_metric
                save_checkpoint(paths["checkpoints"] / "best.pth", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, extra=extra)
            save_checkpoint(latest_path, model, optimizer, scheduler, scaler, epoch, global_step, best_metric, extra=extra)
            if (epoch + 1) % int(base_cfg.train.checkpoint_every_epochs) == 0:
                save_checkpoint(paths["checkpoints"] / f"epoch_{epoch:04d}.pth", model, optimizer, scheduler, scaler, epoch, global_step, best_metric, extra=extra)
    finally:
        logger.close()


if __name__ == "__main__":
    main()
