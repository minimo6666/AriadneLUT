
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
REPO_ROOT = EXPERIMENT_DIR.parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from metrics import psnr
from utils.checkpoint import load_checkpoint, save_checkpoint
from utils.config import load_config
from utils.image_v2 import save_labeled_grid_v2
from utils.logger import ScalarLogger
from utils.seed import seed_everything

from neural_preset_dataset import build_loader
from ariadne_tonal_chromatic import AriadneTonalChromaticStage1
from neural_preset_loss_tonal_chromatic import TonalChromaticStage1Objective


BASE_CONFIG = (
    REPO_ROOT
    / "experiments/stage_1/ariadnet_lut_neural_preset_setting/config.yaml"
)
DEFAULT_OUTPUT = str(EXPERIMENT_DIR / "runs/main_gpu2_b24")


def parse_args():
    p = argparse.ArgumentParser(
        description="From-scratch Stage-1 tonal/chromatic factorization experiment."
    )
    p.add_argument("--config", default=str(BASE_CONFIG))
    p.add_argument("--output-dir", default=DEFAULT_OUTPUT)
    p.add_argument("--resume", default="")
    p.add_argument("--epochs", type=int, default=0)
    p.add_argument("--max-train-batches", type=int, default=0)
    p.add_argument("--max-val-batches", type=int, default=0)
    p.add_argument("--num-workers", type=int, default=-1)
    p.add_argument("--batch-size", type=int, default=0)

    p.add_argument("--lambda-tonal", type=float, default=0.5)
    p.add_argument("--lambda-chromatic", type=float, default=1.0)
    p.add_argument("--lambda-leakage", type=float, default=0.5)
    p.add_argument("--lambda-slider", type=float, default=0.5)
    p.add_argument("--lambda-perceptual", type=float, default=0.03)
    p.add_argument("--perceptual-size", type=int, default=128)
    p.add_argument("--lambda-lut-smooth", type=float, default=0.02)
    p.add_argument("--lambda-gamut", type=float, default=0.02)
    p.add_argument("--chromatic-scale-multiplier", type=float, default=0.5)
    return p.parse_args()


def resolve_path(value: str) -> Path:
    x = Path(value)
    return x if x.is_absolute() else REPO_ROOT / x


def prepare_run(cfg, args) -> dict[str, Path]:
    root = resolve_path(str(args.output_dir))
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    for x in paths.values():
        x.mkdir(parents=True, exist_ok=True)
    shutil.copy2(args.config, root / "base_stage1_config.yaml")
    extra = {
        "experiment": "tonal_chromatic_factorized_stage1",
        "lambda_tonal": args.lambda_tonal,
        "lambda_chromatic": args.lambda_chromatic,
        "lambda_leakage": args.lambda_leakage,
        "lambda_slider": args.lambda_slider,
        "lambda_perceptual": args.lambda_perceptual,
        "perceptual_size": args.perceptual_size,
        "lambda_lut_smooth": args.lambda_lut_smooth,
        "lambda_gamut": args.lambda_gamut,
        "chromatic_scale_multiplier": args.chromatic_scale_multiplier,
        "output_dir": str(root),
    }
    (root / "tonal_chromatic_config.json").write_text(
        json.dumps(extra, indent=2), encoding="utf-8"
    )
    return paths


def move(x, device):
    return x.to(device, non_blocking=True)


def scalar_dict(values):
    return {
        k: float(v.detach().float().cpu()) if torch.is_tensor(v) else float(v)
        for k, v in values.items()
    }


def accumulate(totals, values, weight):
    for k, v in values.items():
        if math.isfinite(float(v)):
            totals[k] = totals.get(k, 0.0) + float(v) * int(weight)


def average(totals, count):
    if count <= 0:
        raise RuntimeError("Cannot average zero samples")
    return {k: v / count for k, v in totals.items()}


def add_metrics(values, result, image_a, image_b):
    oa = result["output_ab"]
    ob = result["output_ba"]
    ca = result["state_a"]["canonical"]
    cb = result["state_b"]["canonical"]
    values.update({
        "psnr_a_to_b": float(psnr(oa.clamp(0, 1), image_b)),
        "psnr_b_to_a": float(psnr(ob.clamp(0, 1), image_a)),
        "canonical_psnr": float(psnr(ca.clamp(0, 1), cb.clamp(0, 1))),
        "tonal_only_psnr_a_to_b": float(
            psnr(result["output_ab_tonal_only"].clamp(0, 1), image_b)
        ),
        "tonal_only_psnr_b_to_a": float(
            psnr(result["output_ba_tonal_only"].clamp(0, 1), image_a)
        ),
        "chroma_only_psnr_a_to_b": float(
            psnr(result["output_ab_chroma_only"].clamp(0, 1), image_b)
        ),
        "chroma_only_psnr_b_to_a": float(
            psnr(result["output_ba_chroma_only"].clamp(0, 1), image_a)
        ),
        "output_out_of_range_ratio": float(
            torch.cat([oa, ob]).sub(0.5).abs().gt(0.5).float().mean()
        ),
    })
    values["mean_bidirectional_psnr"] = 0.5 * (
        values["psnr_a_to_b"] + values["psnr_b_to_a"]
    )


def save_grid(paths, prefix, content, a, b, result, gt_b, gt_a, max_items):
    save_labeled_grid_v2(
        tensors=(
            content,
            a,
            b,
            result["output_ab"],
            gt_b,
            result["output_ab_tonal_only"],
            result["output_ab_chroma_only"],
            result["output_ba"],
            gt_a,
        ),
        path=paths["images"] / f"{prefix}.png",
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
        title="Tonal-Chromatic factorized Stage-1",
        max_items=int(max_items),
    )


@torch.no_grad()
def validate(model, loader, criterion, cfg, device, epoch, paths, max_batches):
    model.eval()
    totals = {}
    count = 0
    first = None
    iterator = loader if max_batches <= 0 else islice(loader, max_batches)
    for batch in tqdm(
        iterator, desc=f"ValTC {epoch:03d}", total=(max_batches or len(loader))
    ):
        content = move(batch["content"], device)
        a = move(batch["image_a"], device)
        b = move(batch["image_b"], device)
        # Deterministic interior slider point during validation. Endpoints are
        # already covered by full / tonal-only / chroma-only outputs.
        alpha_t = torch.full((a.shape[0],), 0.5, device=device, dtype=a.dtype)
        alpha_c = torch.full((a.shape[0],), 0.5, device=device, dtype=a.dtype)
        result = model.forward_pair(
            a, b,
            controlled_tonal_strength=alpha_t,
            controlled_chromatic_strength=alpha_c,
        )
        losses = criterion(result, a, b)
        values = scalar_dict(losses)
        add_metrics(values, result, a, b)
        bs = int(a.shape[0])
        accumulate(totals, values, bs)
        count += bs
        if first is None:
            first = (content, a, b, result)

    if first is None:
        raise RuntimeError("Validation loader produced no batches")
    content, a, b, result = first
    save_grid(
        paths,
        f"val_epoch_{epoch:04d}",
        content, a, b, result, b, a,
        cfg.validation.max_visual_items,
    )
    return average(totals, count)


def write_manifest(cfg, paths, train_dataset, val_dataset):
    payload = {
        "protocol": "Stage1 same-content dual-LUT + tonal/chromatic Style factorization",
        "base_config": str(BASE_CONFIG),
        "coco_root": str(cfg.data.coco_root),
        "train_images": len(train_dataset),
        "validation_images": len(val_dataset),
        "lut_root": str(cfg.data.lut_root),
        "lut_count": len(train_dataset.lut_paths),
    }
    (paths["root"] / "data_manifest.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )


def main():
    args = parse_args()
    cfg = load_config(args.config)

    # Add one experiment-local model option without touching root config.
    cfg.model.chromatic_scale_multiplier = float(args.chromatic_scale_multiplier)

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
        raise RuntimeError("CUDA requested but unavailable")
    device = torch.device(str(cfg.device))
    paths = prepare_run(cfg, args)

    latest_path = paths["checkpoints"] / "latest.pth"
    if latest_path.exists() and not str(cfg.train.resume):
        raise RuntimeError(
            f"Refusing to overwrite {paths['root']}; set --resume or new --output-dir"
        )

    print("Loading original Stage-1 Neural Preset train data...", flush=True)
    train_loader, train_dataset = build_loader(
        cfg, str(cfg.data.train_split), train=True
    )
    print("Loading original Stage-1 validation data...", flush=True)
    val_loader, val_dataset = build_loader(
        cfg, str(cfg.data.val_split), train=False
    )
    write_manifest(cfg, paths, train_dataset, val_dataset)

    model = AriadneTonalChromaticStage1(cfg.model).to(device)
    criterion = TonalChromaticStage1Objective(
        lambda_consistency=float(cfg.loss.lambda_consistency),
        lambda_tonal=args.lambda_tonal,
        lambda_chromatic=args.lambda_chromatic,
        lambda_leakage=args.lambda_leakage,
        lambda_slider=args.lambda_slider,
        lambda_perceptual=args.lambda_perceptual,
        lambda_lut_smooth=args.lambda_lut_smooth,
        lambda_gamut=args.lambda_gamut,
        perceptual_size=args.perceptual_size,
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
    best_metric = float("-inf")
    if str(cfg.train.resume):
        ckpt = load_checkpoint(
            resolve_path(str(cfg.train.resume)),
            model, optimizer, scheduler, scaler,
            map_location=device,
        )
        start_epoch = int(ckpt.get("epoch", -1)) + 1
        global_step = int(ckpt.get("global_step", 0))
        best_metric = float(ckpt.get("best_metric", best_metric))

    print(
        f"Parameters={sum(p.numel() for p in model.parameters() if p.requires_grad):,}; "
        f"train={len(train_dataset):,}; val={len(val_dataset):,}; "
        f"LUTs={len(train_dataset.lut_paths)}; "
        f"batch={cfg.data.batch_size}; accum={cfg.data.gradient_accumulation_steps}",
        flush=True,
    )

    accumulation_steps = int(cfg.data.gradient_accumulation_steps)
    try:
        for epoch in range(start_epoch, int(cfg.train.epochs)):
            model.train()
            optimizer.zero_grad(set_to_none=True)
            totals = {}
            count = 0
            configured_total = len(train_loader)
            total_batches = (
                min(configured_total, args.max_train_batches)
                if args.max_train_batches > 0 else configured_total
            )
            iterator = (
                train_loader if args.max_train_batches <= 0
                else islice(train_loader, args.max_train_batches)
            )
            progress = tqdm(iterator, total=total_batches, desc=f"TrainTC {epoch:03d}")
            current_target = accumulation_steps

            for batch_index, batch in enumerate(progress):
                if batch_index % accumulation_steps == 0:
                    current_target = min(
                        accumulation_steps, total_batches - batch_index
                    )

                content = move(batch["content"], device)
                a = move(batch["image_a"], device)
                b = move(batch["image_b"], device)

                # Explicitly train the interior of the two control axes. Each
                # sample receives independent continuous slider values.
                alpha_t = torch.rand(a.shape[0], device=device, dtype=a.dtype)
                alpha_c = torch.rand(a.shape[0], device=device, dtype=a.dtype)

                with autocast(device_type=device.type, enabled=amp_enabled):
                    result = model.forward_pair(
                        a, b,
                        controlled_tonal_strength=alpha_t,
                        controlled_chromatic_strength=alpha_c,
                    )
                    losses = criterion(result, a, b)
                    scaled = losses["total"] / current_target

                scaler.scale(scaled).backward()
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
                add_metrics(values, result, a, b)
                bs = int(a.shape[0])
                accumulate(totals, values, bs)
                count += bs

                progress.set_postfix(
                    total=f"{values['total']:.4f}",
                    psnr=f"{values['mean_bidirectional_psnr']:.2f}",
                    tone=f"{values['tonal_match_loss']:.3f}",
                    chroma=f"{values['chromatic_match_loss']:.3f}",
                    leak=f"{values['disentanglement_leakage_loss']:.3f}",
                    slider=f"{values['slider_supervision_loss']:.3f}",
                    perc=f"{values['perceptual_loss']:.3f}",
                    smooth=f"{values['lut_smooth_loss']:.3f}",
                    step=global_step,
                )

                if should_step and global_step % int(
                    cfg.train.log_every_optimizer_steps
                ) == 0:
                    logger.log(
                        {**values, "lr": optimizer.param_groups[0]["lr"]},
                        global_step, "train",
                    )

                if should_step and global_step % int(
                    cfg.train.image_every_optimizer_steps
                ) == 0:
                    save_grid(
                        paths,
                        f"train_step_{global_step:08d}",
                        content, a, b, result, b, a,
                        cfg.validation.max_visual_items,
                    )

            train_values = average(totals, count)
            logger.log(train_values, global_step, "train_epoch")

            val = validate(
                model, val_loader, criterion, cfg, device, epoch,
                paths, args.max_val_batches,
            )
            logger.log(val, global_step, "val")
            scheduler.step()

            print(
                "Validation:",
                " ".join(f"{k}={v:.6f}" for k, v in val.items()),
                flush=True,
            )

            extra = {
                "protocol": "tonal_chromatic_factorized_stage1",
                "validation_total": val["total"],
                "validation_mean_bidirectional_psnr": val["mean_bidirectional_psnr"],
                "lambda_tonal": args.lambda_tonal,
                "lambda_chromatic": args.lambda_chromatic,
                "lambda_leakage": args.lambda_leakage,
                "lambda_slider": args.lambda_slider,
                "lambda_perceptual": args.lambda_perceptual,
                "perceptual_size": args.perceptual_size,
                "lambda_lut_smooth": args.lambda_lut_smooth,
                "lambda_gamut": args.lambda_gamut,
                "chromatic_scale_multiplier": args.chromatic_scale_multiplier,
            }

            # Select the comparison checkpoint using the reported fidelity metric,
            # not a weighted auxiliary-loss mixture.
            if val["mean_bidirectional_psnr"] > best_metric:
                best_metric = val["mean_bidirectional_psnr"]
                save_checkpoint(
                    paths["checkpoints"] / "best.pth",
                    model, optimizer, scheduler, scaler,
                    epoch, global_step, best_metric, extra=extra,
                )

            save_checkpoint(
                latest_path,
                model, optimizer, scheduler, scaler,
                epoch, global_step, best_metric, extra=extra,
            )
    finally:
        logger.close()


if __name__ == "__main__":
    main()
