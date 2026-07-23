from __future__ import annotations

from pathlib import Path
import shutil

import torch

from utils.seed import seed_everything


def resolve_device(name: str):
    if str(name).startswith("cuda") and torch.cuda.is_available():
        return torch.device(name)
    return torch.device("cpu")


def prepare(cfg):
    seed_everything(int(cfg.experiment.seed))
    root = Path(cfg.experiment.output_dir)
    paths = {
        "root": root,
        "checkpoints": root / "checkpoints",
        "images": root / "images",
        "logs": root / "logs",
    }
    for path in paths.values():
        path.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cfg.config_path, root / "config.yaml")
    return paths


def move(value, device):
    if torch.is_tensor(value):
        return value.to(device, non_blocking=True)
    if isinstance(value, dict):
        return {key: move(item, device) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return type(value)(move(item, device) for item in value)
    return value


def scalar_dict(values):
    return {
        key: float(value.detach().cpu()) if torch.is_tensor(value) else float(value)
        for key, value in values.items()
    }


def trainable_parameters(model):
    return sum(parameter.numel() for parameter in model.parameters() if parameter.requires_grad)
