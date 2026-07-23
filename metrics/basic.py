from __future__ import annotations

import torch


def psnr(prediction: torch.Tensor, target: torch.Tensor, eps: float = 1e-10):
    mse = (prediction.float() - target.float()).square().mean()
    return -10.0 * torch.log10(mse.clamp_min(eps))
