from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F


def _cfg_value(cfg: Any, name: str, default):
    return getattr(cfg, name, default)


def charbonnier_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-3,
) -> torch.Tensor:
    difference = prediction.float() - target.float()
    return torch.sqrt(difference.square() + float(eps) ** 2).mean()


def low_frequency_l1(
    prediction: torch.Tensor,
    target: torch.Tensor,
    kernel_size: int = 16,
) -> torch.Tensor:
    k = max(int(kernel_size), 1)
    if k == 1:
        return F.l1_loss(prediction.float(), target.float())
    pred_low = F.avg_pool2d(prediction.float(), kernel_size=k, stride=k)
    target_low = F.avg_pool2d(target.float(), kernel_size=k, stride=k)
    return F.l1_loss(pred_low, target_low)


def color_moment_loss(
    prediction: torch.Tensor,
    target: torch.Tensor,
) -> torch.Tensor:
    """Match global RGB mean/std; useful because the task is color grading."""
    pred = prediction.float()
    tgt = target.float()
    dims = (2, 3)
    pred_mean = pred.mean(dim=dims)
    tgt_mean = tgt.mean(dim=dims)
    pred_std = pred.var(dim=dims, unbiased=False).add(1e-6).sqrt()
    tgt_std = tgt.var(dim=dims, unbiased=False).add(1e-6).sqrt()
    return F.l1_loss(pred_mean, tgt_mean) + F.l1_loss(pred_std, tgt_std)


def lut_first_order_smoothness(lut: torch.Tensor) -> torch.Tensor:
    lut = lut.float()
    dd = lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]
    dh = lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]
    dw = lut[:, :, :, :, 1:] - lut[:, :, :, :, :-1]
    return (dd.square().mean() + dh.square().mean() + dw.square().mean()) / 3.0


def lut_curvature(lut: torch.Tensor) -> torch.Tensor:
    lut = lut.float()
    d2_d = lut[:, :, 2:, :, :] - 2.0 * lut[:, :, 1:-1, :, :] + lut[:, :, :-2, :, :]
    d2_h = lut[:, :, :, 2:, :] - 2.0 * lut[:, :, :, 1:-1, :] + lut[:, :, :, :-2, :]
    d2_w = lut[:, :, :, :, 2:] - 2.0 * lut[:, :, :, :, 1:-1] + lut[:, :, :, :, :-2]
    return (d2_d.square().mean() + d2_h.square().mean() + d2_w.square().mean()) / 3.0


def lut_range_penalty(lut: torch.Tensor) -> torch.Tensor:
    lut = lut.float()
    return F.relu(-lut).square().mean() + F.relu(lut - 1.0).square().mean()


class LPIPSLoss(nn.Module):
    def __init__(self, net: str = "alex") -> None:
        super().__init__()
        try:
            import lpips  # type: ignore
        except ImportError as exc:
            raise ImportError(
                "LPIPS loss is enabled but package 'lpips' is unavailable. "
                "Install it in the DavinciLUT environment with: pip install lpips"
            ) from exc
        self.metric = lpips.LPIPS(net=str(net), verbose=False)
        self.metric.eval()
        for parameter in self.metric.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        # LPIPS is always a frozen evaluator.
        super().train(False)
        self.metric.eval()
        return self

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        pred = prediction.float().clamp(0.0, 1.0) * 2.0 - 1.0
        tgt = target.float().clamp(0.0, 1.0) * 2.0 - 1.0
        return self.metric(pred, tgt).mean()


class MovieNetPairStage2Objective(nn.Module):
    """Exact-GT Stage-2 objective.

    The original movie frame is pixel-aligned GT, so appearance supervision is
    direct. LPIPS is auxiliary; the main signal remains exact photometric error.
    """

    def __init__(self, cfg: Any) -> None:
        super().__init__()
        self.w_charbonnier = float(_cfg_value(cfg, "charbonnier", 1.0))
        self.w_lpips = float(_cfg_value(cfg, "lpips", 0.20))
        self.w_low_frequency = float(_cfg_value(cfg, "low_frequency", 0.50))
        self.w_color_moments = float(_cfg_value(cfg, "color_moments", 0.50))
        self.w_delta_l2 = float(_cfg_value(cfg, "delta_l2", 0.01))
        self.w_delta_smooth = float(_cfg_value(cfg, "delta_smooth", 0.05))
        self.w_final_curvature = float(_cfg_value(cfg, "final_lut_curvature", 0.05))
        self.w_range = float(_cfg_value(cfg, "lut_range", 1.0))
        self.low_frequency_kernel = int(_cfg_value(cfg, "low_frequency_kernel", 16))

        self.lpips_loss = None
        if self.w_lpips > 0.0:
            self.lpips_loss = LPIPSLoss(net=str(_cfg_value(cfg, "lpips_net", "alex")))

    def forward(
        self,
        result: dict[str, torch.Tensor],
        target: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        output = result["output"]
        delta_lut = result["delta_lut"]
        final_lut = result["final_lut"]

        losses: dict[str, torch.Tensor] = {}
        losses["charbonnier"] = charbonnier_loss(output, target)
        losses["low_frequency"] = low_frequency_l1(
            output,
            target,
            kernel_size=self.low_frequency_kernel,
        )
        losses["color_moments"] = color_moment_loss(output, target)
        losses["delta_l2"] = delta_lut.float().square().mean()
        losses["delta_smooth"] = lut_first_order_smoothness(delta_lut)
        losses["final_lut_curvature"] = lut_curvature(final_lut)
        losses["lut_range"] = lut_range_penalty(final_lut)

        if self.lpips_loss is not None:
            losses["lpips"] = self.lpips_loss(output, target)
        else:
            losses["lpips"] = output.new_zeros(())

        total = (
            self.w_charbonnier * losses["charbonnier"]
            + self.w_lpips * losses["lpips"]
            + self.w_low_frequency * losses["low_frequency"]
            + self.w_color_moments * losses["color_moments"]
            + self.w_delta_l2 * losses["delta_l2"]
            + self.w_delta_smooth * losses["delta_smooth"]
            + self.w_final_curvature * losses["final_lut_curvature"]
            + self.w_range * losses["lut_range"]
        )
        losses["total"] = total
        return losses
