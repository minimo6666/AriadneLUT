from __future__ import annotations

import torch
import torch.nn.functional as F


def lut_smoothness(lut):
    d2x = lut[:, :, 2:, :, :] - 2 * lut[:, :, 1:-1, :, :] + lut[:, :, :-2, :, :]
    d2y = lut[:, :, :, 2:, :] - 2 * lut[:, :, :, 1:-1, :] + lut[:, :, :, :-2, :]
    d2z = lut[:, :, :, :, 2:] - 2 * lut[:, :, :, :, 1:-1] + lut[:, :, :, :, :-2]
    return d2x.square().mean() + d2y.square().mean() + d2z.square().mean()


def lut_monotonicity(lut):
    dx = lut[:, :, 1:, :, :] - lut[:, :, :-1, :, :]
    dy = lut[:, :, :, 1:, :] - lut[:, :, :, :-1, :]
    dz = lut[:, :, :, :, 1:] - lut[:, :, :, :, :-1]
    return F.relu(-dx).mean() + F.relu(-dy).mean() + F.relu(-dz).mean()


def lut_range(lut):
    return F.relu(-lut).square().mean() + F.relu(lut - 1.0).square().mean()
