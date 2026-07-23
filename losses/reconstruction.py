from __future__ import annotations

import torch
import torch.nn.functional as F


def charbonnier(prediction, target, eps: float = 1e-3):
    return torch.sqrt((prediction - target).square() + eps * eps).mean()


def rgb_to_yuv(image):
    r, g, b = image[:, 0:1], image[:, 1:2], image[:, 2:3]
    y = 0.299 * r + 0.587 * g + 0.114 * b
    u = -0.14713 * r - 0.28886 * g + 0.436 * b
    v = 0.615 * r - 0.51499 * g - 0.10001 * b
    return torch.cat([y, u, v], dim=1)


def gradient_loss(prediction, target):
    pred_dx = prediction[:, :, :, 1:] - prediction[:, :, :, :-1]
    pred_dy = prediction[:, :, 1:, :] - prediction[:, :, :-1, :]
    tgt_dx = target[:, :, :, 1:] - target[:, :, :, :-1]
    tgt_dy = target[:, :, 1:, :] - target[:, :, :-1, :]
    return F.l1_loss(pred_dx, tgt_dx) + F.l1_loss(pred_dy, tgt_dy)


def reconstruction_components(prediction, target):
    pred_yuv = rgb_to_yuv(prediction)
    tgt_yuv = rgb_to_yuv(target)
    pred_low = F.avg_pool2d(prediction, 9, stride=1, padding=4)
    tgt_low = F.avg_pool2d(target, 9, stride=1, padding=4)
    return {
        "charbonnier": charbonnier(prediction, target),
        "mse": F.mse_loss(prediction, target),
        "chroma": F.l1_loss(pred_yuv[:, 1:], tgt_yuv[:, 1:]),
        "low_frequency": F.l1_loss(pred_low, tgt_low),
        "gradient": gradient_loss(prediction, target),
    }


def edge_preservation(content, output):
    return gradient_loss(output, content)


def luminance_correlation(content, output):
    c = rgb_to_yuv(content)[:, 0].flatten(1)
    o = rgb_to_yuv(output)[:, 0].flatten(1)
    c = c - c.mean(dim=1, keepdim=True)
    o = o - o.mean(dim=1, keepdim=True)
    return (1.0 - F.cosine_similarity(c, o, dim=1)).mean()
