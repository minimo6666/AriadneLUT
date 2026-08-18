
from __future__ import annotations

import torch


def srgb_to_linear_channel_first(rgb: torch.Tensor) -> torch.Tensor:
    """Extended sRGB -> linear RGB. Input [B,3,...]."""
    if rgb.ndim < 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have channel dimension 3 at dim=1")
    return torch.where(
        rgb <= 0.04045,
        rgb / 12.92,
        torch.pow(torch.clamp((rgb + 0.055) / 1.055, min=1e-12), 2.4),
    )


def linear_to_srgb_channel_first(rgb: torch.Tensor) -> torch.Tensor:
    """Extended linear RGB -> sRGB. Input [B,3,...]. Does not clamp."""
    if rgb.ndim < 2 or rgb.shape[1] != 3:
        raise ValueError("rgb must have channel dimension 3 at dim=1")
    return torch.where(
        rgb <= 0.0031308,
        12.92 * rgb,
        1.055 * torch.pow(torch.clamp(rgb, min=1e-12), 1.0 / 2.4) - 0.055,
    )


def linear_rgb_to_oklab_channel_first(rgb: torch.Tensor) -> torch.Tensor:
    """Linear RGB -> Oklab. Input/output [B,3,...]."""
    r, g, b = rgb[:, 0], rgb[:, 1], rgb[:, 2]
    l = 0.4122214708 * r + 0.5363325363 * g + 0.0514459929 * b
    m = 0.2119034982 * r + 0.6806995451 * g + 0.1073969566 * b
    s = 0.0883024619 * r + 0.2817188376 * g + 0.6299787005 * b

    l_ = torch.sign(l) * torch.pow(torch.abs(l).clamp_min(1e-12), 1.0 / 3.0)
    m_ = torch.sign(m) * torch.pow(torch.abs(m).clamp_min(1e-12), 1.0 / 3.0)
    s_ = torch.sign(s) * torch.pow(torch.abs(s).clamp_min(1e-12), 1.0 / 3.0)

    L = 0.2104542553 * l_ + 0.7936177850 * m_ - 0.0040720468 * s_
    a = 1.9779984951 * l_ - 2.4285922050 * m_ + 0.4505937099 * s_
    bb = 0.0259040371 * l_ + 0.7827717662 * m_ - 0.8086757660 * s_
    return torch.stack([L, a, bb], dim=1)


def oklab_to_linear_rgb_channel_first(lab: torch.Tensor) -> torch.Tensor:
    """Oklab -> linear RGB. Input/output [B,3,...]."""
    if lab.ndim < 2 or lab.shape[1] != 3:
        raise ValueError("lab must have channel dimension 3 at dim=1")
    L, a, b = lab[:, 0], lab[:, 1], lab[:, 2]

    l_ = L + 0.3963377774 * a + 0.2158037573 * b
    m_ = L - 0.1055613458 * a - 0.0638541728 * b
    s_ = L - 0.0894841775 * a - 1.2914855480 * b

    l, m, s = l_ ** 3, m_ ** 3, s_ ** 3
    r = +4.0767416621 * l - 3.3077115913 * m + 0.2309699292 * s
    g = -1.2684380046 * l + 2.6097574011 * m - 0.3413193965 * s
    bb = -0.0041960863 * l - 0.7034186147 * m + 1.7076147010 * s
    return torch.stack([r, g, bb], dim=1)


def srgb_to_oklab_channel_first(rgb: torch.Tensor) -> torch.Tensor:
    return linear_rgb_to_oklab_channel_first(srgb_to_linear_channel_first(rgb))


def oklab_to_srgb_channel_first(lab: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      raw_srgb: unclamped sRGB [B,3,...]
      linear_rgb: unclamped linear RGB [B,3,...]
    """
    linear = oklab_to_linear_rgb_channel_first(lab)
    return linear_to_srgb_channel_first(linear), linear


def linear_rgb_range_penalty(linear_rgb: torch.Tensor) -> torch.Tensor:
    """Squared out-of-gamut penalty before gamma encoding."""
    low = torch.relu(-linear_rgb)
    high = torch.relu(linear_rgb - 1.0)
    return (low.square() + high.square()).mean()
