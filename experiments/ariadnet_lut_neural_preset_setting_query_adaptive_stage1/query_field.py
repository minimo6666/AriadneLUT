from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class QuerySupport:
    """Per-image query support on one K^3 LUT lattice.

    Tensor layout follows PyTorch grid_sample / Ariadne's LUT convention:
    flattened lattice order is [B-axis, G-axis, R-axis], while the explicit
    coordinate feature is always returned as (R, G, B).
    """

    dimension: int
    mass: torch.Tensor          # [B, K^3], sums to one per image
    local_mass: torch.Tensor    # [B, K^3], local 3x3x3 average of mass
    core_mask: torch.Tensor     # [B, K^3], bool
    active_mask: torch.Tensor   # [B, K^3], bool (core + safety shell)
    gate: torch.Tensor          # [B, K^3], [0,1], tapered shell
    summary: torch.Tensor       # [B, 8] = meanRGB, stdRGB, H_norm, N_eff_ratio

    @property
    def active_ratio(self) -> torch.Tensor:
        return self.active_mask.float().mean(dim=1)

    @property
    def core_ratio(self) -> torch.Tensor:
        return self.core_mask.float().mean(dim=1)

    @property
    def active_count(self) -> torch.Tensor:
        return self.active_mask.sum(dim=1)


def lattice_coordinates(
    dimension: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Return K^3 coordinates [N,3] in flattened LUT order as (R,G,B)."""
    k = int(dimension)
    values = torch.linspace(0.0, 1.0, k, device=device, dtype=dtype)
    blue, green, red = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack((red, green, blue), dim=-1).reshape(-1, 3)


def _sample_actual_pixels(image: torch.Tensor, max_pixels: int) -> torch.Tensor:
    """Return actual RGB pixels without creating interpolated colors.

    We deliberately subsample existing pixels rather than resize the image. A
    bilinear/area resize would manufacture colors that were never queried by the
    real image and can artificially inflate LUT occupancy.
    """
    pixels = image.permute(0, 2, 3, 1).reshape(image.shape[0], -1, 3)
    total = pixels.shape[1]
    limit = int(max_pixels)
    if limit <= 0 or total <= limit:
        return pixels
    index = torch.linspace(
        0,
        total - 1,
        steps=limit,
        device=image.device,
        dtype=torch.float32,
    ).round().long()
    return pixels.index_select(1, index)


@torch.no_grad()
def exact_query_mass(
    image: torch.Tensor,
    dimension: int,
    *,
    max_pixels: int = 16384,
) -> torch.Tensor:
    """Compute exact trilinear query mass on a K^3 LUT lattice.

    This matches Ariadne's `grid_sample(..., padding_mode='border',
    align_corners=True)` coordinate convention. Values outside [0,1] are
    therefore clamped to the border before occupancy is accumulated.

    Returns:
        mass: [B, K^3], non-negative and normalized to sum to one.
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    k = int(dimension)
    if k < 2:
        raise ValueError("dimension must be >= 2")

    pixels = _sample_actual_pixels(image.detach().float(), int(max_pixels))
    pixels = pixels.clamp(0.0, 1.0)
    scaled = pixels * float(k - 1)
    lo = torch.floor(scaled).long()
    hi = (lo + 1).clamp(max=k - 1)
    frac = (scaled - lo.float()).clamp(0.0, 1.0)

    batch = image.shape[0]
    n_vertices = k ** 3
    mass = torch.zeros(batch, n_vertices, device=image.device, dtype=torch.float32)

    # RGB coordinates map to LUT tensor axes W/H/D respectively. Flattening a
    # [D(B), H(G), W(R)] lattice gives flat = B*K^2 + G*K + R.
    for choose_hi_r in (0, 1):
        r = hi[..., 0] if choose_hi_r else lo[..., 0]
        wr = frac[..., 0] if choose_hi_r else (1.0 - frac[..., 0])
        for choose_hi_g in (0, 1):
            g = hi[..., 1] if choose_hi_g else lo[..., 1]
            wg = frac[..., 1] if choose_hi_g else (1.0 - frac[..., 1])
            for choose_hi_b in (0, 1):
                b = hi[..., 2] if choose_hi_b else lo[..., 2]
                wb = frac[..., 2] if choose_hi_b else (1.0 - frac[..., 2])
                weight = (wr * wg * wb).float()
                flat = b * (k * k) + g * k + r
                mass.scatter_add_(1, flat, weight)

    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mass


@torch.no_grad()
def query_summary(mass: torch.Tensor, dimension: int) -> torch.Tensor:
    """Compact global statistics of a joint RGB query distribution."""
    coords = lattice_coordinates(
        int(dimension), device=mass.device, dtype=mass.dtype
    )
    mean = mass @ coords
    second = mass @ (coords * coords)
    std = (second - mean * mean).clamp_min(0.0).sqrt()
    entropy = -(mass.clamp_min(1e-12) * mass.clamp_min(1e-12).log()).sum(dim=1)
    n = mass.shape[1]
    entropy_norm = entropy / max(math.log(float(n)), 1e-12)
    effective_ratio = entropy.exp() / float(n)
    return torch.cat(
        (mean, std, entropy_norm[:, None], effective_ratio[:, None]), dim=1
    )


@torch.no_grad()
def support_from_mass(
    mass: torch.Tensor,
    dimension: int,
    *,
    mass_fraction: float = 0.99,
    dilation: int = 1,
    shell_decay: float = 0.35,
) -> QuerySupport:
    """Find the smallest high-mass support and add a tapered safety shell.

    The support is non-differentiable by design. It is an allocation decision,
    not a learned color transform. Gradients still flow through the generated
    LUT values and through LUT application to the image.
    """
    if mass.ndim != 2:
        raise ValueError("mass must have shape [B,K^3]")
    k = int(dimension)
    if mass.shape[1] != k ** 3:
        raise ValueError("mass second dimension does not match K^3")
    fraction = float(mass_fraction)
    if not 0.0 < fraction <= 1.0:
        raise ValueError("mass_fraction must be in (0,1]")

    sorted_mass, order = torch.sort(mass, dim=1, descending=True)
    cumulative = torch.cumsum(sorted_mass, dim=1)
    # Include the element that crosses the requested cumulative mass.
    keep_sorted = (cumulative - sorted_mass) < fraction
    core = torch.zeros_like(mass, dtype=torch.bool)
    core.scatter_(1, order, keep_sorted)

    core_3d = core.view(mass.shape[0], 1, k, k, k).float()
    active_3d = core_3d
    gate_3d = core_3d.clone()
    previous = core_3d
    for radius in range(1, int(dilation) + 1):
        expanded = F.max_pool3d(previous, kernel_size=3, stride=1, padding=1)
        new_shell = (expanded - previous).clamp(0.0, 1.0)
        gate_3d = gate_3d + (float(shell_decay) ** radius) * new_shell
        previous = expanded
        active_3d = expanded

    # A cheap local density feature; this is one-channel dense work only.
    mass_3d = mass.view(mass.shape[0], 1, k, k, k)
    local = F.avg_pool3d(mass_3d, kernel_size=3, stride=1, padding=1)

    return QuerySupport(
        dimension=k,
        mass=mass,
        local_mass=local.flatten(1),
        core_mask=core,
        active_mask=active_3d.flatten(1).bool(),
        gate=gate_3d.flatten(1).clamp(0.0, 1.0),
        summary=query_summary(mass, k),
    )


@torch.no_grad()
def build_query_support(
    image: torch.Tensor,
    dimension: int,
    *,
    mass_fraction: float,
    dilation: int,
    shell_decay: float,
    max_pixels: int,
) -> QuerySupport:
    mass = exact_query_mass(image, dimension, max_pixels=max_pixels)
    return support_from_mass(
        mass,
        dimension,
        mass_fraction=mass_fraction,
        dilation=dilation,
        shell_decay=shell_decay,
    )
