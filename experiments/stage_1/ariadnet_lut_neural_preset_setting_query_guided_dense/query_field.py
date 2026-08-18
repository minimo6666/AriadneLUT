from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass(frozen=True)
class QueryPyramid:
    """Exact/optionally-subsampled LUT query masses aligned to 8/16/32 vertices."""

    q8: torch.Tensor
    q16: torch.Tensor
    q32: torch.Tensor
    a8: torch.Tensor
    a16: torch.Tensor
    a32: torch.Tensor

    def focus(self, resolution: int) -> torch.Tensor:
        if int(resolution) == 8:
            return self.a8
        if int(resolution) == 16:
            return self.a16
        if int(resolution) == 32:
            return self.a32
        raise ValueError(f"Unsupported query resolution: {resolution}")

    def mass(self, resolution: int) -> torch.Tensor:
        if int(resolution) == 8:
            return self.q8
        if int(resolution) == 16:
            return self.q16
        if int(resolution) == 32:
            return self.q32
        raise ValueError(f"Unsupported query resolution: {resolution}")


def _sample_pixels(image: torch.Tensor, max_query_pixels: int) -> torch.Tensor:
    """Return real image RGB samples [B,N,3]. No resized/synthetic colors are introduced."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")

    pixels = image.detach().float().clamp(0.0, 1.0).permute(0, 2, 3, 1).reshape(image.shape[0], -1, 3)
    maximum = int(max_query_pixels)
    if maximum > 0 and pixels.shape[1] > maximum:
        # Deterministic, image-independent sampling keeps the conditioning reproducible.
        index = torch.linspace(
            0,
            pixels.shape[1] - 1,
            steps=maximum,
            device=pixels.device,
        ).round().long()
        pixels = pixels.index_select(1, index)
    return pixels


@torch.no_grad()
def exact_query_mass_from_pixels(pixels: torch.Tensor, resolution: int) -> torch.Tensor:
    """Accumulate exact trilinear lookup mass on an align_corners=True LUT lattice.

    Args:
        pixels: [B,N,3] RGB values in [0,1].
        resolution: number of LUT vertices per axis (8,16,32 here).

    Returns:
        Dense query mass [B,1,K,K,K] with tensor layout [z=B, y=G, x=R].
        Each sample sums to one (up to floating-point error).
    """
    if pixels.ndim != 3 or pixels.shape[-1] != 3:
        raise ValueError("pixels must have shape [B,N,3]")

    k = int(resolution)
    if k < 2:
        raise ValueError("resolution must be >= 2")

    rgb = pixels.float().clamp(0.0, 1.0)
    position = rgb * float(k - 1)
    lo = torch.floor(position).long()
    hi = (lo + 1).clamp_max(k - 1)
    frac = position - lo.float()

    # RGB component order in pixels is R,G,B. Flattened LUT storage is z,y,x = B,G,R.
    r0, g0, b0 = lo.unbind(dim=-1)
    r1, g1, b1 = hi.unbind(dim=-1)
    fr, fg, fb = frac.unbind(dim=-1)
    wr = (1.0 - fr, fr)
    wg = (1.0 - fg, fg)
    wb = (1.0 - fb, fb)
    rr = (r0, r1)
    gg = (g0, g1)
    bb = (b0, b1)

    batch, count, _ = rgb.shape
    mass = torch.zeros(batch, k * k * k, device=rgb.device, dtype=torch.float32)
    normalizer = 1.0 / float(max(count, 1))

    for iz in range(2):
        for iy in range(2):
            for ix in range(2):
                flat = bb[iz] * (k * k) + gg[iy] * k + rr[ix]
                weight = wb[iz] * wg[iy] * wr[ix] * normalizer
                mass.scatter_add_(1, flat, weight)

    # Guard tiny accumulation drift while preserving the physical distribution.
    mass = mass / mass.sum(dim=1, keepdim=True).clamp_min(1e-12)
    return mass.view(batch, 1, k, k, k)


@torch.no_grad()
def focus_from_mass(mass: torch.Tensor, transform: str = "log1p") -> torch.Tensor:
    """Convert probability mass into a stable [0,1] soft focus field.

    We deliberately avoid a hard support mask in this PSNR-oriented experiment.
    `log1p` compresses the very large dynamic range of color frequencies so rare-but-real
    queried colors are not erased while dominant colors remain emphasized.
    """
    if mass.ndim != 5 or mass.shape[1] != 1:
        raise ValueError("mass must have shape [B,1,K,K,K]")

    k = int(mass.shape[-1])
    key = str(transform).lower()
    if key == "log1p":
        value = torch.log1p(mass.float() * float(k**3))
    elif key == "linear":
        value = mass.float()
    elif key == "sqrt":
        value = torch.sqrt(mass.float().clamp_min(0.0))
    else:
        raise ValueError(f"Unknown focus transform: {transform}")

    maximum = value.flatten(1).amax(dim=1).view(-1, 1, 1, 1, 1)
    return (value / maximum.clamp_min(1e-12)).clamp_(0.0, 1.0)


@torch.no_grad()
def build_query_pyramid(
    image: torch.Tensor,
    max_query_pixels: int = 0,
    focus_transform: str = "log1p",
) -> QueryPyramid:
    """Build exact query conditioning at the *original* 8/16/32 Ariadne scales.

    The three original vertex grids are not nested, so we intentionally measure each
    resolution directly from the same real pixels. This keeps the Dense baseline's
    8->16->32 architecture unchanged and introduces only the new query information.
    """
    pixels = _sample_pixels(image, max_query_pixels=max_query_pixels)
    q8 = exact_query_mass_from_pixels(pixels, 8)
    q16 = exact_query_mass_from_pixels(pixels, 16)
    q32 = exact_query_mass_from_pixels(pixels, 32)
    return QueryPyramid(
        q8=q8,
        q16=q16,
        q32=q32,
        a8=focus_from_mass(q8, focus_transform),
        a16=focus_from_mass(q16, focus_transform),
        a32=focus_from_mass(q32, focus_transform),
    )
