
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oklab_torch import (
    srgb_to_oklab_channel_first,
    oklab_to_srgb_channel_first,
    linear_rgb_range_penalty,
)


def identity_lut_rgb(dimension: int, device=None, dtype=None) -> torch.Tensor:
    values = torch.linspace(
        0.0, 1.0, int(dimension), device=device, dtype=dtype
    )
    blue, green, red = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack([red, green, blue], dim=0).unsqueeze(0)


def identity_oklab_lut(dimension: int, device=None, dtype=None) -> torch.Tensor:
    rgb = identity_lut_rgb(dimension, device=device, dtype=dtype)
    # Oklab numerical conversion is more stable in float32 under AMP.
    lab = srgb_to_oklab_channel_first(rgb.float())
    return lab.to(dtype=rgb.dtype)


class Residual3DBlockV2Local(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if int(channels) % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(int(channels), int(channels), 3, padding=1),
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(int(channels), int(channels), 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _init_trilinear_deconv(layer: nn.ConvTranspose3d) -> None:
    factor = 2
    size = 4
    center = (size - 1) / 2
    coordinates = torch.arange(size, dtype=torch.float32)
    one = (1.0 - torch.abs(coordinates - center) / factor).clamp_min(0.0)
    kernel = (
        one[:, None, None] * one[None, :, None] * one[None, None, :]
    )
    with torch.no_grad():
        layer.weight.zero_()
        for c in range(layer.in_channels):
            layer.weight[c, 0].copy_(kernel)
        if layer.bias is not None:
            layer.bias.zero_()


class Progressive3DLUTGeneratorV2Local(nn.Module):
    """Exact local copy of the baseline RGB progressive LUT generator."""
    def __init__(
        self,
        code_dim: int,
        cfg,
        residual_scale_8: float,
        residual_scale_16: float,
        residual_scale_32: float,
    ):
        super().__init__()
        channels = int(cfg.lut_channels)
        condition_dim = int(cfg.condition_dim)
        self.scale_8 = float(residual_scale_8)
        self.scale_16 = float(residual_scale_16)
        self.scale_32 = float(residual_scale_32)

        self.condition = nn.Sequential(
            nn.LayerNorm(int(code_dim)),
            nn.Linear(int(code_dim), condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.seed = nn.Linear(condition_dim, channels * 8 * 8 * 8)
        self.stage_8 = nn.Sequential(
            Residual3DBlockV2Local(channels),
            Residual3DBlockV2Local(channels),
        )
        self.to_delta_8 = nn.Conv3d(channels, 3, 3, padding=1)

        self.up_features_16 = nn.ConvTranspose3d(
            channels, channels, 4, stride=2, padding=1
        )
        self.up_lut_16 = nn.ConvTranspose3d(
            3, 3, 4, stride=2, padding=1, groups=3
        )
        self.stage_16 = nn.Sequential(
            Residual3DBlockV2Local(channels),
            Residual3DBlockV2Local(channels),
        )
        self.to_delta_16 = nn.Conv3d(channels, 3, 3, padding=1)

        channels_32 = max(channels // 2, 8)
        self.up_features_32 = nn.ConvTranspose3d(
            channels, channels_32, 4, stride=2, padding=1
        )
        self.up_lut_32 = nn.ConvTranspose3d(
            3, 3, 4, stride=2, padding=1, groups=3
        )
        self.stage_32 = nn.Sequential(
            Residual3DBlockV2Local(channels_32),
            Residual3DBlockV2Local(channels_32),
            Residual3DBlockV2Local(channels_32),
        )
        self.to_delta_32 = nn.Conv3d(channels_32, 3, 3, padding=1)

        for layer in (self.to_delta_8, self.to_delta_16, self.to_delta_32):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        _init_trilinear_deconv(self.up_lut_16)
        _init_trilinear_deconv(self.up_lut_32)

    def forward(self, code: torch.Tensor):
        cond = self.condition(code)
        b = cond.shape[0]
        f8 = self.seed(cond).view(b, -1, 8, 8, 8)
        f8 = self.stage_8(f8)
        i8 = identity_lut_rgb(8, f8.device, f8.dtype).expand(b, -1, -1, -1, -1)
        d8 = torch.tanh(self.to_delta_8(f8))
        l8 = i8 + self.scale_8 * d8

        f16 = self.stage_16(self.up_features_16(f8))
        i16 = identity_lut_rgb(16, f16.device, f16.dtype).expand(b, -1, -1, -1, -1)
        b16 = i16 + self.up_lut_16(l8 - i8)
        d16 = torch.tanh(self.to_delta_16(f16))
        l16 = b16 + self.scale_16 * d16

        f32 = self.stage_32(self.up_features_32(f16))
        i32 = identity_lut_rgb(32, f32.device, f32.dtype).expand(b, -1, -1, -1, -1)
        b32 = i32 + self.up_lut_32(l16 - i16)
        d32 = torch.tanh(self.to_delta_32(f32))
        l32 = b32 + self.scale_32 * d32
        return l32, {"lut_8": l8, "lut_16": l16, "lut_32": l32}


class FactorizedStyleLUTGenerator(nn.Module):
    """
    Same progressive hidden feature trunk as the Dense Stage-1 style decoder,
    but the 3-channel LUT residual is factorized into:

      tonal field:      1 channel = Delta L in Oklab
      chromatic field:  2 channels = Delta a, Delta b in Oklab

    The two fields are independently scalable at inference:
      alpha_t controls tonal transfer
      alpha_c controls chromatic transfer

    Importantly, this DOES NOT double the expensive feature trunk.
    """
    def __init__(self, code_dim: int, cfg):
        super().__init__()
        channels = int(cfg.lut_channels)
        condition_dim = int(cfg.condition_dim)

        self.t_scale_8 = float(cfg.style_lut_residual_scale_8)
        self.t_scale_16 = float(cfg.style_lut_residual_scale_16)
        self.t_scale_32 = float(cfg.style_lut_residual_scale_32)

        # Oklab a/b spans a smaller numeric range than RGB. 0.5 is a conservative
        # initial multiplier and is exposed as a model config/attribute.
        self.chroma_scale_multiplier = float(
            getattr(cfg, "chromatic_scale_multiplier", 0.5)
        )
        self.c_scale_8 = self.chroma_scale_multiplier * float(cfg.style_lut_residual_scale_8)
        self.c_scale_16 = self.chroma_scale_multiplier * float(cfg.style_lut_residual_scale_16)
        self.c_scale_32 = self.chroma_scale_multiplier * float(cfg.style_lut_residual_scale_32)

        self.condition = nn.Sequential(
            nn.LayerNorm(int(code_dim)),
            nn.Linear(int(code_dim), condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.seed = nn.Linear(condition_dim, channels * 8 * 8 * 8)

        self.stage_8 = nn.Sequential(
            Residual3DBlockV2Local(channels),
            Residual3DBlockV2Local(channels),
        )
        self.to_tone_8 = nn.Conv3d(channels, 1, 3, padding=1)
        self.to_chroma_8 = nn.Conv3d(channels, 2, 3, padding=1)

        self.up_features_16 = nn.ConvTranspose3d(
            channels, channels, 4, stride=2, padding=1
        )
        self.up_tone_16 = nn.ConvTranspose3d(
            1, 1, 4, stride=2, padding=1, groups=1
        )
        self.up_chroma_16 = nn.ConvTranspose3d(
            2, 2, 4, stride=2, padding=1, groups=2
        )
        self.stage_16 = nn.Sequential(
            Residual3DBlockV2Local(channels),
            Residual3DBlockV2Local(channels),
        )
        self.to_tone_16 = nn.Conv3d(channels, 1, 3, padding=1)
        self.to_chroma_16 = nn.Conv3d(channels, 2, 3, padding=1)

        channels_32 = max(channels // 2, 8)
        self.up_features_32 = nn.ConvTranspose3d(
            channels, channels_32, 4, stride=2, padding=1
        )
        self.up_tone_32 = nn.ConvTranspose3d(
            1, 1, 4, stride=2, padding=1, groups=1
        )
        self.up_chroma_32 = nn.ConvTranspose3d(
            2, 2, 4, stride=2, padding=1, groups=2
        )
        self.stage_32 = nn.Sequential(
            Residual3DBlockV2Local(channels_32),
            Residual3DBlockV2Local(channels_32),
            Residual3DBlockV2Local(channels_32),
        )
        self.to_tone_32 = nn.Conv3d(channels_32, 1, 3, padding=1)
        self.to_chroma_32 = nn.Conv3d(channels_32, 2, 3, padding=1)

        for layer in (
            self.to_tone_8, self.to_chroma_8,
            self.to_tone_16, self.to_chroma_16,
            self.to_tone_32, self.to_chroma_32,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        for layer in (
            self.up_tone_16, self.up_chroma_16,
            self.up_tone_32, self.up_chroma_32,
        ):
            _init_trilinear_deconv(layer)

    def forward(self, code: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        cond = self.condition(code)
        b = cond.shape[0]

        f8 = self.seed(cond).view(b, -1, 8, 8, 8)
        f8 = self.stage_8(f8)
        t8 = self.t_scale_8 * torch.tanh(self.to_tone_8(f8))
        c8 = self.c_scale_8 * torch.tanh(self.to_chroma_8(f8))

        f16 = self.stage_16(self.up_features_16(f8))
        t16 = self.up_tone_16(t8) + self.t_scale_16 * torch.tanh(self.to_tone_16(f16))
        c16 = self.up_chroma_16(c8) + self.c_scale_16 * torch.tanh(self.to_chroma_16(f16))

        f32 = self.stage_32(self.up_features_32(f16))
        t32 = self.up_tone_32(t16) + self.t_scale_32 * torch.tanh(self.to_tone_32(f32))
        c32 = self.up_chroma_32(c16) + self.c_scale_32 * torch.tanh(self.to_chroma_32(f32))

        return {
            "tone": t32,
            "chroma": c32,
            "pyramid": {
                "tone_8": t8, "chroma_8": c8,
                "tone_16": t16, "chroma_16": c16,
                "tone_32": t32, "chroma_32": c32,
            },
        }

    @staticmethod
    def _strength_tensor(value, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(value):
            x = value.to(device=ref.device, dtype=ref.dtype)
        else:
            x = torch.tensor(float(value), device=ref.device, dtype=ref.dtype)
        if x.ndim == 0:
            return x
        # Per-sample slider [B] -> [B,1,1,1,1]
        if x.ndim == 1:
            return x.view(-1, 1, 1, 1, 1)
        return x

    def compose_lut(
        self,
        fields: dict[str, torch.Tensor | dict],
        tonal_strength=1.0,
        chromatic_strength=1.0,
    ) -> dict[str, torch.Tensor]:
        tone = fields["tone"]
        chroma = fields["chroma"]
        b = tone.shape[0]

        ident_lab = identity_oklab_lut(
            32, device=tone.device, dtype=tone.dtype
        ).expand(b, -1, -1, -1, -1)

        at = self._strength_tensor(tonal_strength, tone)
        ac = self._strength_tensor(chromatic_strength, tone)

        lab = ident_lab.clone()
        lab[:, 0:1] = ident_lab[:, 0:1] + at * tone
        lab[:, 1:3] = ident_lab[:, 1:3] + ac * chroma

        # Run color conversion in float32 for AMP stability, then cast back.
        srgb, linear = oklab_to_srgb_channel_first(lab.float())
        srgb = srgb.to(dtype=tone.dtype)
        linear = linear.to(dtype=tone.dtype)

        return {
            "lut": srgb,
            "lut_oklab": lab,
            "lut_linear_rgb": linear,
            "gamut_penalty": linear_rgb_range_penalty(linear.float()),
        }


def apply_lut_v2_local(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must be [B,3,H,W]")
    if lut.ndim != 5 or lut.shape[1] != 3:
        raise ValueError("lut must be [B,3,D,D,D]")
    grid = image.permute(0, 2, 3, 1).unsqueeze(1)
    grid = (grid - 0.5) * 2.0
    out = F.grid_sample(
        lut, grid, mode="bilinear", padding_mode="border", align_corners=True
    )
    return out.squeeze(2)
