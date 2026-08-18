from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from query_field import QueryPyramid


def identity_lut_v2(
    dimension: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """Batch-independent identity LUT [1,3,D,D,D], x->R, y->G, z->B."""
    values = torch.linspace(0.0, 1.0, int(dimension), device=device, dtype=dtype)
    blue, green, red = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack([red, green, blue], dim=0).unsqueeze(0)


class QueryGuidedResidual3DBlockV2(nn.Module):
    """Original Residual3DBlockV2 plus one explicit query-conditioning signal.

    Baseline:
        y = x + R(x)

    This controlled experiment:
        y = x + (1 + lambda * A) * R(x + E(A))

    A is a soft RGB-space query focus map in [0,1]. The original Conv3D stack,
    normalization, activation, channel count, and zero-residual initialization are
    preserved. No site is pruned; the baseline dense representation remains intact.
    """

    def __init__(self, channels: int, focus_strength: float = 0.25):
        super().__init__()
        channels = int(channels)
        groups = 8 if channels % 8 == 0 else 1
        self.focus_strength = float(focus_strength)

        # Construct the original residual operator first, in exactly the original order.
        # This preserves the baseline RNG stream for all pre-existing parameters.
        self.net = nn.Sequential(
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, channels),
            nn.GELU(),
            nn.Conv3d(channels, channels, kernel_size=3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

        # A manual zero-initialized 1x1x1 Conv3D avoids consuming random numbers,
        # so with the same experiment seed every original baseline parameter starts
        # from the same initialization it would have had without Query conditioning.
        self.query_weight = nn.Parameter(torch.zeros(channels, 1, 1, 1, 1))
        self.query_bias = nn.Parameter(torch.zeros(channels))

    def forward(self, features: torch.Tensor, focus: torch.Tensor) -> torch.Tensor:
        if focus.ndim != 5 or focus.shape[1] != 1:
            raise ValueError("focus must have shape [B,1,D,H,W]")
        if tuple(focus.shape[-3:]) != tuple(features.shape[-3:]):
            raise ValueError(
                f"focus spatial shape {tuple(focus.shape[-3:])} does not match "
                f"features {tuple(features.shape[-3:])}"
            )
        focus = focus.to(device=features.device, dtype=features.dtype)
        query_embedding = F.conv3d(focus, self.query_weight, self.query_bias)
        conditioned = features + query_embedding
        residual = self.net(conditioned)
        gate = 1.0 + self.focus_strength * focus
        return features + gate * residual


class QueryGuidedProgressive3DLUTGeneratorV2(nn.Module):
    """Original Dense 8->16->32 progressive LUT decoder with Query-aware Conv3D blocks.

    Nothing is sparsified or removed. The sole architectural change is that every
    residual 3-D block receives a soft query field aligned to its existing resolution.
    """

    def __init__(
        self,
        code_dim: int,
        cfg,
        residual_scale_8: float,
        residual_scale_16: float,
        residual_scale_32: float,
        focus_strength: float = 0.25,
    ):
        super().__init__()
        channels = int(cfg.lut_channels)
        condition_dim = int(cfg.condition_dim)
        self.scale_8 = float(residual_scale_8)
        self.scale_16 = float(residual_scale_16)
        self.scale_32 = float(residual_scale_32)

        # Identical baseline condition/seed path.
        self.condition = nn.Sequential(
            nn.LayerNorm(int(code_dim)),
            nn.Linear(int(code_dim), condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.seed = nn.Linear(condition_dim, channels * 8 * 8 * 8)

        self.stage_8 = nn.ModuleList(
            [
                QueryGuidedResidual3DBlockV2(channels, focus_strength),
                QueryGuidedResidual3DBlockV2(channels, focus_strength),
            ]
        )
        self.to_delta_8 = nn.Conv3d(channels, 3, kernel_size=3, padding=1)

        self.up_features_16 = nn.ConvTranspose3d(
            channels, channels, kernel_size=4, stride=2, padding=1
        )
        self.up_lut_16 = nn.ConvTranspose3d(
            3, 3, kernel_size=4, stride=2, padding=1, groups=3
        )
        self.stage_16 = nn.ModuleList(
            [
                QueryGuidedResidual3DBlockV2(channels, focus_strength),
                QueryGuidedResidual3DBlockV2(channels, focus_strength),
            ]
        )
        self.to_delta_16 = nn.Conv3d(channels, 3, kernel_size=3, padding=1)

        channels_32 = max(channels // 2, 8)
        self.up_features_32 = nn.ConvTranspose3d(
            channels, channels_32, kernel_size=4, stride=2, padding=1
        )
        self.up_lut_32 = nn.ConvTranspose3d(
            3, 3, kernel_size=4, stride=2, padding=1, groups=3
        )
        self.stage_32 = nn.ModuleList(
            [
                QueryGuidedResidual3DBlockV2(channels_32, focus_strength),
                QueryGuidedResidual3DBlockV2(channels_32, focus_strength),
                QueryGuidedResidual3DBlockV2(channels_32, focus_strength),
            ]
        )
        self.to_delta_32 = nn.Conv3d(channels_32, 3, kernel_size=3, padding=1)

        # Preserve the original near-identity LUT initialization exactly.
        for layer in (self.to_delta_8, self.to_delta_16, self.to_delta_32):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self._initialize_trainable_lut_upsampler(self.up_lut_16)
        self._initialize_trainable_lut_upsampler(self.up_lut_32)

    @staticmethod
    def _initialize_trainable_lut_upsampler(layer: nn.ConvTranspose3d) -> None:
        factor = 2
        size = 4
        center = (size - 1) / 2
        coordinates = torch.arange(size, dtype=torch.float32)
        one_dimensional = (
            1.0 - torch.abs(coordinates - center) / factor
        ).clamp_min(0.0)
        kernel = (
            one_dimensional[:, None, None]
            * one_dimensional[None, :, None]
            * one_dimensional[None, None, :]
        )
        with torch.no_grad():
            layer.weight.zero_()
            for channel in range(3):
                layer.weight[channel, 0].copy_(kernel)
            if layer.bias is not None:
                layer.bias.zero_()

    @staticmethod
    def _run_stage(
        features: torch.Tensor,
        focus: torch.Tensor,
        blocks: nn.ModuleList,
    ) -> torch.Tensor:
        for block in blocks:
            features = block(features, focus)
        return features

    def forward(self, code: torch.Tensor, query: QueryPyramid):
        if query is None:
            raise ValueError("QueryGuidedProgressive3DLUTGeneratorV2 requires a QueryPyramid")

        condition = self.condition(code)
        batch_size = condition.shape[0]

        features_8 = self.seed(condition).view(batch_size, -1, 8, 8, 8)
        features_8 = self._run_stage(features_8, query.a8, self.stage_8)
        identity_8 = identity_lut_v2(
            8, device=features_8.device, dtype=features_8.dtype
        ).expand(batch_size, -1, -1, -1, -1)
        delta_8 = torch.tanh(self.to_delta_8(features_8))
        lut_8 = identity_8 + self.scale_8 * delta_8

        features_16 = self.up_features_16(features_8)
        features_16 = self._run_stage(features_16, query.a16, self.stage_16)
        identity_16 = identity_lut_v2(
            16, device=features_16.device, dtype=features_16.dtype
        ).expand(batch_size, -1, -1, -1, -1)
        base_16 = identity_16 + self.up_lut_16(lut_8 - identity_8)
        delta_16 = torch.tanh(self.to_delta_16(features_16))
        lut_16 = base_16 + self.scale_16 * delta_16

        features_32 = self.up_features_32(features_16)
        features_32 = self._run_stage(features_32, query.a32, self.stage_32)
        identity_32 = identity_lut_v2(
            32, device=features_32.device, dtype=features_32.dtype
        ).expand(batch_size, -1, -1, -1, -1)
        base_32 = identity_32 + self.up_lut_32(lut_16 - identity_16)
        delta_32 = torch.tanh(self.to_delta_32(features_32))
        lut_32 = base_32 + self.scale_32 * delta_32

        pyramid = {
            "lut_8": lut_8,
            "lut_16": lut_16,
            "lut_32": lut_32,
            # Detached query fields are returned only for diagnostics/visualization.
            "query_mass_8": query.q8,
            "query_mass_16": query.q16,
            "query_mass_32": query.q32,
            "query_focus_8": query.a8,
            "query_focus_16": query.a16,
            "query_focus_32": query.a32,
        }
        return lut_32, pyramid


def apply_lut_v2(image: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
    """Exact local copy of the original Ariadne V2 LUT application."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError("image must have shape [B,3,H,W]")
    if lut.ndim != 5 or lut.shape[1] != 3:
        raise ValueError("lut must have shape [B,3,D,D,D]")
    if image.shape[0] != lut.shape[0]:
        raise ValueError("image and lut must have the same batch size")

    grid = image.permute(0, 2, 3, 1).unsqueeze(1)
    grid = (grid - 0.5) * 2.0
    output = F.grid_sample(
        lut,
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return output.squeeze(2)
