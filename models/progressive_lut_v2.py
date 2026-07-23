from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def identity_lut_v2(
    dimension: int,
    device=None,
    dtype=None,
) -> torch.Tensor:
    """
    Return a batch-independent identity 3D LUT with shape:

        [1, 3, dimension, dimension, dimension]

    The LUT coordinate order follows PyTorch grid_sample:
    x -> R, y -> G, z -> B.
    """
    values = torch.linspace(
        0.0,
        1.0,
        int(dimension),
        device=device,
        dtype=dtype,
    )
    blue, green, red = torch.meshgrid(
        values,
        values,
        values,
        indexing="ij",
    )
    return torch.stack(
        [red, green, blue],
        dim=0,
    ).unsqueeze(0)


class Residual3DBlockV2(nn.Module):
    def __init__(self, channels: int):
        super().__init__()

        groups = 8 if int(channels) % 8 == 0 else 1

        self.net = nn.Sequential(
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(
                int(channels),
                int(channels),
                kernel_size=3,
                padding=1,
            ),
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(
                int(channels),
                int(channels),
                kernel_size=3,
                padding=1,
            ),
        )

        # Each residual block starts as an identity mapping.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return features + self.net(features)


class Progressive3DLUTGeneratorV2(nn.Module):
    """
    Basis-free progressive 3D LUT generator.

    A single branch-specific code is projected into a learned 8^3 feature
    volume. The LUT is then progressively refined:

        identity + residual at 8^3
        learned refinement to 16^3
        learned refinement to 32^3

    The two V2 branches instantiate independent copies of this generator:
    one for normalization LUTs and one for reusable style LUTs.
    """

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

        self.seed = nn.Linear(
            condition_dim,
            channels * 8 * 8 * 8,
        )

        self.stage_8 = nn.Sequential(
            Residual3DBlockV2(channels),
            Residual3DBlockV2(channels),
        )
        self.to_delta_8 = nn.Conv3d(
            channels,
            3,
            kernel_size=3,
            padding=1,
        )

        self.up_features_16 = nn.ConvTranspose3d(
            channels,
            channels,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_lut_16 = nn.ConvTranspose3d(
            3,
            3,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=3,
        )
        self.stage_16 = nn.Sequential(
            Residual3DBlockV2(channels),
            Residual3DBlockV2(channels),
        )
        self.to_delta_16 = nn.Conv3d(
            channels,
            3,
            kernel_size=3,
            padding=1,
        )

        channels_32 = max(channels // 2, 8)
        self.up_features_32 = nn.ConvTranspose3d(
            channels,
            channels_32,
            kernel_size=4,
            stride=2,
            padding=1,
        )
        self.up_lut_32 = nn.ConvTranspose3d(
            3,
            3,
            kernel_size=4,
            stride=2,
            padding=1,
            groups=3,
        )
        self.stage_32 = nn.Sequential(
            Residual3DBlockV2(channels_32),
            Residual3DBlockV2(channels_32),
            Residual3DBlockV2(channels_32),
        )
        self.to_delta_32 = nn.Conv3d(
            channels_32,
            3,
            kernel_size=3,
            padding=1,
        )

        for layer in (
            self.to_delta_8,
            self.to_delta_16,
            self.to_delta_32,
        ):
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

        self._initialize_trainable_lut_upsampler(
            self.up_lut_16
        )
        self._initialize_trainable_lut_upsampler(
            self.up_lut_32
        )

    @staticmethod
    def _initialize_trainable_lut_upsampler(
        layer: nn.ConvTranspose3d,
    ) -> None:
        """
        Initialize the learned LUT refinement with a trilinear-like kernel.

        The layer remains fully trainable; this is not a fixed interpolation
        operation.
        """
        factor = 2
        size = 4
        center = (size - 1) / 2

        coordinates = torch.arange(
            size,
            dtype=torch.float32,
        )
        one_dimensional = (
            1.0
            - torch.abs(coordinates - center) / factor
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

    def forward(self, code: torch.Tensor):
        condition = self.condition(code)
        batch_size = condition.shape[0]

        features_8 = self.seed(condition).view(
            batch_size,
            -1,
            8,
            8,
            8,
        )
        features_8 = self.stage_8(features_8)

        identity_8 = identity_lut_v2(
            8,
            device=features_8.device,
            dtype=features_8.dtype,
        ).expand(
            batch_size,
            -1,
            -1,
            -1,
            -1,
        )

        delta_8 = torch.tanh(
            self.to_delta_8(features_8)
        )
        lut_8 = identity_8 + self.scale_8 * delta_8

        features_16 = self.up_features_16(features_8)
        features_16 = self.stage_16(features_16)

        identity_16 = identity_lut_v2(
            16,
            device=features_16.device,
            dtype=features_16.dtype,
        ).expand(
            batch_size,
            -1,
            -1,
            -1,
            -1,
        )

        base_16 = (
            identity_16
            + self.up_lut_16(lut_8 - identity_8)
        )
        delta_16 = torch.tanh(
            self.to_delta_16(features_16)
        )
        lut_16 = base_16 + self.scale_16 * delta_16

        features_32 = self.up_features_32(features_16)
        features_32 = self.stage_32(features_32)

        identity_32 = identity_lut_v2(
            32,
            device=features_32.device,
            dtype=features_32.dtype,
        ).expand(
            batch_size,
            -1,
            -1,
            -1,
            -1,
        )

        base_32 = (
            identity_32
            + self.up_lut_32(lut_16 - identity_16)
        )
        delta_32 = torch.tanh(
            self.to_delta_32(features_32)
        )
        lut_32 = base_32 + self.scale_32 * delta_32

        pyramid = {
            "lut_8": lut_8,
            "lut_16": lut_16,
            "lut_32": lut_32,
        }
        return lut_32, pyramid


def apply_lut_v2(
    image: torch.Tensor,
    lut: torch.Tensor,
) -> torch.Tensor:
    """
    Apply one 3D LUT to each image in a batch.

    image: [B, 3, H, W], RGB values normally in [0, 1]
    lut:   [B, 3, D, D, D]
    """
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(
            "image must have shape [B, 3, H, W]"
        )

    if lut.ndim != 5 or lut.shape[1] != 3:
        raise ValueError(
            "lut must have shape [B, 3, D, D, D]"
        )

    if image.shape[0] != lut.shape[0]:
        raise ValueError(
            "image and lut must have the same batch size"
        )

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
