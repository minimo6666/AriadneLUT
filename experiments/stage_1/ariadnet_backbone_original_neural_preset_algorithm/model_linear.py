from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders_v2 import NormalizationEncoderV2, StyleEncoderV2


def apply_global_matrix(
    image: torch.Tensor,
    matrix: torch.Tensor,
) -> torch.Tensor:
    """Apply one bias-free 3x3 RGB matrix to every pixel of each image."""
    if image.ndim != 4 or image.shape[1] != 3:
        raise ValueError(f"Expected image [B,3,H,W], got {tuple(image.shape)}")
    if matrix.shape != (image.shape[0], 3, 3):
        raise ValueError(
            f"Expected matrix [{image.shape[0]},3,3], got {tuple(matrix.shape)}"
        )
    return torch.bmm(matrix, image.flatten(2)).reshape_as(image)


class NeuralPresetLinearDecoder(nn.Module):
    """Neural-Preset's P @ M @ Q global color parameterization.

    Ariadne's two encoders emit compact codes instead of Neural-Preset's
    16x16 matrices directly, so each code gets one linear projection to that
    matrix space. P and Q are shared between normalization and style exactly
    as they are shared between d and r in Neural-Preset.
    """

    def __init__(
        self,
        normalization_dim: int,
        style_dim: int,
        matrix_rank: int = 16,
    ) -> None:
        super().__init__()
        self.matrix_rank = int(matrix_rank)
        coefficient_count = self.matrix_rank**2
        self.normalization_projection = nn.Linear(
            int(normalization_dim), coefficient_count
        )
        self.style_projection = nn.Linear(int(style_dim), coefficient_count)
        self.transform_p = nn.Parameter(torch.rand(3, self.matrix_rank))
        self.transform_q = nn.Parameter(torch.rand(self.matrix_rank, 3))
        nn.init.normal_(self.normalization_projection.weight, std=1e-3)
        nn.init.zeros_(self.normalization_projection.bias)
        nn.init.normal_(self.style_projection.weight, std=1e-3)
        nn.init.zeros_(self.style_projection.bias)

    def _decode(
        self,
        code: torch.Tensor,
        projection: nn.Linear,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        coefficients = projection(code).reshape(
            code.shape[0], self.matrix_rank, self.matrix_rank
        )
        matrix = self.transform_p.unsqueeze(0) @ coefficients
        matrix = matrix @ self.transform_q.unsqueeze(0)
        matrix = matrix / self.matrix_rank
        identity = torch.eye(3, dtype=matrix.dtype, device=matrix.device)
        matrix = matrix + identity.unsqueeze(0)
        return matrix, coefficients

    def normalization(
        self, code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decode(code, self.normalization_projection)

    def style(
        self, code: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self._decode(code, self.style_projection)


class AriadneLinearColorV2(nn.Module):
    """Ariadne V2 encoders with only Neural-Preset global linear color heads."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder_size = int(cfg.encoder_size)
        self.normalization_encoder = NormalizationEncoderV2(cfg)
        self.style_encoder = StyleEncoderV2(cfg)
        self.linear_decoder = NeuralPresetLinearDecoder(
            normalization_dim=int(cfg.normalization_dim),
            style_dim=int(cfg.style_dim),
            matrix_rank=int(cfg.matrix_rank),
        )

    def _resize_for_encoder(self, image: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            image,
            size=(self.encoder_size, self.encoder_size),
            mode="bilinear",
            align_corners=False,
        )

    def encode_normalization(self, image: torch.Tensor):
        return self.normalization_encoder(self._resize_for_encoder(image))

    def encode_style(self, image: torch.Tensor):
        return self.style_encoder(self._resize_for_encoder(image))

    def normalize(self, content: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, code = self.encode_normalization(content)
        matrix, coefficients = self.linear_decoder.normalization(code)
        canonical = apply_global_matrix(content, matrix)
        return {
            "normalization_tokens": tokens,
            "normalization_code": code,
            "normalization_coefficients": coefficients,
            "normalization_matrix": matrix,
            "canonical": canonical,
        }

    def extract_style(self, style: torch.Tensor) -> dict[str, torch.Tensor]:
        tokens, code = self.encode_style(style)
        matrix, coefficients = self.linear_decoder.style(code)
        return {
            "style_tokens": tokens,
            "style_code": code,
            "style_coefficients": coefficients,
            "style_matrix": matrix,
        }

    @staticmethod
    def apply_style(
        canonical: torch.Tensor,
        style_matrix: torch.Tensor,
    ) -> torch.Tensor:
        return apply_global_matrix(canonical, style_matrix)

    def decompose(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        return {**self.normalize(image), **self.extract_style(image)}

    def forward_pair(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> dict[str, dict[str, torch.Tensor] | torch.Tensor]:
        state_a = self.decompose(image_a)
        state_b = self.decompose(image_b)
        output_ab = self.apply_style(
            state_a["canonical"], state_b["style_matrix"]
        )
        output_ba = self.apply_style(
            state_b["canonical"], state_a["style_matrix"]
        )
        return {
            "state_a": state_a,
            "state_b": state_b,
            "output_ab": output_ab,
            "output_ba": output_ba,
        }

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        _forward_pair: bool = False,
    ) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
        if _forward_pair:
            return self.forward_pair(content, style)
        normalization_state = self.normalize(content)
        style_state = self.extract_style(style)
        output = self.apply_style(
            normalization_state["canonical"], style_state["style_matrix"]
        )
        return {"output": output, **normalization_state, **style_state}
