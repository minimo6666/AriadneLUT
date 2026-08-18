from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders_v2_local import NormalizationEncoderV2, StyleEncoderV2
from progressive_lut_query_guided import (
    QueryGuidedProgressive3DLUTGeneratorV2,
    apply_lut_v2,
)
from query_field import QueryPyramid, build_query_pyramid


class AriadneLUTQueryGuidedDense(nn.Module):
    """Dense Ariadne V2 with explicit LUT-query conditioning at 8/16/32.

    The baseline encoder, condition/seed path, progressive 3-D feature hierarchy,
    LUT upsampling, reconstruction objective, and dataset are preserved. Query maps
    enter only the existing 3-D residual blocks.
    """

    def __init__(self, cfg, query_cfg: dict):
        super().__init__()
        self.cfg = cfg
        self.query_cfg = dict(query_cfg)
        self.encoder_size = int(cfg.encoder_size)
        self.max_query_pixels = int(self.query_cfg.get("max_query_pixels", 0))
        self.focus_transform = str(self.query_cfg.get("focus_transform", "log1p"))
        self.focus_strength = float(self.query_cfg.get("focus_strength", 0.25))

        self.normalization_encoder = NormalizationEncoderV2(cfg)
        self.style_encoder = StyleEncoderV2(cfg)

        self.normalization_lut_generator = QueryGuidedProgressive3DLUTGeneratorV2(
            code_dim=int(cfg.normalization_dim),
            cfg=cfg,
            residual_scale_8=float(cfg.normalization_lut_residual_scale_8),
            residual_scale_16=float(cfg.normalization_lut_residual_scale_16),
            residual_scale_32=float(cfg.normalization_lut_residual_scale_32),
            focus_strength=self.focus_strength,
        )
        self.style_lut_generator = QueryGuidedProgressive3DLUTGeneratorV2(
            code_dim=int(cfg.style_dim),
            cfg=cfg,
            residual_scale_8=float(cfg.style_lut_residual_scale_8),
            residual_scale_16=float(cfg.style_lut_residual_scale_16),
            residual_scale_32=float(cfg.style_lut_residual_scale_32),
            focus_strength=self.focus_strength,
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

    def build_query(self, image: torch.Tensor) -> QueryPyramid:
        # Conditioning only. No new gradient route is introduced through Q.
        return build_query_pyramid(
            image.detach(),
            max_query_pixels=self.max_query_pixels,
            focus_transform=self.focus_transform,
        )

    def normalize(self, content: torch.Tensor) -> dict[str, torch.Tensor | dict]:
        normalization_tokens, normalization_code = self.encode_normalization(content)
        query = self.build_query(content)
        normalization_lut, normalization_lut_pyramid = self.normalization_lut_generator(
            normalization_code, query
        )
        canonical = apply_lut_v2(content, normalization_lut)
        return {
            "normalization_tokens": normalization_tokens,
            "normalization_code": normalization_code,
            "normalization_lut": normalization_lut,
            "normalization_lut_pyramid": normalization_lut_pyramid,
            "normalization_query": query,
            "canonical": canonical,
        }

    def _style_state_from_code(
        self,
        style_tokens: torch.Tensor,
        style_code: torch.Tensor,
        query_image: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        query = self.build_query(query_image)
        style_lut, style_lut_pyramid = self.style_lut_generator(style_code, query)
        return {
            "style_tokens": style_tokens,
            "style_code": style_code,
            "style_lut": style_lut,
            "style_lut_pyramid": style_lut_pyramid,
            "style_query": query,
        }

    def extract_style(
        self,
        style: torch.Tensor,
        query_image: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """Generate a style-defined, query-adapted LUT.

        `style` determines WHAT transform to generate; `query_image` tells the 3-D
        residual blocks WHERE the LUT will actually be queried.
        """
        style_tokens, style_code = self.encode_style(style)
        return self._style_state_from_code(style_tokens, style_code, query_image)

    @staticmethod
    def apply_style(canonical: torch.Tensor, style_lut: torch.Tensor) -> torch.Tensor:
        return apply_lut_v2(canonical, style_lut)

    def forward_pair(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> dict[str, dict | torch.Tensor]:
        """Stage-1 bidirectional reconstruction with the correct query source.

        Norm A is guided by Q(A); Norm B by Q(B).
        Style B is applied to canonical A, so its generator is guided by Q(Z_A).
        Style A is applied to canonical B, so its generator is guided by Q(Z_B).
        """
        norm_a = self.normalize(image_a)
        norm_b = self.normalize(image_b)

        style_tokens_a, style_code_a = self.encode_style(image_a)
        style_tokens_b, style_code_b = self.encode_style(image_b)

        style_b_for_a = self._style_state_from_code(
            style_tokens_b, style_code_b, norm_a["canonical"].detach()
        )
        style_a_for_b = self._style_state_from_code(
            style_tokens_a, style_code_a, norm_b["canonical"].detach()
        )

        output_ab = self.apply_style(norm_a["canonical"], style_b_for_a["style_lut"])
        output_ba = self.apply_style(norm_b["canonical"], style_a_for_b["style_lut"])

        # Preserve the original state_a/state_b keys expected by the Stage-1 loss.
        # state_a's style fields describe Style A adapted to the image it is actually
        # applied to in B->A; state_b analogously describes Style B for A->B.
        state_a = {**norm_a, **style_a_for_b}
        state_b = {**norm_b, **style_b_for_a}
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
    ) -> dict[str, torch.Tensor | dict]:
        # DDP must enter through Module.forward so reducer hooks cover the full pair loss.
        if _forward_pair:
            return self.forward_pair(content, style)
        normalization_state = self.normalize(content)
        style_state = self.extract_style(
            style,
            query_image=normalization_state["canonical"].detach(),
        )
        output = self.apply_style(
            normalization_state["canonical"],
            style_state["style_lut"],
        )
        return {
            "output": output,
            **normalization_state,
            **style_state,
        }
