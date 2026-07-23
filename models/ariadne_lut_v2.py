from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoders_v2 import (
    NormalizationEncoderV2,
    StyleEncoderV2,
)
from .progressive_lut_v2 import (
    Progressive3DLUTGeneratorV2,
    apply_lut_v2,
)


class AriadneLUTV2(nn.Module):
    """
    Explicit canonicalization model with two independent LUT branches.

    Normalization branch:
        C -> d_C -> L_N^C -> Z_C

    Style branch:
        S -> r_S -> L_S^S

    Final transfer:
        O^{C->S} = A(Z_C, L_S^S)

    The Style LUT generator never receives the content image, content code,
    canonical image, or normalization LUT. Therefore the generated Style LUT
    is structurally content-independent and can be reused on any image that
    has first been mapped into the learned canonical color space.
    """

    def __init__(self, cfg):
        super().__init__()

        self.cfg = cfg
        self.encoder_size = int(cfg.encoder_size)

        self.normalization_encoder = (
            NormalizationEncoderV2(cfg)
        )
        self.style_encoder = StyleEncoderV2(cfg)

        self.normalization_lut_generator = (
            Progressive3DLUTGeneratorV2(
                code_dim=int(cfg.normalization_dim),
                cfg=cfg,
                residual_scale_8=float(
                    cfg.normalization_lut_residual_scale_8
                ),
                residual_scale_16=float(
                    cfg.normalization_lut_residual_scale_16
                ),
                residual_scale_32=float(
                    cfg.normalization_lut_residual_scale_32
                ),
            )
        )

        self.style_lut_generator = (
            Progressive3DLUTGeneratorV2(
                code_dim=int(cfg.style_dim),
                cfg=cfg,
                residual_scale_8=float(
                    cfg.style_lut_residual_scale_8
                ),
                residual_scale_16=float(
                    cfg.style_lut_residual_scale_16
                ),
                residual_scale_32=float(
                    cfg.style_lut_residual_scale_32
                ),
            )
        )

    def _resize_for_encoder(
        self,
        image: torch.Tensor,
    ) -> torch.Tensor:
        return F.interpolate(
            image,
            size=(
                self.encoder_size,
                self.encoder_size,
            ),
            mode="bilinear",
            align_corners=False,
        )

    def encode_normalization(
        self,
        image: torch.Tensor,
    ):
        resized = self._resize_for_encoder(image)
        return self.normalization_encoder(resized)

    def encode_style(
        self,
        image: torch.Tensor,
    ):
        resized = self._resize_for_encoder(image)
        return self.style_encoder(resized)

    def normalize(
        self,
        content: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Map one raw content image into the learned canonical color space.
        """
        normalization_tokens, normalization_code = (
            self.encode_normalization(content)
        )

        normalization_lut, normalization_lut_pyramid = (
            self.normalization_lut_generator(
                normalization_code
            )
        )

        canonical = apply_lut_v2(
            content,
            normalization_lut,
        )

        return {
            "normalization_tokens": normalization_tokens,
            "normalization_code": normalization_code,
            "normalization_lut": normalization_lut,
            "normalization_lut_pyramid": (
                normalization_lut_pyramid
            ),
            "canonical": canonical,
        }

    def extract_style(
        self,
        style: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Generate a reusable Style LUT from the style image alone.
        """
        style_tokens, style_code = self.encode_style(style)

        style_lut, style_lut_pyramid = (
            self.style_lut_generator(style_code)
        )

        return {
            "style_tokens": style_tokens,
            "style_code": style_code,
            "style_lut": style_lut,
            "style_lut_pyramid": style_lut_pyramid,
        }

    @staticmethod
    def apply_style(
        canonical: torch.Tensor,
        style_lut: torch.Tensor,
    ) -> torch.Tensor:
        return apply_lut_v2(
            canonical,
            style_lut,
        )

    def decompose(
        self,
        image: torch.Tensor,
    ) -> dict[str, torch.Tensor | dict]:
        """
        Compute both independent descriptions of one image.

        This method is used during same-content Stage-1 training so each image
        is encoded only once per branch.
        """
        normalization_state = self.normalize(image)
        style_state = self.extract_style(image)

        return {
            **normalization_state,
            **style_state,
        }

    def forward_pair(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> dict[str, dict | torch.Tensor]:
        """
        Bidirectional Stage-1 reconstruction for two grades of one content.

        A -> canonical_A -> Style LUT B -> output_AB
        B -> canonical_B -> Style LUT A -> output_BA
        """
        state_a = self.decompose(image_a)
        state_b = self.decompose(image_b)

        output_ab = self.apply_style(
            state_a["canonical"],
            state_b["style_lut"],
        )
        output_ba = self.apply_style(
            state_b["canonical"],
            state_a["style_lut"],
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
    ) -> dict[str, torch.Tensor | dict]:
        """
        Transfer the color style of S to arbitrary content C.

        The content image and style image may contain entirely different
        scenes. The two branches remain independent until the Style LUT is
        applied to the canonicalized content.
        """
        normalization_state = self.normalize(content)
        style_state = self.extract_style(style)

        output = self.apply_style(
            normalization_state["canonical"],
            style_state["style_lut"],
        )

        return {
            "output": output,
            **normalization_state,
            **style_state,
        }
