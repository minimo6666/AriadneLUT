
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from encoders_v2_local import NormalizationEncoderV2Local, StyleEncoderV2Local
from progressive_lut_tonal_chromatic import (
    Progressive3DLUTGeneratorV2Local,
    FactorizedStyleLUTGenerator,
    apply_lut_v2_local,
)


class AriadneTonalChromaticStage1(nn.Module):
    """
    Stage-1 Ariadne with an unchanged normalization branch and a factorized
    Style LUT decoder.

    C -> Norm -> Z_C
    S -> Style Encoder -> shared style code
                       -> shared progressive 3D feature trunk
                       -> Delta-L tonal field
                       -> Delta-a/b chromatic field

    Final style LUT at control strengths (alpha_T, alpha_C):
        Oklab(identity_RGB) + [alpha_T * DeltaL,
                               alpha_C * Deltaa,
                               alpha_C * Deltab]
        -> RGB 32^3 LUT
    """
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.encoder_size = int(cfg.encoder_size)

        self.normalization_encoder = NormalizationEncoderV2Local(cfg)
        self.style_encoder = StyleEncoderV2Local(cfg)

        self.normalization_lut_generator = Progressive3DLUTGeneratorV2Local(
            code_dim=int(cfg.normalization_dim),
            cfg=cfg,
            residual_scale_8=float(cfg.normalization_lut_residual_scale_8),
            residual_scale_16=float(cfg.normalization_lut_residual_scale_16),
            residual_scale_32=float(cfg.normalization_lut_residual_scale_32),
        )
        self.style_lut_generator = FactorizedStyleLUTGenerator(
            code_dim=int(cfg.style_dim), cfg=cfg
        )

    def _resize(self, x: torch.Tensor) -> torch.Tensor:
        return F.interpolate(
            x,
            size=(self.encoder_size, self.encoder_size),
            mode="bilinear",
            align_corners=False,
        )

    def encode_normalization(self, x: torch.Tensor):
        return self.normalization_encoder(self._resize(x))

    def encode_style(self, x: torch.Tensor):
        return self.style_encoder(self._resize(x))

    def normalize(self, content: torch.Tensor) -> dict:
        tokens, code = self.encode_normalization(content)
        lut, pyramid = self.normalization_lut_generator(code)
        canonical = apply_lut_v2_local(content, lut)
        return {
            "normalization_tokens": tokens,
            "normalization_code": code,
            "normalization_lut": lut,
            "normalization_lut_pyramid": pyramid,
            "canonical": canonical,
        }

    def extract_style(self, style: torch.Tensor) -> dict:
        tokens, code = self.encode_style(style)
        fields = self.style_lut_generator(code)
        return {
            "style_tokens": tokens,
            "style_code": code,
            "style_fields": fields,
        }

    def compose_style(self, style_state: dict, tonal_strength=1.0, chromatic_strength=1.0):
        return self.style_lut_generator.compose_lut(
            style_state["style_fields"],
            tonal_strength=tonal_strength,
            chromatic_strength=chromatic_strength,
        )

    @staticmethod
    def apply_style(canonical: torch.Tensor, lut: torch.Tensor) -> torch.Tensor:
        return apply_lut_v2_local(canonical, lut)

    def decompose(self, image: torch.Tensor) -> dict:
        return {**self.normalize(image), **self.extract_style(image)}

    def _apply_views(
        self,
        canonical: torch.Tensor,
        style_state: dict,
        controlled_tonal_strength=None,
        controlled_chromatic_strength=None,
    ) -> dict:
        full = self.compose_style(style_state, 1.0, 1.0)
        tonal = self.compose_style(style_state, 1.0, 0.0)
        chroma = self.compose_style(style_state, 0.0, 1.0)
        auxiliary_canonical = canonical.detach()
        result = {
            "full": self.apply_style(canonical, full["lut"]),
            "tonal_only": self.apply_style(auxiliary_canonical, tonal["lut"]),
            "chroma_only": self.apply_style(auxiliary_canonical, chroma["lut"]),
            "full_lut_state": full,
            "tonal_lut_state": tonal,
            "chroma_lut_state": chroma,
        }
        if controlled_tonal_strength is not None and controlled_chromatic_strength is not None:
            controlled = self.compose_style(
                style_state,
                controlled_tonal_strength,
                controlled_chromatic_strength,
            )
            result["controlled"] = self.apply_style(
                auxiliary_canonical, controlled["lut"]
            )
            result["controlled_lut_state"] = controlled
        return result

    def forward_pair(
        self,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
        controlled_tonal_strength=None,
        controlled_chromatic_strength=None,
    ) -> dict:
        state_a = self.decompose(image_a)
        state_b = self.decompose(image_b)

        ab = self._apply_views(
            state_a["canonical"],
            state_b,
            controlled_tonal_strength,
            controlled_chromatic_strength,
        )
        ba = self._apply_views(
            state_b["canonical"],
            state_a,
            controlled_tonal_strength,
            controlled_chromatic_strength,
        )

        gamut_terms = [
            ab["full_lut_state"]["gamut_penalty"],
            ba["full_lut_state"]["gamut_penalty"],
            0.5 * ab["tonal_lut_state"]["gamut_penalty"],
            0.5 * ba["tonal_lut_state"]["gamut_penalty"],
            0.5 * ab["chroma_lut_state"]["gamut_penalty"],
            0.5 * ba["chroma_lut_state"]["gamut_penalty"],
        ]
        denom = 4.0
        if "controlled_lut_state" in ab:
            gamut_terms += [
                0.5 * ab["controlled_lut_state"]["gamut_penalty"],
                0.5 * ba["controlled_lut_state"]["gamut_penalty"],
            ]
            denom += 1.0

        result = {
            "state_a": state_a,
            "state_b": state_b,
            "output_ab": ab["full"],
            "output_ba": ba["full"],
            "output_ab_tonal_only": ab["tonal_only"],
            "output_ba_tonal_only": ba["tonal_only"],
            "output_ab_chroma_only": ab["chroma_only"],
            "output_ba_chroma_only": ba["chroma_only"],
            "gamut_penalty": sum(gamut_terms) / denom,
        }
        if "controlled" in ab:
            result.update({
                "output_ab_controlled": ab["controlled"],
                "output_ba_controlled": ba["controlled"],
                "controlled_tonal_strength": controlled_tonal_strength,
                "controlled_chromatic_strength": controlled_chromatic_strength,
            })
        return result

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        tonal_strength=1.0,
        chromatic_strength=1.0,
        *,
        _forward_pair: bool = False,
        controlled_tonal_strength=None,
        controlled_chromatic_strength=None,
    ) -> dict:
        if _forward_pair:
            return self.forward_pair(
                content,
                style,
                controlled_tonal_strength=controlled_tonal_strength,
                controlled_chromatic_strength=controlled_chromatic_strength,
            )
        n = self.normalize(content)
        s = self.extract_style(style)
        composed = self.compose_style(
            s,
            tonal_strength=tonal_strength,
            chromatic_strength=chromatic_strength,
        )
        output = self.apply_style(n["canonical"], composed["lut"])
        return {
            "output": output,
            **n,
            **s,
            "style_lut": composed["lut"],
            "style_lut_oklab": composed["lut_oklab"],
            "style_gamut_penalty": composed["gamut_penalty"],
            "tonal_strength": tonal_strength,
            "chromatic_strength": chromatic_strength,
        }
