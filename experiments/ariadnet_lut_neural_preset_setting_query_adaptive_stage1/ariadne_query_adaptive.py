from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.encoders_v2 import NormalizationEncoderV2, StyleEncoderV2
from models.progressive_lut_v2 import apply_lut_v2

from query_adaptive_lut import QueryAdaptiveHierarchicalLUTGenerator


class AriadneLUTQueryAdaptiveStage1(nn.Module):
    """Unified query-adaptive Stage-1 Ariadne.

    Both transforms follow the same principle:
      dense global 8^3 mapping + query-local 16^3/32^3 displacement refinement.

    Normalization:
      image C -> Q_C decides where the normalization LUT receives fine capacity.

    Styling:
      canonical Z_C -> Q_ZC decides where the style LUT receives fine capacity;
      canonicalized style reference provides an evidence field Q_ZS, while the
      8^3 coarse style prior remains generated from the style code alone.
    """

    def __init__(self, base_model_cfg, qa_model_cfg: dict):
        super().__init__()
        self.cfg = base_model_cfg
        self.qa_cfg = qa_model_cfg
        self.encoder_size = int(base_model_cfg.encoder_size)

        self.normalization_encoder = NormalizationEncoderV2(base_model_cfg)
        self.style_encoder = StyleEncoderV2(base_model_cfg)

        norm_cfg = dict(qa_model_cfg.get("normalization", {}))
        style_cfg = dict(qa_model_cfg.get("style", {}))
        common = dict(qa_model_cfg.get("common", {}))
        norm_cfg = {**common, **norm_cfg}
        style_cfg = {**common, **style_cfg}

        self.use_style_evidence = bool(style_cfg.get("use_style_evidence", True))
        self.normalization_lut_generator = QueryAdaptiveHierarchicalLUTGenerator(
            code_dim=int(base_model_cfg.normalization_dim),
            base_cfg=base_model_cfg,
            qa_cfg=norm_cfg,
            residual_scale_8=float(base_model_cfg.normalization_lut_residual_scale_8),
            residual_scale_16=float(base_model_cfg.normalization_lut_residual_scale_16),
            residual_scale_32=float(base_model_cfg.normalization_lut_residual_scale_32),
            use_evidence=False,
        )
        self.style_lut_generator = QueryAdaptiveHierarchicalLUTGenerator(
            code_dim=int(base_model_cfg.style_dim),
            base_cfg=base_model_cfg,
            qa_cfg=style_cfg,
            residual_scale_8=float(base_model_cfg.style_lut_residual_scale_8),
            residual_scale_16=float(base_model_cfg.style_lut_residual_scale_16),
            residual_scale_32=float(base_model_cfg.style_lut_residual_scale_32),
            use_evidence=self.use_style_evidence,
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

    def normalize(self, content: torch.Tensor) -> dict:
        tokens, code = self.encode_normalization(content)
        lut, pyramid = self.normalization_lut_generator(
            code,
            query_image=content,
            evidence_image=None,
        )
        canonical = apply_lut_v2(content, lut)
        return {
            "normalization_tokens": tokens,
            "normalization_code": code,
            "normalization_lut": lut,
            "normalization_lut_pyramid": pyramid,
            "canonical": canonical,
        }

    def _style_from_code(
        self,
        *,
        style_tokens: torch.Tensor,
        style_code: torch.Tensor,
        application_query: torch.Tensor,
        style_evidence: torch.Tensor | None,
    ) -> dict:
        lut, pyramid = self.style_lut_generator(
            style_code,
            query_image=application_query,
            evidence_image=style_evidence if self.use_style_evidence else None,
        )
        return {
            "style_tokens": style_tokens,
            "style_code": style_code,
            "style_lut": lut,
            "style_lut_pyramid": pyramid,
        }

    @staticmethod
    def apply_style(canonical: torch.Tensor, style_lut: torch.Tensor) -> torch.Tensor:
        return apply_lut_v2(canonical, style_lut)

    def forward_pair(self, image_a: torch.Tensor, image_b: torch.Tensor) -> dict:
        # Normalize first. This gives the actual images on which the opposite
        # style LUT will be queried.
        norm_a = self.normalize(image_a)
        norm_b = self.normalize(image_b)

        style_tokens_a, style_code_a = self.encode_style(image_a)
        style_tokens_b, style_code_b = self.encode_style(image_b)

        # Style A is applied to canonical B; Style B is applied to canonical A.
        # The reference's own canonical query field is supplied only as evidence
        # and never chooses the active application region.
        style_a = self._style_from_code(
            style_tokens=style_tokens_a,
            style_code=style_code_a,
            application_query=norm_b["canonical"],
            style_evidence=norm_a["canonical"],
        )
        style_b = self._style_from_code(
            style_tokens=style_tokens_b,
            style_code=style_code_b,
            application_query=norm_a["canonical"],
            style_evidence=norm_b["canonical"],
        )

        state_a = {**norm_a, **style_a}
        state_b = {**norm_b, **style_b}
        output_ab = self.apply_style(norm_a["canonical"], style_b["style_lut"])
        output_ba = self.apply_style(norm_b["canonical"], style_a["style_lut"])
        return {
            "state_a": state_a,
            "state_b": state_b,
            "output_ab": output_ab,
            "output_ba": output_ba,
        }

    def forward(self, content: torch.Tensor, style: torch.Tensor) -> dict:
        norm_c = self.normalize(content)
        # The style reference is canonicalized only to build Q_ZS evidence. Its
        # normalization LUT is also query-adaptive and shares the same Stage-1
        # architecture. This extra branch can later be cached per style image.
        norm_s = self.normalize(style) if self.use_style_evidence else None
        style_tokens, style_code = self.encode_style(style)
        style_state = self._style_from_code(
            style_tokens=style_tokens,
            style_code=style_code,
            application_query=norm_c["canonical"],
            style_evidence=(norm_s["canonical"] if norm_s is not None else None),
        )
        output = self.apply_style(norm_c["canonical"], style_state["style_lut"])
        result = {"output": output, **norm_c, **style_state}
        if norm_s is not None:
            result["style_canonical"] = norm_s["canonical"]
            result["style_normalization_state"] = norm_s
        return result
