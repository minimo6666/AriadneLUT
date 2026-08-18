from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from oklab_torch import srgb_to_oklab_channel_first
from perceptual_loss import VGG16PerceptualLoss


class TonalChromaticStage1Objective(nn.Module):
    """Stage-1 reconstruction + explicit tonal/chromatic disentanglement.

    Full output:
      - original bidirectional RGB L1 reconstruction
      - optional VGG16 perceptual reconstruction

    Tonal-only output:
      - match target Oklab L
      - preserve canonical a/b

    Chromatic-only output:
      - match target Oklab a/b
      - preserve canonical L

    Random controlled output:
      - sampled alpha_T and alpha_C are explicitly supervised against the
        corresponding Oklab interpolation between canonical and target. This is
        what trains the *interior* of the two slider axes rather than only the
        four endpoint behaviours.

    Style-field smoothness:
      - 3-D first-order TV on Delta-L and Delta-a/b fields at 8/16/32.
    """

    def __init__(
        self,
        lambda_consistency: float = 10.0,
        lambda_tonal: float = 0.5,
        lambda_chromatic: float = 1.0,
        lambda_leakage: float = 0.5,
        lambda_slider: float = 0.5,
        lambda_perceptual: float = 0.03,
        lambda_lut_smooth: float = 0.02,
        lambda_gamut: float = 0.02,
        perceptual_size: int = 128,
    ):
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_tonal = float(lambda_tonal)
        self.lambda_chromatic = float(lambda_chromatic)
        self.lambda_leakage = float(lambda_leakage)
        self.lambda_slider = float(lambda_slider)
        self.lambda_perceptual = float(lambda_perceptual)
        self.lambda_lut_smooth = float(lambda_lut_smooth)
        self.lambda_gamut = float(lambda_gamut)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

        self.perceptual = (
            VGG16PerceptualLoss(image_size=int(perceptual_size))
            if self.lambda_perceptual > 0.0
            else None
        )

    @staticmethod
    def _lab(x: torch.Tensor) -> torch.Tensor:
        return srgb_to_oklab_channel_first(x.float())

    @staticmethod
    def _ab_l1(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return 0.5 * (
            F.l1_loss(a[:, 1:2], b[:, 1:2])
            + F.l1_loss(a[:, 2:3], b[:, 2:3])
        )

    @staticmethod
    def _alpha_image(alpha, ref: torch.Tensor) -> torch.Tensor:
        if torch.is_tensor(alpha):
            x = alpha.to(device=ref.device, dtype=ref.dtype)
        else:
            x = torch.tensor(float(alpha), device=ref.device, dtype=ref.dtype)
        if x.ndim == 0:
            return x
        if x.ndim == 1:
            return x.view(-1, 1, 1, 1)
        return x

    @staticmethod
    def _tv3d(field: torch.Tensor) -> torch.Tensor:
        terms = []
        if field.shape[-1] > 1:
            terms.append((field[..., 1:] - field[..., :-1]).abs().mean())
        if field.shape[-2] > 1:
            terms.append((field[..., 1:, :] - field[..., :-1, :]).abs().mean())
        if field.shape[-3] > 1:
            terms.append((field[..., 1:, :, :] - field[..., :-1, :, :]).abs().mean())
        if not terms:
            return field.new_zeros(())
        return sum(terms) / len(terms)

    def _style_field_smoothness(self, result: dict) -> tuple[torch.Tensor, torch.Tensor]:
        weights = {8: 0.25, 16: 0.5, 32: 1.0}
        tone_terms = []
        chroma_terms = []
        for state_key in ("state_a", "state_b"):
            pyramid = result[state_key]["style_fields"]["pyramid"]
            for resolution, weight in weights.items():
                tone_terms.append(weight * self._tv3d(pyramid[f"tone_{resolution}"]))
                chroma_terms.append(weight * self._tv3d(pyramid[f"chroma_{resolution}"]))
        norm = 2.0 * sum(weights.values())
        tone_smooth = sum(tone_terms) / norm
        chroma_smooth = sum(chroma_terms) / norm
        return tone_smooth, chroma_smooth

    def _direction_losses(
        self,
        tonal_output: torch.Tensor,
        chroma_output: torch.Tensor,
        canonical: torch.Tensor,
        target: torch.Tensor,
    ):
        tonal_lab = self._lab(tonal_output)
        chroma_lab = self._lab(chroma_output)
        target_lab = self._lab(target)
        canonical_lab = self._lab(canonical).detach()

        tonal_match = self.l1(tonal_lab[:, 0:1], target_lab[:, 0:1])
        tonal_chroma_leak = self._ab_l1(tonal_lab, canonical_lab)

        chroma_match = self._ab_l1(chroma_lab, target_lab)
        chroma_tone_leak = self.l1(chroma_lab[:, 0:1], canonical_lab[:, 0:1])
        return tonal_match, tonal_chroma_leak, chroma_match, chroma_tone_leak

    def _slider_direction_loss(
        self,
        controlled_output: torch.Tensor,
        canonical: torch.Tensor,
        target: torch.Tensor,
        alpha_t,
        alpha_c,
    ) -> torch.Tensor:
        output_lab = self._lab(controlled_output)
        canonical_lab = self._lab(canonical).detach()
        target_lab = self._lab(target).detach()

        at = self._alpha_image(alpha_t, output_lab)
        ac = self._alpha_image(alpha_c, output_lab)

        target_l = (1.0 - at) * canonical_lab[:, 0:1] + at * target_lab[:, 0:1]
        target_ab = (1.0 - ac) * canonical_lab[:, 1:3] + ac * target_lab[:, 1:3]

        loss_l = F.l1_loss(output_lab[:, 0:1], target_l)
        loss_ab = F.l1_loss(output_lab[:, 1:3], target_ab)
        return loss_l + loss_ab

    def forward(self, result: dict, image_a: torch.Tensor, image_b: torch.Tensor) -> dict:
        ca = result["state_a"]["canonical"]
        cb = result["state_b"]["canonical"]

        consistency = self.mse(ca, cb)
        rec_ab = self.l1(result["output_ab"], image_b)
        rec_ba = self.l1(result["output_ba"], image_a)
        reconstruction = rec_ab + rec_ba

        if self.perceptual is not None:
            perceptual_ab = self.perceptual(result["output_ab"], image_b)
            perceptual_ba = self.perceptual(result["output_ba"], image_a)
            perceptual = perceptual_ab + perceptual_ba
        else:
            perceptual = reconstruction.new_zeros(())

        t_ab, tl_ab, c_ab, cl_ab = self._direction_losses(
            result["output_ab_tonal_only"],
            result["output_ab_chroma_only"],
            ca,
            image_b,
        )
        t_ba, tl_ba, c_ba, cl_ba = self._direction_losses(
            result["output_ba_tonal_only"],
            result["output_ba_chroma_only"],
            cb,
            image_a,
        )

        tonal_match = t_ab + t_ba
        chromatic_match = c_ab + c_ba
        tonal_chroma_leak = tl_ab + tl_ba
        chroma_tone_leak = cl_ab + cl_ba
        leakage = tonal_chroma_leak + chroma_tone_leak

        if "output_ab_controlled" in result:
            slider_ab = self._slider_direction_loss(
                result["output_ab_controlled"],
                ca,
                image_b,
                result["controlled_tonal_strength"],
                result["controlled_chromatic_strength"],
            )
            slider_ba = self._slider_direction_loss(
                result["output_ba_controlled"],
                cb,
                image_a,
                result["controlled_tonal_strength"],
                result["controlled_chromatic_strength"],
            )
            slider = slider_ab + slider_ba
        else:
            slider = reconstruction.new_zeros(())

        tone_smooth, chroma_smooth = self._style_field_smoothness(result)
        lut_smooth = tone_smooth + chroma_smooth
        gamut = result["gamut_penalty"]

        total = (
            reconstruction
            + self.lambda_consistency * consistency
            + self.lambda_tonal * tonal_match
            + self.lambda_chromatic * chromatic_match
            + self.lambda_leakage * leakage
            + self.lambda_slider * slider
            + self.lambda_perceptual * perceptual
            + self.lambda_lut_smooth * lut_smooth
            + self.lambda_gamut * gamut
        )

        return {
            "consistency_loss": consistency,
            "reconstruction_a_to_b": rec_ab,
            "reconstruction_b_to_a": rec_ba,
            "reconstruction_loss": reconstruction,
            "perceptual_loss": perceptual,
            "tonal_match_loss": tonal_match,
            "chromatic_match_loss": chromatic_match,
            "tonal_chroma_leak_loss": tonal_chroma_leak,
            "chroma_tone_leak_loss": chroma_tone_leak,
            "disentanglement_leakage_loss": leakage,
            "slider_supervision_loss": slider,
            "tone_lut_smooth_loss": tone_smooth,
            "chroma_lut_smooth_loss": chroma_smooth,
            "lut_smooth_loss": lut_smooth,
            "gamut_loss": gamut,
            "total": total,
        }
