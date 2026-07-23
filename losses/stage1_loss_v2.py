from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .reconstruction import reconstruction_components
from .regularization import (
    lut_monotonicity,
    lut_range,
    lut_smoothness,
)


def canonical_margin_consistency_v2(
    canonical_a: torch.Tensor,
    canonical_b: torch.Tensor,
    margin: float,
    eps: float = 1e-3,
):
    """
    Keep two canonical images of the same content within a tolerated range.

    A per-sample Charbonnier distance is measured first. Only the amount above
    `margin` is penalized, which avoids forcing exact equality when synthetic
    grading has caused irreversible clipping or quantization-like loss.
    """
    per_pixel = torch.sqrt(
        (canonical_a - canonical_b).square()
        + float(eps) ** 2
    )
    distance_per_sample = per_pixel.mean(
        dim=(1, 2, 3)
    )

    excess = F.relu(
        distance_per_sample - float(margin)
    )
    consistency = excess.square().mean()

    active_ratio = (
        distance_per_sample > float(margin)
    ).float().mean()

    return (
        consistency,
        distance_per_sample.mean(),
        active_ratio,
    )


class Stage1LossV2(nn.Module):
    """
    Stage-1 V2 objective.

    The loss contains:

    1. Canonical consistency:
       the two grades of the same content must normalize into nearby
       canonical images.

    2. Bidirectional cross-grade reconstruction:
       canonical_A + StyleLUT_B -> image_B
       canonical_B + StyleLUT_A -> image_A

    3. Independent regularization for Normalization LUTs and Style LUTs.

    No normalization-code consistency is used. Different source grades are
    allowed to produce different normalization codes and different
    Normalization LUTs, as long as they reach the same canonical space.
    """

    def __init__(self, cfg):
        super().__init__()

        self.weights = cfg.loss
        self.canonical_margin = float(
            cfg.loss.canonical_margin
        )

    def forward(
        self,
        pair_result,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ):
        state_a = pair_result["state_a"]
        state_b = pair_result["state_b"]

        reconstruction_ab = reconstruction_components(
            pair_result["output_ab"],
            image_b,
        )
        reconstruction_ba = reconstruction_components(
            pair_result["output_ba"],
            image_a,
        )

        losses = {
            key: 0.5
            * (
                reconstruction_ab[key]
                + reconstruction_ba[key]
            )
            for key in reconstruction_ab
        }

        (
            canonical_consistency,
            canonical_distance,
            canonical_active_ratio,
        ) = canonical_margin_consistency_v2(
            canonical_a=state_a["canonical"],
            canonical_b=state_b["canonical"],
            margin=self.canonical_margin,
        )

        losses["canonical_consistency"] = (
            canonical_consistency
        )

        # Diagnostics are logged but have no weight in the YAML.
        losses["canonical_distance"] = canonical_distance
        losses["canonical_active_ratio"] = (
            canonical_active_ratio
        )

        normalization_lut_a = state_a[
            "normalization_lut"
        ]
        normalization_lut_b = state_b[
            "normalization_lut"
        ]

        style_lut_a = state_a["style_lut"]
        style_lut_b = state_b["style_lut"]

        losses["normalization_lut_smoothness"] = (
            0.5
            * (
                lut_smoothness(normalization_lut_a)
                + lut_smoothness(
                    normalization_lut_b
                )
            )
        )
        losses["normalization_lut_monotonicity"] = (
            0.5
            * (
                lut_monotonicity(
                    normalization_lut_a
                )
                + lut_monotonicity(
                    normalization_lut_b
                )
            )
        )
        losses["normalization_lut_range"] = (
            0.5
            * (
                lut_range(normalization_lut_a)
                + lut_range(normalization_lut_b)
            )
        )

        losses["style_lut_smoothness"] = (
            0.5
            * (
                lut_smoothness(style_lut_a)
                + lut_smoothness(style_lut_b)
            )
        )
        losses["style_lut_monotonicity"] = (
            0.5
            * (
                lut_monotonicity(style_lut_a)
                + lut_monotonicity(style_lut_b)
            )
        )
        losses["style_lut_range"] = (
            0.5
            * (
                lut_range(style_lut_a)
                + lut_range(style_lut_b)
            )
        )

        total = torch.zeros(
            (),
            device=image_a.device,
        )

        for key, value in losses.items():
            if hasattr(self.weights, key):
                total = (
                    total
                    + float(getattr(self.weights, key))
                    * value
                )

        losses["total"] = total
        return losses
