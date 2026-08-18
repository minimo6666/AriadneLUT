from __future__ import annotations

import torch
import torch.nn as nn


class NeuralPresetObjectiveForAriadne(nn.Module):
    """Neural Preset's exact bidirectional reconstruction objective."""

    def __init__(self, lambda_consistency: float = 10.0) -> None:
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def forward(
        self,
        pair_result: dict,
        image_a: torch.Tensor,
        image_b: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        canonical_a = pair_result["state_a"]["canonical"]
        canonical_b = pair_result["state_b"]["canonical"]

        consistency_loss = self.mse(canonical_a, canonical_b)
        reconstruction_a_to_b = self.l1(pair_result["output_ab"], image_b)
        reconstruction_b_to_a = self.l1(pair_result["output_ba"], image_a)
        reconstruction_loss = reconstruction_a_to_b + reconstruction_b_to_a
        total = reconstruction_loss + self.lambda_consistency * consistency_loss

        return {
            "consistency_loss": consistency_loss,
            "reconstruction_a_to_b": reconstruction_a_to_b,
            "reconstruction_b_to_a": reconstruction_b_to_a,
            "reconstruction_loss": reconstruction_loss,
            "total": total,
        }
