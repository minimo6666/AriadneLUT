from __future__ import annotations

import torch
import torch.nn as nn



def _weighted_tv(delta: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    """Query-weighted first-order smoothness of a sparse fine displacement."""
    b, _, k, _, _ = delta.shape
    weight = mass.view(b, 1, k, k, k).to(delta.dtype)
    total = delta.new_zeros(())
    denom = delta.new_zeros(())
    for axis in (2, 3, 4):
        left = [slice(None)] * 5
        right = [slice(None)] * 5
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        d = (delta[tuple(right)] - delta[tuple(left)]).abs()
        w = 0.5 * (weight[tuple(right)] + weight[tuple(left)])
        total = total + (d * w).sum()
        denom = denom + w.sum() * delta.shape[1]
    return total / denom.clamp_min(1e-8)


def _query_weighted_energy(delta: torch.Tensor, mass: torch.Tensor) -> torch.Tensor:
    b, _, k, _, _ = delta.shape
    weight = mass.view(b, 1, k, k, k).to(delta.dtype)
    return (delta.square() * weight).sum() / (weight.sum() * 3.0).clamp_min(1e-8)


def _shell_energy(delta: torch.Tensor, core: torch.Tensor, active: torch.Tensor) -> torch.Tensor:
    b, _, k, _, _ = delta.shape
    shell = (active & ~core).view(b, 1, k, k, k).to(delta.dtype)
    if float(shell.sum().detach()) == 0.0:
        return delta.new_zeros(())
    return (delta.abs() * shell).sum() / (shell.sum() * 3.0).clamp_min(1.0)


def _curvature(lut: torch.Tensor) -> torch.Tensor:
    parts = []
    for axis in (2, 3, 4):
        a = [slice(None)] * 5
        b = [slice(None)] * 5
        c = [slice(None)] * 5
        a[axis] = slice(2, None)
        b[axis] = slice(1, -1)
        c[axis] = slice(None, -2)
        parts.append((lut[tuple(a)] - 2.0 * lut[tuple(b)] + lut[tuple(c)]).square().mean())
    return sum(parts)


class QueryAdaptiveNeuralPresetObjective(nn.Module):
    """Original Neural-Preset objective + light query-local regularization.

    The reconstruction/canonical losses are intentionally unchanged so the new
    run remains a controlled architecture comparison. Extra terms only regularize
    the *fine sparse displacement* and the tiny dense 8^3 coarse LUT.
    """

    def __init__(self, *, lambda_consistency: float, qa_loss_cfg: dict) -> None:
        super().__init__()
        self.lambda_consistency = float(lambda_consistency)
        self.lambda_delta_energy = float(qa_loss_cfg.get("delta_energy", 0.001))
        self.lambda_query_tv = float(qa_loss_cfg.get("query_tv", 0.01))
        self.lambda_shell = float(qa_loss_cfg.get("shell_taper", 0.002))
        self.lambda_coarse_curvature = float(qa_loss_cfg.get("coarse_curvature", 0.001))
        self.l1 = nn.L1Loss()
        self.mse = nn.MSELoss()

    def _pyramid_regularizers(self, pyramid: dict) -> tuple[torch.Tensor, ...]:
        reference = pyramid["lut_8"]
        energy = reference.new_zeros(())
        tv = reference.new_zeros(())
        shell = reference.new_zeros(())
        for key in ("level_16", "level_32"):
            level = pyramid[key]
            energy = energy + _query_weighted_energy(level.delta, level.support.mass)
            tv = tv + _weighted_tv(level.delta, level.support.mass)
            shell = shell + _shell_energy(
                level.delta,
                level.support.core_mask,
                level.support.active_mask,
            )
        coarse = _curvature(pyramid["lut_8"])
        return energy, tv, shell, coarse

    def forward(self, pair_result: dict, image_a: torch.Tensor, image_b: torch.Tensor) -> dict:
        state_a = pair_result["state_a"]
        state_b = pair_result["state_b"]
        canonical_a = state_a["canonical"]
        canonical_b = state_b["canonical"]

        consistency_loss = self.mse(canonical_a, canonical_b)
        reconstruction_a_to_b = self.l1(pair_result["output_ab"], image_b)
        reconstruction_b_to_a = self.l1(pair_result["output_ba"], image_a)
        reconstruction_loss = reconstruction_a_to_b + reconstruction_b_to_a

        regs = []
        for state in (state_a, state_b):
            regs.append(self._pyramid_regularizers(state["normalization_lut_pyramid"]))
            regs.append(self._pyramid_regularizers(state["style_lut_pyramid"]))
        delta_energy = sum(item[0] for item in regs) / len(regs)
        query_tv = sum(item[1] for item in regs) / len(regs)
        shell_taper = sum(item[2] for item in regs) / len(regs)
        coarse_curvature = sum(item[3] for item in regs) / len(regs)

        base_total = reconstruction_loss + self.lambda_consistency * consistency_loss
        regularization = (
            self.lambda_delta_energy * delta_energy
            + self.lambda_query_tv * query_tv
            + self.lambda_shell * shell_taper
            + self.lambda_coarse_curvature * coarse_curvature
        )
        total = base_total + regularization
        return {
            "consistency_loss": consistency_loss,
            "reconstruction_a_to_b": reconstruction_a_to_b,
            "reconstruction_b_to_a": reconstruction_b_to_a,
            "reconstruction_loss": reconstruction_loss,
            "base_neural_preset_total": base_total,
            "qa_delta_energy": delta_energy,
            "qa_query_tv": query_tv,
            "qa_shell_taper": shell_taper,
            "qa_coarse_curvature": coarse_curvature,
            "qa_regularization": regularization,
            "total": total,
        }
