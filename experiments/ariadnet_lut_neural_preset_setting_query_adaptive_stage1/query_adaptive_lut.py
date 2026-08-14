from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F

from query_field import QuerySupport, build_query_support, lattice_coordinates


def identity_lut(dimension: int, device=None, dtype=None) -> torch.Tensor:
    values = torch.linspace(0.0, 1.0, int(dimension), device=device, dtype=dtype)
    blue, green, red = torch.meshgrid(values, values, values, indexing="ij")
    return torch.stack((red, green, blue), dim=0).unsqueeze(0)


class Residual3DBlock(nn.Module):
    """The only dense 3D feature block retained by the new decoder (8^3)."""

    def __init__(self, channels: int):
        super().__init__()
        groups = 8 if int(channels) % 8 == 0 else 1
        self.net = nn.Sequential(
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(int(channels), int(channels), 3, padding=1),
            nn.GroupNorm(groups, int(channels)),
            nn.GELU(),
            nn.Conv3d(int(channels), int(channels), 3, padding=1),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


def _fourier_coordinates(coords: torch.Tensor, frequencies: int) -> torch.Tensor:
    pieces = [coords]
    for power in range(int(frequencies)):
        omega = (2.0 ** power) * torch.pi
        pieces.append(torch.sin(omega * coords))
        pieces.append(torch.cos(omega * coords))
    return torch.cat(pieces, dim=-1)


@dataclass
class SparseLevelOutput:
    lut: torch.Tensor
    delta: torch.Tensor
    support: QuerySupport
    evidence_mass: Optional[torch.Tensor]
    evidence_summary: Optional[torch.Tensor]
    token_count: int


class SparseColorDisplacementRefiner(nn.Module):
    """Decode only the queried high-resolution color displacement tokens.

    The model does *not* construct a high-channel K^3 feature volume. It packs
    only active lattice vertices into a token matrix, predicts an RGB residual
    displacement for those tokens, and scatters the 3-channel result back to a
    LUT solely for the final trilinear lookup.
    """

    def __init__(
        self,
        *,
        condition_dim: int,
        hidden_dim: int,
        depth: int,
        coord_frequencies: int,
        use_evidence: bool,
        local_smoothing_blend: float,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.coord_frequencies = int(coord_frequencies)
        self.use_evidence = bool(use_evidence)
        self.local_smoothing_blend = float(local_smoothing_blend)

        coord_dim = 3 * (1 + 2 * self.coord_frequencies)
        # coord PE + query mass + local mass + base RGB + base displacement
        local_dim = coord_dim + 1 + 1 + 3 + 3
        if self.use_evidence:
            local_dim += 2  # evidence mass + local evidence mass

        # Query summary = meanRGB(3), stdRGB(3), normalized entropy, N_eff ratio.
        global_dim = int(condition_dim) + 8 + (8 if self.use_evidence else 0) + 1

        self.local_proj = nn.Sequential(
            nn.Linear(local_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )
        self.global_proj = nn.Sequential(
            nn.Linear(global_dim, self.hidden_dim),
            nn.GELU(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

        blocks = []
        for _ in range(max(int(depth), 1)):
            blocks.append(
                nn.Sequential(
                    nn.LayerNorm(self.hidden_dim),
                    nn.Linear(self.hidden_dim, self.hidden_dim * 2),
                    nn.GELU(),
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim),
                )
            )
        self.blocks = nn.ModuleList(blocks)
        self.to_delta = nn.Linear(self.hidden_dim, 3)
        # Start from the dense coarse interpolation. Fine capacity is introduced
        # only as training finds evidence for it.
        nn.init.zeros_(self.to_delta.weight)
        nn.init.zeros_(self.to_delta.bias)

    def _smooth_sparse_delta(
        self,
        delta: torch.Tensor,
        active_mask: torch.Tensor,
    ) -> torch.Tensor:
        blend = self.local_smoothing_blend
        if blend <= 0.0:
            return delta
        active = active_mask.float()
        numerator = F.avg_pool3d(delta * active, 3, stride=1, padding=1)
        denominator = F.avg_pool3d(active, 3, stride=1, padding=1).clamp_min(1e-6)
        local_mean = numerator / denominator
        return ((1.0 - blend) * delta + blend * local_mean) * active

    def forward(
        self,
        *,
        condition: torch.Tensor,
        base_lut: torch.Tensor,
        support: QuerySupport,
        residual_scale: float,
        level_scalar: float,
        evidence_support: Optional[QuerySupport] = None,
    ) -> SparseLevelOutput:
        batch, channels, k, _, _ = base_lut.shape
        if channels != 3 or k != support.dimension:
            raise ValueError("base_lut/support dimension mismatch")
        if self.use_evidence and evidence_support is None:
            raise ValueError("evidence support is required for this refiner")

        active = support.active_mask
        batch_index, flat_index = active.nonzero(as_tuple=True)
        token_count = int(flat_index.numel())
        if token_count == 0:
            # This should not happen for non-empty images, but keep the forward
            # numerically well-defined.
            zero = torch.zeros_like(base_lut)
            return SparseLevelOutput(
                lut=base_lut,
                delta=zero,
                support=support,
                evidence_mass=None,
                evidence_summary=None,
                token_count=0,
            )

        n = k ** 3
        coords = lattice_coordinates(k, device=base_lut.device, dtype=base_lut.dtype)
        token_coords = coords.index_select(0, flat_index)
        coord_features = _fourier_coordinates(token_coords, self.coord_frequencies)

        base_flat = base_lut.permute(0, 2, 3, 4, 1).reshape(batch, n, 3)
        base_rgb = base_flat[batch_index, flat_index]
        base_displacement = base_rgb - token_coords

        q = support.mass[batch_index, flat_index]
        q_local = support.local_mass[batch_index, flat_index]
        # log1p(N*q) keeps rare and dominant colors in a useful numeric range.
        q_feature = torch.log1p(q * float(n))[:, None]
        q_local_feature = torch.log1p(q_local * float(n))[:, None]

        local_pieces = [
            coord_features,
            q_feature,
            q_local_feature,
            base_rgb,
            base_displacement,
        ]

        evidence_mass = None
        evidence_summary = None
        if self.use_evidence:
            assert evidence_support is not None
            evidence_mass = evidence_support.mass
            evidence_summary = evidence_support.summary
            eq = evidence_support.mass[batch_index, flat_index]
            eql = evidence_support.local_mass[batch_index, flat_index]
            local_pieces.extend(
                [
                    torch.log1p(eq * float(n))[:, None],
                    torch.log1p(eql * float(n))[:, None],
                ]
            )

        local = self.local_proj(torch.cat(local_pieces, dim=-1))

        global_pieces = [condition, support.summary]
        if self.use_evidence:
            assert evidence_summary is not None
            global_pieces.append(evidence_summary)
        global_pieces.append(
            torch.full(
                (batch, 1),
                float(level_scalar),
                device=condition.device,
                dtype=condition.dtype,
            )
        )
        global_feature = self.global_proj(torch.cat(global_pieces, dim=-1))
        hidden = local + global_feature[batch_index]
        for block in self.blocks:
            hidden = hidden + block(hidden)

        raw_delta = torch.tanh(self.to_delta(hidden))
        token_gate = support.gate[batch_index, flat_index][:, None].to(raw_delta.dtype)
        token_delta = float(residual_scale) * token_gate * raw_delta

        dense_flat = torch.zeros(
            batch,
            n,
            3,
            device=base_lut.device,
            dtype=base_lut.dtype,
        )
        dense_flat[batch_index, flat_index] = token_delta.to(base_lut.dtype)
        delta = dense_flat.view(batch, k, k, k, 3).permute(0, 4, 1, 2, 3)

        active_3d = active.view(batch, 1, k, k, k)
        delta = self._smooth_sparse_delta(delta, active_3d)
        lut = base_lut + delta
        return SparseLevelOutput(
            lut=lut,
            delta=delta,
            support=support,
            evidence_mass=evidence_mass,
            evidence_summary=evidence_summary,
            token_count=token_count,
        )


class QueryAdaptiveHierarchicalLUTGenerator(nn.Module):
    """Dense 8^3 global mapping + sparse 16^3/32^3 displacement refinement."""

    def __init__(
        self,
        *,
        code_dim: int,
        base_cfg,
        qa_cfg: dict,
        residual_scale_8: float,
        residual_scale_16: float,
        residual_scale_32: float,
        use_evidence: bool,
    ) -> None:
        super().__init__()
        channels = int(base_cfg.lut_channels)
        condition_dim = int(base_cfg.condition_dim)
        self.condition_dim = condition_dim
        self.scale_8 = float(residual_scale_8)
        self.scale_16 = float(residual_scale_16)
        self.scale_32 = float(residual_scale_32)
        self.use_evidence = bool(use_evidence)

        self.mass_fraction_16 = float(qa_cfg.get("mass_fraction_16", 0.99))
        self.mass_fraction_32 = float(qa_cfg.get("mass_fraction_32", 0.99))
        self.dilation_16 = int(qa_cfg.get("dilation_16", 1))
        self.dilation_32 = int(qa_cfg.get("dilation_32", 1))
        self.shell_decay = float(qa_cfg.get("shell_decay", 0.35))
        self.max_query_pixels = int(qa_cfg.get("max_query_pixels", 16384))

        self.condition = nn.Sequential(
            nn.LayerNorm(int(code_dim)),
            nn.Linear(int(code_dim), condition_dim),
            nn.GELU(),
            nn.Linear(condition_dim, condition_dim),
        )
        self.seed = nn.Linear(condition_dim, channels * 8 * 8 * 8)
        self.stage_8 = nn.Sequential(
            Residual3DBlock(channels),
            Residual3DBlock(channels),
        )
        self.to_delta_8 = nn.Conv3d(channels, 3, kernel_size=3, padding=1)
        nn.init.zeros_(self.to_delta_8.weight)
        nn.init.zeros_(self.to_delta_8.bias)

        hidden = int(qa_cfg.get("token_hidden_dim", 192))
        depth = int(qa_cfg.get("token_depth", 2))
        coord_freq = int(qa_cfg.get("coord_fourier_frequencies", 4))
        smooth_blend = float(qa_cfg.get("local_smoothing_blend", 0.10))
        self.refine_16 = SparseColorDisplacementRefiner(
            condition_dim=condition_dim,
            hidden_dim=hidden,
            depth=depth,
            coord_frequencies=coord_freq,
            use_evidence=self.use_evidence,
            local_smoothing_blend=smooth_blend,
        )
        self.refine_32 = SparseColorDisplacementRefiner(
            condition_dim=condition_dim,
            hidden_dim=hidden,
            depth=depth,
            coord_frequencies=coord_freq,
            use_evidence=self.use_evidence,
            local_smoothing_blend=smooth_blend,
        )

    def _support(self, image: torch.Tensor, k: int) -> QuerySupport:
        if int(k) == 16:
            fraction, dilation = self.mass_fraction_16, self.dilation_16
        elif int(k) == 32:
            fraction, dilation = self.mass_fraction_32, self.dilation_32
        else:
            raise ValueError("only 16 and 32 are sparse refinement levels")
        return build_query_support(
            image,
            int(k),
            mass_fraction=fraction,
            dilation=dilation,
            shell_decay=self.shell_decay,
            max_pixels=self.max_query_pixels,
        )

    def forward(
        self,
        code: torch.Tensor,
        *,
        query_image: torch.Tensor,
        evidence_image: Optional[torch.Tensor] = None,
    ):
        condition = self.condition(code)
        batch = condition.shape[0]

        features_8 = self.seed(condition).view(batch, -1, 8, 8, 8)
        features_8 = self.stage_8(features_8)
        identity_8 = identity_lut(
            8, device=features_8.device, dtype=features_8.dtype
        ).expand(batch, -1, -1, -1, -1)
        delta_8 = self.scale_8 * torch.tanh(self.to_delta_8(features_8))
        lut_8 = identity_8 + delta_8

        base_16 = F.interpolate(
            lut_8, size=(16, 16, 16), mode="trilinear", align_corners=True
        )
        support_16 = self._support(query_image, 16)
        evidence_16 = self._support(evidence_image, 16) if self.use_evidence else None
        level_16 = self.refine_16(
            condition=condition,
            base_lut=base_16,
            support=support_16,
            residual_scale=self.scale_16,
            level_scalar=0.5,
            evidence_support=evidence_16,
        )
        lut_16 = level_16.lut

        base_32 = F.interpolate(
            lut_16, size=(32, 32, 32), mode="trilinear", align_corners=True
        )
        support_32 = self._support(query_image, 32)
        evidence_32 = self._support(evidence_image, 32) if self.use_evidence else None
        level_32 = self.refine_32(
            condition=condition,
            base_lut=base_32,
            support=support_32,
            residual_scale=self.scale_32,
            level_scalar=1.0,
            evidence_support=evidence_32,
        )
        lut_32 = level_32.lut

        pyramid = {
            "lut_8": lut_8,
            "lut_16": lut_16,
            "lut_32": lut_32,
            "delta_8": delta_8,
            "level_16": level_16,
            "level_32": level_32,
        }
        return lut_32, pyramid
