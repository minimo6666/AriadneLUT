from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.ariadne_lut_v2 import AriadneLUTV2


def _group_count(channels: int) -> int:
    for groups in (8, 4, 2, 1):
        if channels % groups == 0:
            return groups
    return 1


def build_trilinear_query_field(
    canonical: torch.Tensor,
    dimension: int = 32,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Rasterize the exact trilinear LUT query mass of a canonical RGB image.

    Args:
        canonical: [B, 3, H, W] in [0, 1].
        dimension: LUT side length D.

    Returns:
        q: [B, 1, D, D, D], normalized to sum to one per image.

    Axis convention matches ``apply_lut_v2`` / grid_sample:
      LUT tensor layout is [channel, blue(z), green(y), red(x)].

    Stage-1 is frozen during Stage-2, so the discrete floor/index operation is
    intentionally detached from canonical-image gradients. The resulting query
    field is a conditioning descriptor, not a path for updating the normalizer.
    """
    if canonical.ndim != 4 or canonical.shape[1] != 3:
        raise ValueError("canonical must have shape [B, 3, H, W]")
    d = int(dimension)
    if d < 2:
        raise ValueError("dimension must be >= 2")

    image = canonical.detach().float().clamp(0.0, 1.0)
    bsz, _, height, width = image.shape
    colors = image.permute(0, 2, 3, 1).reshape(bsz, -1, 3)
    coord = colors * float(d - 1)
    lower = torch.floor(coord).to(torch.long)
    upper = torch.clamp(lower + 1, max=d - 1)
    frac = coord - lower.to(coord.dtype)

    q_flat = torch.zeros(
        bsz,
        d * d * d,
        device=image.device,
        dtype=torch.float32,
    )

    # RGB coordinates map to LUT [B-axis, G-axis, R-axis].
    for use_hi_r in (0, 1):
        for use_hi_g in (0, 1):
            for use_hi_b in (0, 1):
                r_idx = upper[..., 0] if use_hi_r else lower[..., 0]
                g_idx = upper[..., 1] if use_hi_g else lower[..., 1]
                b_idx = upper[..., 2] if use_hi_b else lower[..., 2]

                wr = frac[..., 0] if use_hi_r else (1.0 - frac[..., 0])
                wg = frac[..., 1] if use_hi_g else (1.0 - frac[..., 1])
                wb = frac[..., 2] if use_hi_b else (1.0 - frac[..., 2])
                weight = (wr * wg * wb).to(torch.float32)

                flat_index = (b_idx * d + g_idx) * d + r_idx
                q_flat.scatter_add_(1, flat_index, weight)

    q = q_flat.view(bsz, 1, d, d, d)
    q = q / q.sum(dim=(2, 3, 4), keepdim=True).clamp_min(float(eps))
    return q


class CanonicalColorEncoder(nn.Module):
    """Small global encoder for Z_C.

    It is deliberately much smaller than the Stage-1 Transformer encoders.
    Stage-2 needs a global color-domain descriptor, not a second semantic image
    backbone. Spatial information is aggressively pooled into one global code.
    """

    def __init__(self, code_dim: int = 96, base_channels: int = 24):
        super().__init__()
        c = int(base_channels)
        self.net = nn.Sequential(
            nn.Conv2d(3, c, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            nn.Conv2d(c, c * 2, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c * 2), c * 2),
            nn.GELU(),
            nn.Conv2d(c * 2, c * 3, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c * 3), c * 3),
            nn.GELU(),
            nn.Conv2d(c * 3, c * 4, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(_group_count(c * 4), c * 4),
            nn.GELU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.proj = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c * 4, int(code_dim)),
            nn.LayerNorm(int(code_dim)),
            nn.GELU(),
        )

    def forward(self, canonical: torch.Tensor) -> torch.Tensor:
        return self.proj(self.net(canonical))


class Residual3DBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        c = int(channels)
        self.net = nn.Sequential(
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            nn.Conv3d(c, c, kernel_size=3, padding=1),
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            nn.Conv3d(c, c, kernel_size=3, padding=1),
        )
        # Identity at initialization.
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class ContentAwareResidualLUTRefiner(nn.Module):
    """Predict a bounded residual LUT on top of frozen Stage-1 L_S.

    Inputs live in the LUT lattice itself:
      * Stage-1 base Style LUT L_S          : 3 channels
      * content canonical query field Q_C  : 1 channel
      * style canonical query field Q_S    : 1 channel
      * global (style-code, Z_C-code) condition broadcast into the cube

    The final convolution is zero initialized. Therefore Stage-2 starts exactly
    from the Stage-1 output: Delta L = 0 at initialization.
    """

    def __init__(
        self,
        style_dim: int,
        content_dim: int = 96,
        condition_dim: int = 64,
        condition_volume_channels: int = 8,
        channels: int = 32,
        blocks: int = 4,
        max_delta: float = 0.20,
    ) -> None:
        super().__init__()
        self.max_delta = float(max_delta)
        self.condition_volume_channels = int(condition_volume_channels)

        self.condition = nn.Sequential(
            nn.LayerNorm(int(style_dim) + int(content_dim)),
            nn.Linear(int(style_dim) + int(content_dim), int(condition_dim)),
            nn.GELU(),
            nn.Linear(int(condition_dim), int(condition_volume_channels)),
            nn.Tanh(),
        )

        input_channels = 3 + 1 + 1 + int(condition_volume_channels)
        c = int(channels)
        self.stem = nn.Conv3d(input_channels, c, kernel_size=3, padding=1)
        self.blocks = nn.Sequential(*[Residual3DBlock(c) for _ in range(int(blocks))])
        self.head = nn.Sequential(
            nn.GroupNorm(_group_count(c), c),
            nn.GELU(),
            nn.Conv3d(c, 3, kernel_size=3, padding=1),
        )
        # Critical for smooth Stage-1 -> Stage-2 initialization.
        nn.init.zeros_(self.head[-1].weight)
        nn.init.zeros_(self.head[-1].bias)

    def forward(
        self,
        base_lut: torch.Tensor,
        query_content: torch.Tensor,
        query_style: torch.Tensor,
        style_code: torch.Tensor,
        content_code: torch.Tensor,
    ) -> torch.Tensor:
        if base_lut.ndim != 5 or base_lut.shape[1] != 3:
            raise ValueError("base_lut must have shape [B,3,D,D,D]")
        d = base_lut.shape[-1]
        if query_content.shape[-3:] != (d, d, d):
            raise ValueError("query_content LUT dimension does not match base_lut")
        if query_style.shape[-3:] != (d, d, d):
            raise ValueError("query_style LUT dimension does not match base_lut")

        condition = self.condition(torch.cat([style_code, content_code], dim=1))
        condition = condition[:, :, None, None, None].expand(-1, -1, d, d, d)
        x = torch.cat(
            [
                base_lut.float(),
                query_content.float(),
                query_style.float(),
                condition.float(),
            ],
            dim=1,
        )
        x = self.stem(x)
        x = self.blocks(x)
        raw_delta = self.head(x)
        return self.max_delta * torch.tanh(raw_delta)


class AriadneMovieNetPairStage2(nn.Module):
    """MovieNet cross-content Stage-2 wrapper around a frozen AriadneLUTV2.

    Frozen Stage-1:
        C -> N(C) = Z_C
        S -> r_S -> L_base(S)

    Trainable Stage-2:
        (Z_C, Q_C, Q_S, r_S, L_base) -> Delta L_{C,S}
        L_{C,S} = L_base + alpha * Delta L_{C,S}
        O = L_{C,S}(Z_C)

    ``alpha`` is supplied by the training script and ramps 0 -> 1 during the
    first optimizer steps for a deliberately gentle fine-tuning transition.
    """

    def __init__(self, stage1: AriadneLUTV2, cfg: Any) -> None:
        super().__init__()
        self.stage1 = stage1
        self.query_dimension = int(getattr(cfg, "query_dimension", 32))

        content_dim = int(getattr(cfg, "content_dim", 96))
        self.content_encoder = CanonicalColorEncoder(
            code_dim=content_dim,
            base_channels=int(getattr(cfg, "content_base_channels", 24)),
        )
        self.residual_refiner = ContentAwareResidualLUTRefiner(
            style_dim=int(stage1.cfg.style_dim),
            content_dim=content_dim,
            condition_dim=int(getattr(cfg, "condition_dim", 64)),
            condition_volume_channels=int(
                getattr(cfg, "condition_volume_channels", 8)
            ),
            channels=int(getattr(cfg, "refiner_channels", 32)),
            blocks=int(getattr(cfg, "refiner_blocks", 4)),
            max_delta=float(getattr(cfg, "max_delta", 0.20)),
        )
        self.freeze_stage1()

    def freeze_stage1(self) -> None:
        self.stage1.eval()
        for parameter in self.stage1.parameters():
            parameter.requires_grad_(False)

    def train(self, mode: bool = True):
        super().train(mode)
        # Never let model.train() put the frozen Stage-1 backbone back in train mode.
        self.stage1.eval()
        return self

    def trainable_parameters(self):
        for module in (self.content_encoder, self.residual_refiner):
            yield from module.parameters()

    def stage2_state_dict(self) -> dict[str, dict[str, torch.Tensor]]:
        return {
            "content_encoder": self.content_encoder.state_dict(),
            "residual_refiner": self.residual_refiner.state_dict(),
        }

    def load_stage2_state_dict(
        self,
        state: dict[str, dict[str, torch.Tensor]],
        strict: bool = True,
    ) -> None:
        self.content_encoder.load_state_dict(state["content_encoder"], strict=strict)
        self.residual_refiner.load_state_dict(state["residual_refiner"], strict=strict)

    def forward(
        self,
        content: torch.Tensor,
        style: torch.Tensor,
        residual_strength: float | torch.Tensor = 1.0,
    ) -> dict[str, torch.Tensor | dict]:
        # Stage-1 is intentionally fully frozen. No graph is retained for it.
        with torch.no_grad():
            content_state = self.stage1.normalize(content)
            style_state = self.stage1.extract_style(style)
            style_norm_state = self.stage1.normalize(style)

            canonical = content_state["canonical"].detach()
            style_canonical = style_norm_state["canonical"].detach()
            base_lut = style_state["style_lut"].detach()
            style_code = style_state["style_code"].detach()
            baseline_output = self.stage1.apply_style(canonical, base_lut).detach()

            query_content = build_trilinear_query_field(
                canonical,
                dimension=self.query_dimension,
            )
            query_style = build_trilinear_query_field(
                style_canonical,
                dimension=self.query_dimension,
            )

        content_code = self.content_encoder(canonical.float())
        delta_lut = self.residual_refiner(
            base_lut=base_lut,
            query_content=query_content,
            query_style=query_style,
            style_code=style_code.float(),
            content_code=content_code,
        )

        if torch.is_tensor(residual_strength):
            alpha = residual_strength.to(device=delta_lut.device, dtype=delta_lut.dtype)
        else:
            alpha = delta_lut.new_tensor(float(residual_strength))
        final_lut = base_lut.float() + alpha * delta_lut
        output = self.stage1.apply_style(canonical.float(), final_lut)

        return {
            "output": output,
            "baseline_output": baseline_output,
            "canonical": canonical,
            "style_canonical": style_canonical,
            "base_lut": base_lut,
            "delta_lut": delta_lut,
            "final_lut": final_lut,
            "query_content": query_content,
            "query_style": query_style,
            "style_code": style_code,
            "content_code": content_code,
        }
