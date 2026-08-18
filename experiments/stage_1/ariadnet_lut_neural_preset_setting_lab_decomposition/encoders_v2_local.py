
from __future__ import annotations

import torch
import torch.nn as nn

# Unchanged primitive Transformer blocks from the repository.
from models.transformer import PatchEmbed, SelfAttentionBlock


class _ImageCodeEncoderV2Local(nn.Module):
    def __init__(self, cfg, code_dim: int):
        super().__init__()
        hidden_dim = int(cfg.hidden_dim)
        self.patch = PatchEmbed(
            image_size=int(cfg.encoder_size),
            patch_size=int(cfg.patch_size),
            dim=hidden_dim,
        )
        self.blocks = nn.ModuleList([
            SelfAttentionBlock(
                dim=hidden_dim,
                heads=int(cfg.num_heads),
                mlp_ratio=float(cfg.mlp_ratio),
                dropout=float(cfg.dropout),
            )
            for _ in range(int(cfg.depth))
        ])
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, int(code_dim))

    def forward(self, image: torch.Tensor):
        tokens = self.patch(image)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        code = self.head(tokens.mean(dim=1))
        return tokens, code


class NormalizationEncoderV2Local(_ImageCodeEncoderV2Local):
    def __init__(self, cfg):
        super().__init__(cfg, int(cfg.normalization_dim))


class StyleEncoderV2Local(_ImageCodeEncoderV2Local):
    """
    Deliberately keep ONE shared style code in the first controlled experiment.

    The tonal/chromatic split happens inside the Style LUT decoder, so parameter
    count and encoder capacity remain close to the Dense Stage-1 baseline.
    """
    def __init__(self, cfg):
        super().__init__(cfg, int(cfg.style_dim))
