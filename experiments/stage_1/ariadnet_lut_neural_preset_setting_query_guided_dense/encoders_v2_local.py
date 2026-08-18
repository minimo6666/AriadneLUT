from __future__ import annotations

import torch
import torch.nn as nn

# Intentionally reuse the repository's unchanged Transformer primitives so the
# encoder is bit-for-bit the same architecture as the Dense baseline. No root
# model source is modified by this experiment.
from models.transformer import PatchEmbed, SelfAttentionBlock


class _ImageCodeEncoderV2(nn.Module):
    """Local copy of the original V2 image-code encoder."""

    def __init__(self, cfg, code_dim: int):
        super().__init__()
        hidden_dim = int(cfg.hidden_dim)
        depth = int(cfg.depth)
        self.patch = PatchEmbed(
            image_size=int(cfg.encoder_size),
            patch_size=int(cfg.patch_size),
            dim=hidden_dim,
        )
        self.blocks = nn.ModuleList(
            [
                SelfAttentionBlock(
                    dim=hidden_dim,
                    heads=int(cfg.num_heads),
                    mlp_ratio=float(cfg.mlp_ratio),
                    dropout=float(cfg.dropout),
                )
                for _ in range(depth)
            ]
        )
        self.norm = nn.LayerNorm(hidden_dim)
        self.head = nn.Linear(hidden_dim, int(code_dim))

    def forward(self, image: torch.Tensor):
        tokens = self.patch(image)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        code = self.head(tokens.mean(dim=1))
        return tokens, code


class NormalizationEncoderV2(_ImageCodeEncoderV2):
    def __init__(self, cfg):
        super().__init__(cfg=cfg, code_dim=int(cfg.normalization_dim))


class StyleEncoderV2(_ImageCodeEncoderV2):
    def __init__(self, cfg):
        super().__init__(cfg=cfg, code_dim=int(cfg.style_dim))
