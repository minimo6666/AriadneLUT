from __future__ import annotations

import torch
import torch.nn as nn

from .transformer import PatchEmbed, SelfAttentionBlock


class _ImageCodeEncoderV2(nn.Module):
    """
    Transformer image encoder used by one independent V2 branch.

    The normalization encoder and style encoder have the same architecture
    but do not share parameters. Each branch therefore receives only its own
    image and cannot inspect information from the other branch.
    """

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
    """
    Extract one normalization code d_C from the input image C.

    The code is allowed to change with the current source grade because
    differently graded inputs require different normalization LUTs.
    """

    def __init__(self, cfg):
        super().__init__(
            cfg=cfg,
            code_dim=int(cfg.normalization_dim),
        )


class StyleEncoderV2(_ImageCodeEncoderV2):
    """
    Extract one style code r_S from the style image S.

    The resulting code is passed only to the Style LUT generator.
    """

    def __init__(self, cfg):
        super().__init__(
            cfg=cfg,
            code_dim=int(cfg.style_dim),
        )
