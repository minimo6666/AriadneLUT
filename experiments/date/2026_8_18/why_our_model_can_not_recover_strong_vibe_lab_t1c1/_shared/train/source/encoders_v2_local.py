
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

# Unchanged primitive Transformer blocks from the repository.
from models.transformer import PatchEmbed, SelfAttentionBlock
from oklab_torch import srgb_to_oklab_channel_first


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
        self.style_pooling = str(getattr(cfg, "style_pooling", "mean"))
        self.chroma_pool_blend = float(
            getattr(cfg, "style_chroma_pool_blend", 0.5)
        )
        self.patch_size = int(cfg.patch_size)
        if self.style_pooling not in {"mean", "chroma_aware"}:
            raise ValueError(
                f"Unknown style_pooling={self.style_pooling!r}; "
                "expected 'mean' or 'chroma_aware'"
            )
        if not 0.0 <= self.chroma_pool_blend <= 1.0:
            raise ValueError("style_chroma_pool_blend must be in [0,1]")

    def forward(self, image: torch.Tensor):
        tokens = self.patch(image)
        for block in self.blocks:
            tokens = block(tokens)
        tokens = self.norm(tokens)
        uniform_pool = tokens.mean(dim=1)
        if self.style_pooling == "mean":
            pooled = uniform_pool
        else:
            # Deterministic target-independent saliency: patches with stronger
            # reference Oklab chroma receive more style-code weight. A 50/50
            # blend retains global context and changes only the pooling rule.
            with torch.no_grad():
                lab = srgb_to_oklab_channel_first(image.float())
                chroma = torch.sqrt(
                    lab[:, 1:2].square() + lab[:, 2:3].square() + 1e-12
                )
                patch_chroma = F.avg_pool2d(
                    chroma,
                    kernel_size=self.patch_size,
                    stride=self.patch_size,
                ).flatten(1)
                weights = patch_chroma + 1e-6
                weights = weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-6)
            chroma_pool = (
                tokens * weights.to(dtype=tokens.dtype).unsqueeze(-1)
            ).sum(dim=1)
            blend = self.chroma_pool_blend
            pooled = (1.0 - blend) * uniform_pool + blend * chroma_pool
        code = self.head(pooled)
        return tokens, code
