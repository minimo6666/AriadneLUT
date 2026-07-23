from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def build_2d_sincos_position_embedding(grid_size: int, dim: int) -> torch.Tensor:
    if dim % 4 != 0:
        raise ValueError("hidden_dim must be divisible by 4 for 2D sin/cos embedding")
    y, x = torch.meshgrid(
        torch.arange(grid_size, dtype=torch.float32),
        torch.arange(grid_size, dtype=torch.float32),
        indexing="ij",
    )
    omega = torch.arange(dim // 4, dtype=torch.float32)
    omega = 1.0 / (10000 ** (omega / max(dim // 4 - 1, 1)))
    x = x.reshape(-1, 1) * omega.reshape(1, -1)
    y = y.reshape(-1, 1) * omega.reshape(1, -1)
    pos = torch.cat([x.sin(), x.cos(), y.sin(), y.cos()], dim=1)
    return pos.unsqueeze(0)


class PatchEmbed(nn.Module):
    def __init__(self, image_size: int, patch_size: int, dim: int):
        super().__init__()
        if image_size % patch_size != 0:
            raise ValueError("image_size must be divisible by patch_size")
        self.grid_size = image_size // patch_size
        self.proj = nn.Conv2d(3, dim, kernel_size=patch_size, stride=patch_size)
        pos = build_2d_sincos_position_embedding(self.grid_size, dim)
        self.register_buffer("position", pos, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        tokens = self.proj(x).flatten(2).transpose(1, 2)
        return tokens + self.position.to(device=tokens.device, dtype=tokens.dtype)


class SelfAttentionBlock(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        hidden = int(dim * mlp_ratio)
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
        )
        self.scale_attn = nn.Parameter(torch.full((dim,), 1e-4))
        self.scale_ff = nn.Parameter(torch.full((dim,), 1e-4))
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        qkv = self.qkv(self.norm1(x)).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        q, k, v = [item.transpose(1, 2) for item in (q, k, v)]
        attn = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        attn = attn.transpose(1, 2).reshape(b, n, d)
        x = x + self.scale_attn * self.out(attn)
        x = x + self.scale_ff * self.ff(self.norm2(x))
        return x


class MMDiTJointBlock(nn.Module):
    """
    Lightweight MMDiT-style block.

    Content and style have modality-specific projections and MLPs, while the
    attention keys/queries/values are concatenated for bidirectional joint
    attention. This keeps the central MMDiT idea without SD3-scale width.
    """

    def __init__(self, dim: int, heads: int, mlp_ratio: float, dropout: float):
        super().__init__()
        if dim % heads != 0:
            raise ValueError("dim must be divisible by heads")
        self.heads = heads
        self.head_dim = dim // heads
        self.dropout = float(dropout)

        self.c_norm1 = nn.LayerNorm(dim)
        self.s_norm1 = nn.LayerNorm(dim)
        self.c_qkv = nn.Linear(dim, dim * 3)
        self.s_qkv = nn.Linear(dim, dim * 3)
        self.c_out = nn.Linear(dim, dim)
        self.s_out = nn.Linear(dim, dim)

        hidden = int(dim * mlp_ratio)
        self.c_norm2 = nn.LayerNorm(dim)
        self.s_norm2 = nn.LayerNorm(dim)
        self.c_ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))
        self.s_ff = nn.Sequential(nn.Linear(dim, hidden), nn.GELU(), nn.Linear(hidden, dim))

        self.c_attn_scale = nn.Parameter(torch.full((dim,), 1e-4))
        self.s_attn_scale = nn.Parameter(torch.full((dim,), 1e-4))
        self.c_ff_scale = nn.Parameter(torch.full((dim,), 1e-4))
        self.s_ff_scale = nn.Parameter(torch.full((dim,), 1e-4))

    def _qkv(self, layer, normed):
        b, n, d = normed.shape
        qkv = layer(normed).view(b, n, 3, self.heads, self.head_dim)
        q, k, v = qkv.unbind(dim=2)
        return [item.transpose(1, 2) for item in (q, k, v)]

    def forward(self, content: torch.Tensor, style: torch.Tensor):
        c_q, c_k, c_v = self._qkv(self.c_qkv, self.c_norm1(content))
        s_q, s_k, s_v = self._qkv(self.s_qkv, self.s_norm1(style))
        q = torch.cat([c_q, s_q], dim=2)
        k = torch.cat([c_k, s_k], dim=2)
        v = torch.cat([c_v, s_v], dim=2)
        joint = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.dropout if self.training else 0.0
        )
        c_len = content.shape[1]
        c_joint = joint[:, :, :c_len].transpose(1, 2).reshape_as(content)
        s_joint = joint[:, :, c_len:].transpose(1, 2).reshape_as(style)
        content = content + self.c_attn_scale * self.c_out(c_joint)
        style = style + self.s_attn_scale * self.s_out(s_joint)
        content = content + self.c_ff_scale * self.c_ff(self.c_norm2(content))
        style = style + self.s_ff_scale * self.s_ff(self.s_norm2(style))
        return content, style
