from __future__ import annotations

from types import SimpleNamespace
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from query_field import exact_query_mass, support_from_mass
from query_adaptive_lut import QueryAdaptiveHierarchicalLUTGenerator


def fake_cfg():
    return SimpleNamespace(lut_channels=16, condition_dim=32)


def qa_cfg():
    return {
        "max_query_pixels": 1024,
        "token_hidden_dim": 48,
        "token_depth": 1,
        "coord_fourier_frequencies": 2,
        "shell_decay": 0.35,
        "local_smoothing_blend": 0.1,
        "mass_fraction_16": 0.99,
        "mass_fraction_32": 0.99,
        "dilation_16": 1,
        "dilation_32": 1,
    }


def test_query_mass_normalizes_and_is_sparse_for_constant_color():
    image = torch.full((2, 3, 32, 32), 0.5)
    mass = exact_query_mass(image, 32, max_pixels=1024)
    assert torch.allclose(mass.sum(1), torch.ones(2), atol=1e-5)
    support = support_from_mass(mass, 32, mass_fraction=0.99, dilation=1, shell_decay=0.35)
    assert float(support.active_ratio.max()) < 0.01


def test_generator_shapes_and_gradients_without_evidence():
    torch.manual_seed(0)
    gen = QueryAdaptiveHierarchicalLUTGenerator(
        code_dim=8,
        base_cfg=fake_cfg(),
        qa_cfg=qa_cfg(),
        residual_scale_8=0.3,
        residual_scale_16=0.2,
        residual_scale_32=0.12,
        use_evidence=False,
    )
    code = torch.randn(2, 8, requires_grad=True)
    image = torch.rand(2, 3, 32, 32) * 0.2 + 0.35
    lut, pyr = gen(code, query_image=image)
    assert lut.shape == (2, 3, 32, 32, 32)
    assert pyr["level_16"].delta.shape == (2, 3, 16, 16, 16)
    assert pyr["level_32"].token_count < 2 * (32 ** 3)
    loss = lut.mean()
    loss.backward()
    assert gen.to_delta_8.weight.grad is not None
    assert gen.refine_32.to_delta.weight.grad is not None


def test_generator_with_style_evidence():
    torch.manual_seed(1)
    gen = QueryAdaptiveHierarchicalLUTGenerator(
        code_dim=8,
        base_cfg=fake_cfg(),
        qa_cfg=qa_cfg(),
        residual_scale_8=0.3,
        residual_scale_16=0.2,
        residual_scale_32=0.12,
        use_evidence=True,
    )
    code = torch.randn(1, 8)
    query = torch.rand(1, 3, 24, 24) * 0.15 + 0.4
    evidence = torch.rand(1, 3, 24, 24) * 0.15 + 0.45
    lut, pyr = gen(code, query_image=query, evidence_image=evidence)
    assert lut.shape == (1, 3, 32, 32, 32)
    assert pyr["level_32"].evidence_mass is not None
