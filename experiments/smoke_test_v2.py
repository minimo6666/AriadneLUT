from __future__ import annotations

from types import SimpleNamespace

import torch

from losses.stage1_loss_v2 import Stage1LossV2
from models.ariadne_lut_v2 import AriadneLUTV2


def _model_cfg():
    return SimpleNamespace(
        encoder_size=32,
        patch_size=8,
        hidden_dim=64,
        depth=2,
        num_heads=4,
        mlp_ratio=2.0,
        dropout=0.0,
        normalization_dim=32,
        style_dim=32,
        condition_dim=96,
        lut_channels=32,
        normalization_lut_residual_scale_8=0.20,
        normalization_lut_residual_scale_16=0.15,
        normalization_lut_residual_scale_32=0.10,
        style_lut_residual_scale_8=0.20,
        style_lut_residual_scale_16=0.15,
        style_lut_residual_scale_32=0.10,
    )


def _loss_cfg():
    weights = SimpleNamespace(
        charbonnier=1.0,
        mse=5.0,
        chroma=0.5,
        low_frequency=0.5,
        gradient=0.05,
        canonical_consistency=5.0,
        canonical_margin=0.01,
        normalization_lut_smoothness=0.02,
        normalization_lut_monotonicity=0.05,
        normalization_lut_range=0.02,
        style_lut_smoothness=0.02,
        style_lut_monotonicity=0.05,
        style_lut_range=0.02,
    )
    return SimpleNamespace(loss=weights)


def main() -> None:
    torch.manual_seed(7)

    model = AriadneLUTV2(
        _model_cfg()
    )

    image_a = torch.rand(
        1,
        3,
        32,
        32,
    )
    image_b = torch.rand(
        1,
        3,
        32,
        32,
    )

    pair_result = model.forward_pair(
        image_a,
        image_b,
    )

    assert pair_result["output_ab"].shape == (
        1,
        3,
        32,
        32,
    )
    assert pair_result[
        "state_a"
    ]["normalization_lut"].shape == (
        1,
        3,
        32,
        32,
        32,
    )
    assert pair_result[
        "state_b"
    ]["style_lut"].shape == (
        1,
        3,
        32,
        32,
        32,
    )

    criterion = Stage1LossV2(
        _loss_cfg()
    )
    losses = criterion(
        pair_result=pair_result,
        image_a=image_a,
        image_b=image_b,
    )

    losses["total"].backward()

    style = torch.rand(
        1,
        3,
        32,
        32,
    )
    content_1 = torch.rand(
        1,
        3,
        32,
        32,
    )
    content_2 = torch.rand(
        1,
        3,
        32,
        32,
    )

    with torch.no_grad():
        result_1 = model(
            content_1,
            style,
        )
        result_2 = model(
            content_2,
            style,
        )

    # The same style image must generate exactly the same Style LUT,
    # regardless of which content image it will later be applied to.
    assert torch.equal(
        result_1["style_lut"],
        result_2["style_lut"],
    )

    assert result_1["output"].shape == (
        1,
        3,
        32,
        32,
    )

    print(
        "Stage 1 V2 pair forward passed:",
        tuple(
            pair_result[
                "output_ab"
            ].shape
        ),
    )
    print(
        "Stage 1 V2 backward passed:",
        float(
            losses["total"].detach()
        ),
    )
    print(
        "Reusable Style LUT independence passed:",
        tuple(
            result_1["style_lut"].shape
        ),
    )
    print("Stage 1 V2 smoke test passed.")


if __name__ == "__main__":
    main()
