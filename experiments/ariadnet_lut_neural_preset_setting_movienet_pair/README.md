# AriadneLUT — MovieNet Pair Stage 2

This folder implements the algorithmic/training side of the proposed cross-content Stage-2. The MovieNet pair loader is intentionally left to Codex because it can inspect the newly generated `qwen_proxy_100k_v1` manifests directly.

## Training target

For two different-content movie frames `X` and `Y` selected as having the same final movie color style:

```text
C_Y = corruption(Y),  S_X = X,  GT_Y = Y
C_X = corruption(X),  S_Y = Y,  GT_X = X
```

Train both directions:

```text
corruption(Y) + X -> Y
corruption(X) + Y -> X
```

The GT is the real graded movie frame, not a Qwen-generated image and not a same-LUT pseudo target.

## Frozen Stage-1

Stage-1 is loaded from:

```text
experiments/ariadnet_lut_neural_preset_setting/
  runs/main_gpu2_b24/checkpoints/best.pth
```

Its architecture already separates:

```text
C -> normalization branch -> Z_C
S -> style branch -> L_base(S)
```

All Stage-1 parameters are frozen in the first Stage-2 experiment.

## New trainable residual branch

The new branch predicts a **content-dependent residual Style LUT**:

```text
Z_C ---------------------------> CanonicalColorEncoder ----┐
                                                           |
S -> frozen StyleEncoder -> r_S -> frozen L_base(S) -------+--> residual 3D refiner
                                                           |
Q_C = QueryField(Z_C) -------------------------------------+
Q_S = QueryField(Z_S) -------------------------------------+

Delta L_{C,S} = refiner(L_base, Q_C, Q_S, r_S, code(Z_C))
L_{C,S}       = L_base + alpha * Delta L_{C,S}
O             = L_{C,S}(Z_C)
```

`Q_C`/`Q_S` are exact trilinear query-mass volumes in the same `[B,G,R]` 32^3 lattice used by `apply_lut_v2`. This makes the content condition directly relevant to the LUT coordinates that the current canonical image actually uses.

## Smooth Stage-1 -> Stage-2 transition

Two safeguards make the fine-tune start exactly from Stage-1:

1. the residual head is **zero initialized**, so `Delta L = 0` at step 0;
2. `alpha` is ramped from 0 to 1 during the first `residual_ramp_steps` optimizer steps.

Therefore the initial forward is Stage-1 behavior and the new module can only learn corrections gradually.

## Loss

Exact movie GT allows direct supervision:

```text
L = 1.00 Charbonnier(output, GT)
  + 0.20 LPIPS(output, GT)
  + 0.50 low-frequency L1(output, GT)
  + 0.50 RGB mean/std matching(output, GT)
  + 0.01 ||Delta L||^2
  + 0.05 smoothness(Delta L)
  + 0.05 curvature(L_final)
  + 1.00 range_penalty(L_final)
```

The first four terms supervise the desired movie appearance. The remaining terms keep the learned correction small and smooth and discourage LUT values outside `[0,1]`.

## Why the base Style LUT remains frozen initially

The first experiment is deliberately diagnostic. If a small content-conditioned residual can improve the frozen Stage-1 output on different-content movie pairs, we isolate the failure to cross-content LUT adaptation rather than destabilizing the learned canonical space. Only after this succeeds should we consider a second run that unfreezes the Style Encoder / base LUT generator at a much smaller learning rate.

## Dataset hand-off

See `DATASET_CONTRACT.md`. Codex only needs to implement:

```text
movienet_pair_dataset.py
```

against:

```text
/mnt/data/0/mohao/data/MovieNet/AriadnetLUT/qwen_proxy_100k_v1
```

## Install LPIPS

In the existing environment if it is not already installed:

```bash
pip install lpips
```

## Launch (after Codex implements the dataset file)

```bash
bash experiments/ariadnet_lut_neural_preset_setting_movienet_pair/launch_gpu2.sh
```

## What to watch first

The log explicitly reports:

- `baseline_psnr`: frozen Stage-1 on the exact same batch;
- `refined_psnr`: Stage-2 output;
- `psnr_gain`: Stage-2 minus Stage-1;
- `delta_abs_mean`;
- `final_lut_oob`.

Validation grids show:

```text
corrupted content | movie reference | frozen Z_C | Stage1 baseline | Stage2 refined | movie GT
```

The first success criterion is not simply lower training loss; it is consistent positive held-out `psnr_gain` plus visibly cleaner cross-content color restoration without LUT artifacts.
