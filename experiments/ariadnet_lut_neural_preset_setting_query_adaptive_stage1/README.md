# Ariadne Query-Adaptive Stage 1

New experiment directory:

```text
experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/
```

It reuses the existing experiment's **dataset code and base config at runtime**, so COCO paths, 308 Neural
Preset LUTs, resize protocol, batch/accumulation, optimizer, scheduler, epochs, fixed giraffe/snow pair, and
validation choices stay matched automatically.

## What changed

The old `Progressive3DLUTGeneratorV2` constructs dense high-channel feature volumes at 8^3, 16^3, and 32^3.
This version keeps dense features only at 8^3. At 16^3 and 32^3 it:

1. computes the image's exact trilinear LUT query field;
2. selects the requested cumulative query mass + r1 safety shell;
3. packs those vertices into color tokens;
4. predicts only a residual RGB displacement on those tokens;
5. scatters a **3-channel** sparse correction onto a trilinearly upsampled coarse LUT.

Thus we still materialize a 3-channel LUT for stock `grid_sample`, but we no longer run 96/48-channel Conv3D
blocks over the mostly empty 16^3/32^3 color cube. This is already a real decoder-compute reduction without a
custom CUDA sparse lookup kernel.

## First run: sanity

```bash
cd /home/minimo/Project/AriadneLUT
bash experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/launch_sanity_gpu2.sh
```

Check that:

- loss decreases;
- `norm_active_32` and `style_active_32` are far below 1;
- no NaN / OOM;
- validation images remain sensible.

## Full fair run from scratch

```bash
bash experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/launch_gpu2.sh
```

This is the clean architecture comparison to the existing dense Stage 1.

## Optional fast initialization from the existing dense checkpoint

```bash
bash experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/launch_from_dense_init_gpu2.sh
```

Only name-and-shape-compatible tensors are transplanted. This naturally includes the encoders and the identical
8^3 condition/seed/coarse blocks; dense 16^3/32^3 decoder tensors are not loaded.

## Benchmark after training

```bash
CUDA_VISIBLE_DEVICES=2 \
/home/minimo/miniconda3/envs/DavinciLUT/bin/python \
  experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/benchmark.py \
  --dense-checkpoint experiments/ariadnet_lut_neural_preset_setting/runs/main_gpu2_b24/checkpoints/best.pth \
  --qa-checkpoint experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/runs/main_gpu2_b24/checkpoints/best.pth \
  --output experiments/ariadnet_lut_neural_preset_setting_query_adaptive_stage1/runs/benchmark_dense_vs_qa.json
```

The benchmark reports validation PSNR, parameter count, actual CUDA latency, peak allocated VRAM, and active
refinement fractions. Active fraction is **not** presented as a fake FLOPs/speedup number.

## Conservative defaults

Normalization uses 99.5% mass + r1 because raw/LUT-graded RGB is less sparse. Styling uses 99% + r1 because
canonical query sparsity was measured around 1.7% at K=32. These are configurable in `config.yaml` and should
later be swept for a quality-compute curve (95 / 97.5 / 99 / 99.5 / 100%).
