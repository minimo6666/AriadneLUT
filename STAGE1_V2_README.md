# AriadneLUT Stage 1 V2

This patch adds a new Stage-1 implementation without replacing the original
Stage-1 files.

## V2 model

The V2 model explicitly separates two independent LUT branches:

```text
raw content C
    -> Normalization Encoder
    -> normalization code d_C
    -> Normalization LUT L_N^C
    -> canonical content Z_C

style image S
    -> Style Encoder
    -> style code r_S
    -> reusable Style LUT L_S^S

canonical content Z_C + reusable Style LUT L_S^S
    -> stylized output O
```

The Style LUT branch cannot see the content image, normalization code,
Normalization LUT, or canonical image.

## Stage-1 supervision

For one source image X, two independent grades are sampled:

```text
X_a = P_a(X)
X_b = P_b(X)
```

The model learns:

```text
X_a -> L_N^a -> Z_a
X_b -> L_N^b -> Z_b
Z_a approximately equals Z_b within a configured margin

Z_a + L_S^b -> X_b
Z_b + L_S^a -> X_a
```

No normalization-code consistency is imposed because different source grades
need different normalization operations.

## Add files without overwriting Stage 1

From the AriadneLUT project root:

```bash
unzip -o AriadneLUT_stage1_v2_patch.zip
```

Every Python file added by this patch contains `v2` in its filename. The
original Stage-1 files remain unchanged.

## Smoke test

```bash
conda activate DavinciLUT
cd ~/Project/AriadneLUT

python -m experiments.smoke_test_v2
```

## Short-run test

Temporarily set the following values in `configs/stage1_v2.yaml`:

```yaml
data:
  batch_size: 2
  num_workers: 0
  steps_per_epoch: 100
  val_steps: 20

train:
  epochs: 1
  image_every: 20
```

Then run:

```bash
CUDA_VISIBLE_DEVICES=5 python train_v2.py --config configs/stage1_v2.yaml
```

## Formal training

Restore the intended settings and run:

```bash
python train_v2.py --config configs/stage1_v2.yaml
```

Checkpoints:

```text
experiments/ariadne_stage1_v2/checkpoints/best.pth
experiments/ariadne_stage1_v2/checkpoints/latest.pth
```

## Validation images

Each epoch saves three separate images:

```text
val_stage1_v2_reconstruction_epoch_XXXX.png
```

Columns:

```text
content
synthetic_style_A_on_same_content
synthetic_style_B_on_same_content
style_A_predict_style_B
ground_truth_style_B
style_B_predict_style_A
ground_truth_style_A
```

```text
val_stage1_v2_canonical_epoch_XXXX.png
```

Columns:

```text
synthetic_style_A
canonical_A
synthetic_style_B
canonical_B
```

```text
val_stage1_v2_arbitrary_transfer_epoch_XXXX.png
```

Columns:

```text
content
style
stylized
```

For the arbitrary-transfer visualization, `content` and `style` are rolled
within the validation batch so they come from different source images. This
view has no GT and is used only to inspect cross-content transfer behavior.

## Inference

```bash
python inference_v2.py \
  --config configs/stage1_v2.yaml \
  --checkpoint experiments/ariadne_stage1_v2/checkpoints/best.pth \
  --content /path/to/content.jpg \
  --style /path/to/style.jpg \
  --output outputs/stylized_v2.png \
  --canonical-output outputs/canonical_v2.png
```
