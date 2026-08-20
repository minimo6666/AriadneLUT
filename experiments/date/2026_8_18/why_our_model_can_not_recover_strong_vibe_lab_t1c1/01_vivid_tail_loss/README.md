# 高色度尾部监督

- 实验身份：`vivid_tail_loss`
- 唯一变化：只增加 `lambda_vivid_tail=1.0`，在每张 GT 色度最高 10% 像素匹配 Oklab a/b。
- 其余模型、loss、数据切分、seed、LR、epoch、有效 batch、TV/gamut 与另外两组完全一致。
- 判读：若本组明显改善，说明整图平均目标稀释了少量浓色区域的梯度。

## GitHub 代码范围

- `train/launch_train.sh`：正式训练入口。
- `train/RUN_SPEC.json`：机器可读实验身份与常量。
- `../_shared/train/source/`：模型、dataset、vivid-tail loss 与单卡/DDP trainer 的共享实现。
- `outputs/`、`metrics/`、`samples/`、checkpoint 和日志属于生成结果，不上传 GitHub。

训练依赖仓库中已有的基础配置：

```text
experiments/stage_1/ariadnet_lut_neural_preset_setting/config.yaml
```

运行前请把该配置中的 `data.coco_root` 和 `data.lut_root` 改为当前机器的实际数据路径，或者通过 `--config` 传入自己的配置。

## 训练

```bash
conda activate DavinciLUT
bash train/launch_train.sh
```

指定 GPU、Python 或配置：

```bash
ARIADNE_TRAIN_GPU=4 \
ARIADNE_PYTHON="$(command -v python)" \
bash train/launch_train.sh --config /path/to/config.yaml
```

断点续训：

```bash
bash train/launch_train.sh \
  --resume experiments/date/2026_8_18/why_our_model_can_not_recover_strong_vibe_lab_t1c1/01_vivid_tail_loss/outputs/train/checkpoints/last.pth
```
