from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import VGG16_Weights, vgg16


class VGG16PerceptualLoss(nn.Module):
    """Frozen ImageNet VGG16 feature loss for full reconstructed outputs only.

    The loss uses relu1_2, relu2_2 and relu3_3 features. Inputs are resized to
    `image_size` before ImageNet normalization. Target features are detached.

    This module intentionally affects only the training objective; it is not
    part of inference and therefore does not change slider runtime.
    """

    def __init__(self, image_size: int = 128) -> None:
        super().__init__()
        self.image_size = int(image_size)

        try:
            features = vgg16(weights=VGG16_Weights.IMAGENET1K_V1).features
        except Exception as exc:
            raise RuntimeError(
                "Failed to load pretrained VGG16 weights required by the perceptual loss. "
                "Make sure torchvision can access/cache ImageNet VGG16 weights, or launch "
                "training with --lambda-perceptual 0 to disable this term."
            ) from exc

        self.blocks = nn.ModuleList(
            [
                nn.Sequential(*features[:4]),   # relu1_2
                nn.Sequential(*features[4:9]),  # relu2_2
                nn.Sequential(*features[9:16]), # relu3_3
            ]
        )
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

        mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
        self.register_buffer("mean", mean, persistent=False)
        self.register_buffer("std", std, persistent=False)

    def train(self, mode: bool = True):
        # Keep VGG permanently in eval mode even when the parent criterion is trained.
        super().train(False)
        return self

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        x = x.float().clamp(0.0, 1.0)
        if self.image_size > 0 and (
            x.shape[-2] != self.image_size or x.shape[-1] != self.image_size
        ):
            x = F.interpolate(
                x,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (x - self.mean) / self.std

    def _features(self, x: torch.Tensor):
        outputs = []
        for block in self.blocks:
            x = block(x)
            outputs.append(x)
        return outputs

    def forward(self, prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Do feature extraction in FP32 for numerical stability, even when the outer
        # training loop uses AMP.
        device_type = prediction.device.type
        with torch.autocast(device_type=device_type, enabled=False):
            pred = self._prepare(prediction)
            tgt = self._prepare(target)
            pred_features = self._features(pred)
            with torch.no_grad():
                target_features = self._features(tgt)
            losses = [
                F.l1_loss(p, t)
                for p, t in zip(pred_features, target_features)
            ]
            return sum(losses) / len(losses)
