from __future__ import annotations

import torch


class FilmGradeAugmentor:
    def __init__(self, cfg):
        self.cfg = cfg

    def _uniform(self, low, high, shape, device, dtype):
        return torch.empty(shape, device=device, dtype=dtype).uniform_(low, high)

    def sample(self, batch: int, device, dtype, shared: bool):
        n = 1 if shared else batch
        cfg = self.cfg
        params = {
            "exposure": self._uniform(-cfg.exposure, cfg.exposure, (n, 1, 1, 1), device, dtype),
            "gamma": torch.exp(self._uniform(-cfg.gamma, cfg.gamma, (n, 1, 1, 1), device, dtype)),
            "contrast": 1.0 + self._uniform(-cfg.contrast, cfg.contrast, (n, 1, 1, 1), device, dtype),
            "saturation": 1.0 + self._uniform(-cfg.saturation, cfg.saturation, (n, 1, 1, 1), device, dtype),
            "temperature": self._uniform(-cfg.temperature, cfg.temperature, (n, 1, 1, 1), device, dtype),
            "tint": self._uniform(-cfg.tint, cfg.tint, (n, 1, 1, 1), device, dtype),
            "split_shadow": self._uniform(-cfg.split_tone, cfg.split_tone, (n, 3, 1, 1), device, dtype),
            "split_high": self._uniform(-cfg.split_tone, cfg.split_tone, (n, 3, 1, 1), device, dtype),
            "lift": self._uniform(-cfg.lift, cfg.lift, (n, 1, 1, 1), device, dtype),
            "shoulder": self._uniform(-cfg.shoulder, cfg.shoulder, (n, 1, 1, 1), device, dtype),
        }
        eye = torch.eye(3, device=device, dtype=dtype).unsqueeze(0).expand(n, -1, -1)
        noise = self._uniform(-cfg.channel_mix, cfg.channel_mix, (n, 3, 3), device, dtype)
        params["matrix"] = eye + noise
        if shared and batch > 1:
            params = {key: value.expand(batch, *value.shape[1:]) for key, value in params.items()}
        return params

    def apply(self, image: torch.Tensor, params):
        x = image.clamp(0, 1)
        x = x * torch.pow(torch.tensor(2.0, device=x.device, dtype=x.dtype), params["exposure"])
        x = x.clamp_min(1e-5).pow(params["gamma"])
        x = (x - 0.5) * params["contrast"] + 0.5

        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        x = lum + params["saturation"] * (x - lum)

        temperature = params["temperature"]
        tint = params["tint"]
        scale = torch.cat([1.0 + temperature, 1.0 + tint, 1.0 - temperature], dim=1)
        x = x * scale
        x = torch.einsum("bij,bjhw->bihw", params["matrix"], x)

        lum = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        shadow_weight = (1.0 - lum.clamp(0, 1)).square()
        highlight_weight = lum.clamp(0, 1).square()
        x = x + shadow_weight * params["split_shadow"] + highlight_weight * params["split_high"]

        x = x + params["lift"] * (1.0 - x)
        shoulder = params["shoulder"]
        x = x - shoulder * x.square() * (1.0 - x)
        return x.clamp(0, 1)

    def two_views(self, image: torch.Tensor, shared: bool = True):
        params_a = self.sample(image.shape[0], image.device, image.dtype, shared=shared)
        params_b = self.sample(image.shape[0], image.device, image.dtype, shared=shared)
        return self.apply(image, params_a), self.apply(image, params_b)

    def corrupt(self, image: torch.Tensor):
        params = self.sample(image.shape[0], image.device, image.dtype, shared=False)
        return self.apply(image, params)
