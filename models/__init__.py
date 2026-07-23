from .ariadne_lut_v2 import AriadneLUTV2


def build_model(cfg):
    return AriadneLUTV2(cfg.model)


__all__ = ["AriadneLUTV2", "build_model"]
