from __future__ import annotations

from torchvision import transforms as T


def make_transform(image_size: int, train: bool):
    if train:
        return T.Compose(
            [
                T.Resize(image_size, interpolation=T.InterpolationMode.BILINEAR),
                T.CenterCrop((image_size, int(round(image_size * 1.25)))),
                T.RandomCrop(image_size),
                T.RandomHorizontalFlip(p=0.5),
                T.ToTensor(),
            ]
        )
    return T.Compose(
        [
            T.Resize(image_size, interpolation=T.InterpolationMode.BICUBIC),
            T.CenterCrop(image_size),
            T.ToTensor(),
        ]
    )
