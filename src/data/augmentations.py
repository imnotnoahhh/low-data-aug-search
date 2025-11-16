from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Sequence

import torch
from torchvision import transforms as T
from torchvision.transforms import functional as F


CIFAR_MEAN = (0.5071, 0.4867, 0.4408)
CIFAR_STD = (0.2675, 0.2565, 0.2761)
IMAGE_SIZE = 32


@dataclass
class TransformSpec:
    """描述单个可搜索增强的配置."""

    name: str
    prob: float
    params: Dict[str, Any] = field(default_factory=dict)

    def with_updates(self, **overrides: Any) -> "TransformSpec":
        new_params = dict(self.params)
        new_prob = overrides.pop("prob", self.prob)
        new_params.update(overrides)
        return TransformSpec(name=self.name, prob=new_prob, params=new_params)


def build_no_aug_transform(normalize: bool = True) -> T.Compose:
    ops: List[T.transforms] = [T.ToTensor()]
    if normalize:
        ops.append(T.Normalize(CIFAR_MEAN, CIFAR_STD))
    return T.Compose(ops)


def _wrap_with_probability(transform: T.transforms, prob: float) -> T.transforms:
    if prob >= 1.0:
        return transform
    return T.RandomApply([transform], p=prob)


def _build_single_transform(spec: TransformSpec) -> T.transforms:
    """根据 TransformSpec 构建 torchvision 增强."""

    name = spec.name
    params = dict(spec.params)

    if name == "RandomResizedCrop":
        transform = T.RandomResizedCrop(
            size=IMAGE_SIZE,
            scale=params.get("scale", (0.5, 1.0)),
            ratio=params.get("ratio", (0.75, 1.33)),
        )
    elif name == "RandomCrop":
        transform = T.RandomCrop(
            IMAGE_SIZE,
            padding=params.get("padding", 4),
        )
    elif name == "RandomRotation":
        transform = T.RandomRotation(degrees=params.get("degrees", 15))
    elif name == "RandomPerspective":
        transform = T.RandomPerspective(
            distortion_scale=params.get("distortion_scale", 0.1)
        )
    elif name == "RandomHorizontalFlip":
        transform = T.RandomHorizontalFlip(p=1.0)
    elif name == "ColorJitter":
        transform = T.ColorJitter(
            brightness=params.get("brightness", 0.4),
            contrast=params.get("contrast", 0.4),
            saturation=params.get("saturation", 0.4),
            hue=params.get("hue", 0.1),
        )
    elif name == "RandomGrayscale":
        transform = T.RandomGrayscale(p=1.0)
    elif name == "GaussianBlur":
        transform = T.GaussianBlur(
            kernel_size=params.get("kernel_size", 3),
            sigma=params.get("sigma", (0.1, 2.0)),
        )
    elif name == "GaussianNoise":
        sigma = params.get("sigma", 0.05)

        def _add_noise(img):
            if not isinstance(img, torch.Tensor):
                tensor = F.pil_to_tensor(img).float().div_(255.0)
            else:
                tensor = img.clone()
            noise = torch.randn_like(img) * sigma
            noisy = torch.clamp(tensor + noise, 0.0, 1.0)
            if isinstance(img, torch.Tensor):
                return noisy
            return F.to_pil_image(noisy)

        transform = T.Lambda(_add_noise)
    elif name == "RandomErasing":
        base = T.RandomErasing(
            p=1.0,
            scale=params.get("scale", (0.02, 0.33)),
            ratio=params.get("ratio", (0.3, 3.3)),
            value=params.get("value", "random"),
        )

        def _erase(img):
            if not isinstance(img, torch.Tensor):
                tensor = F.pil_to_tensor(img).float().div_(255.0)
            else:
                tensor = img.clone()
            erased = base(tensor)
            if isinstance(img, torch.Tensor):
                return erased
            return F.to_pil_image(erased)

        transform = T.Lambda(_erase)
    else:
        raise ValueError(f"Unsupported transform: {name}")

    return _wrap_with_probability(transform, spec.prob)


def build_aug_chain(
    specs: Sequence[TransformSpec],
) -> T.Compose:
    """根据若干 TransformSpec 组合出完整 pipeline."""

    ops: List[T.transforms] = []
    for spec in specs:
        ops.append(_build_single_transform(spec))

    ops.append(T.ToTensor())
    ops.append(T.Normalize(CIFAR_MEAN, CIFAR_STD))

    return T.Compose(ops)


def build_eval_transform() -> T.Compose:
    return T.Compose(
        [
            T.ToTensor(),
            T.Normalize(CIFAR_MEAN, CIFAR_STD),
        ]
    )

