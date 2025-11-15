from __future__ import annotations

from pathlib import Path
from typing import Iterable, Sequence, Tuple

import torch
from PIL import Image
from torchvision.utils import make_grid, save_image


def ensure_dir(path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def save_side_by_side(
    originals: Sequence[torch.Tensor],
    augmented: Sequence[torch.Tensor],
    filepath: str,
    nrow: int = 4,
) -> None:
    """将若干原图与增强图并排保存."""

    if len(originals) != len(augmented):
        raise ValueError("originals 与 augmented 数量需一致")

    paired: Sequence[torch.Tensor] = []
    for orig, aug in zip(originals, augmented):
        paired += [orig, aug]

    grid = make_grid(torch.stack(paired), nrow=nrow * 2, normalize=True, value_range=(0, 1))
    ensure_dir(filepath)
    save_image(grid, filepath)

