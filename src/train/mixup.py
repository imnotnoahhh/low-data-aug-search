from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn.functional as F


@dataclass
class MixupConfig:
    alpha: float = 0.2
    enabled: bool = True


def mixup_batch(
    inputs: torch.Tensor,
    targets: torch.Tensor,
    cfg: MixupConfig,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float]:
    if not cfg.enabled or cfg.alpha <= 0.0:
        return inputs, targets, targets, 1.0

    beta_dist = torch.distributions.Beta(cfg.alpha, cfg.alpha)
    lam = float(beta_dist.sample().item())
    batch_size = inputs.size(0)
    index = torch.randperm(batch_size, device=inputs.device)
    mixed_inputs = lam * inputs + (1 - lam) * inputs[index, :]
    targets_shuffled = targets[index]
    return mixed_inputs, targets, targets_shuffled, lam


def mixup_criterion(
    loss_fn,
    preds: torch.Tensor,
    targets_a: torch.Tensor,
    targets_b: torch.Tensor,
    lam: float,
) -> torch.Tensor:
    return lam * loss_fn(preds, targets_a) + (1 - lam) * loss_fn(preds, targets_b)

