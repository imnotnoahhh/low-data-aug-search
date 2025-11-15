from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Dict, List, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset
from src.data.augmentations import CIFAR_MEAN, CIFAR_STD
from src.train.engine import accuracy

CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression",
]


@dataclass
class CIFAR100CConfig:
    root: str
    batch_size: int = 256
    num_workers: int = 4


class CIFAR100CEvaluator:
    def __init__(self, cfg: CIFAR100CConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _build_loader(
        self,
        images: np.ndarray,
        labels: np.ndarray,
    ) -> DataLoader:
        tensor = torch.from_numpy(images).permute(0, 3, 1, 2).float() / 255.0
        mean = torch.tensor(CIFAR_MEAN).view(1, 3, 1, 1)
        std = torch.tensor(CIFAR_STD).view(1, 3, 1, 1)
        tensor = (tensor - mean) / std
        targets = torch.from_numpy(labels).long()
        dataset = TensorDataset(tensor, targets)
        return DataLoader(
            dataset,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=True,
        )

    def evaluate(
        self,
        model: nn.Module,
        severities: Sequence[int] = (1, 2, 3, 4, 5),
    ) -> Dict[str, float]:
        model = model.to(self.device)
        model.eval()
        corruption_scores: Dict[str, float] = {}

        labels = np.load(os.path.join(self.cfg.root, "labels.npy"))

        for corruption in CORRUPTIONS:
            acc_sum = 0.0
            count = 0
            data = np.load(os.path.join(self.cfg.root, f"{corruption}.npy"))
            for severity in severities:
                start = (severity - 1) * 10000
                end = severity * 10000
                images = data[start:end]
                loader = self._build_loader(images, labels[start:end])
                acc = self._run_loader(model, loader)
                acc_sum += acc
                count += 1
            corruption_scores[corruption] = acc_sum / count

        mca = sum(corruption_scores.values()) / len(corruption_scores)
        corruption_scores["mCA"] = mca
        return corruption_scores

    def _run_loader(self, model: nn.Module, loader: DataLoader) -> float:
        total_top1 = 0.0
        total_samples = 0
        with torch.no_grad():
            for images, targets in loader:
                images = images.to(self.device, non_blocking=True)
                targets = targets.to(self.device, non_blocking=True)
                outputs = model(images)
                top1, _ = accuracy(outputs, targets, topk=(1, 5))
                batch_size = targets.size(0)
                total_top1 += top1 * batch_size
                total_samples += batch_size
        return total_top1 / total_samples

