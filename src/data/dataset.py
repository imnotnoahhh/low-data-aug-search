from __future__ import annotations

import json
import os
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from .augmentations import build_eval_transform, build_weak_augmentation


@dataclass
class LowDataSplitConfig:
    root: str
    download: bool = True
    samples_per_class: int = 100
    train_per_class: int = 90
    val_per_class: int = 10
    seed: int = 42
    subset_indices_path: Optional[str] = None  # 可选，若提供则直接加载

    def validate(self) -> None:
        if self.samples_per_class != self.train_per_class + self.val_per_class:
            raise ValueError("samples_per_class 应等于 train_per_class + val_per_class")
        if self.samples_per_class > 500:
            raise ValueError("CIFAR-100 每类只有 500 张样本")


def _save_indices(indices: Dict[str, List[int]], path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(indices, f, indent=2)


def _load_indices(path: str) -> Dict[str, List[int]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return {k: list(map(int, v)) for k, v in data.items()}


def _stratified_indices(targets: List[int], cfg: LowDataSplitConfig) -> Dict[str, List[int]]:
    random.seed(cfg.seed)
    per_class = {c: [] for c in range(100)}
    for idx, label in enumerate(targets):
        per_class[int(label)].append(idx)

    indices = {"train": [], "val": []}
    for cls, idxs in per_class.items():
        if len(idxs) < cfg.samples_per_class:
            raise ValueError(f"类别 {cls} 样本不足 {cfg.samples_per_class} 张")
        random.shuffle(idxs)
        selected = idxs[: cfg.samples_per_class]
        indices["train"].extend(selected[: cfg.train_per_class])
        indices["val"].extend(selected[cfg.train_per_class : cfg.samples_per_class])
    return indices


def prepare_low_data_split(cfg: LowDataSplitConfig) -> Dict[str, List[int]]:
    """生成或加载低数据划分索引."""

    cfg.validate()
    if cfg.subset_indices_path and os.path.exists(cfg.subset_indices_path):
        return _load_indices(cfg.subset_indices_path)

    dataset = datasets.CIFAR100(
        root=cfg.root,
        train=True,
        download=cfg.download,
        transform=None,
    )
    indices = _stratified_indices(dataset.targets, cfg)

    if cfg.subset_indices_path:
        _save_indices(indices, cfg.subset_indices_path)
    return indices


@dataclass
class DataModuleConfig:
    root: str
    batch_size: int = 256
    num_workers: int = 8
    pin_memory: bool = True
    drop_last: bool = True
    split: LowDataSplitConfig = field(
        default_factory=lambda: LowDataSplitConfig(root="data")
    )


class CIFAR100DataModule:
    """封装 train / val / test DataLoader 构建逻辑."""

    def __init__(
        self,
        config: DataModuleConfig,
        train_transform=None,
        val_transform=None,
        test_transform=None,
    ) -> None:
        self.cfg = config
        self.train_transform = train_transform or build_weak_augmentation()
        self.val_transform = val_transform or build_eval_transform()
        self.test_transform = test_transform or build_eval_transform()

        self.indices = prepare_low_data_split(config.split)
        self._train_set = None
        self._val_set = None
        self._test_set = None

    def _build_dataset(self, train: bool, transform) -> datasets.CIFAR100:
        return datasets.CIFAR100(
            root=self.cfg.root,
            train=train,
            download=True,
            transform=transform,
        )

    def setup(self) -> None:
        train_base = self._build_dataset(train=True, transform=self.train_transform)
        val_base = self._build_dataset(train=True, transform=self.val_transform)
        test_base = self._build_dataset(train=False, transform=self.test_transform)

        self._train_set = Subset(train_base, self.indices["train"])
        self._val_set = Subset(val_base, self.indices["val"])
        self._test_set = test_base

    def train_dataloader(self) -> DataLoader:
        if self._train_set is None:
            self.setup()
        return DataLoader(
            self._train_set,
            batch_size=self.cfg.batch_size,
            shuffle=True,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
            drop_last=self.cfg.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        if self._val_set is None:
            self.setup()
        return DataLoader(
            self._val_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def test_dataloader(self) -> DataLoader:
        if self._test_set is None:
            self.setup()
        return DataLoader(
            self._test_set,
            batch_size=self.cfg.batch_size,
            shuffle=False,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

