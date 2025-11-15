from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets

from src.data.augmentations import TransformSpec, build_aug_chain, build_eval_transform
from src.data.dataset import LowDataSplitConfig, prepare_low_data_split
from src.models.resnet import build_resnet18
from src.train.engine import (
    OptimizerConfig,
    SchedulerConfig,
    TrainingConfig,
    TrainingSession,
    build_optimizer,
    build_scheduler,
)
from src.train.mixup import MixupConfig


def load_stage_b_policies(path: str) -> List[List[TransformSpec]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    policies: List[List[TransformSpec]] = []
    for entry in data:
        specs = [
            TransformSpec(name=s["name"], prob=s["prob"], params=s["params"])
            for s in entry["specs"]
        ]
        policies.append(specs)
    return policies


class PolicyBatchCollator:
    """自定义 collate：每个 batch 等概率采样一条策略，对整批样本一起增强。"""

    def __init__(self, policy_transforms: Sequence):
        self.policy_transforms = list(policy_transforms)

    def __call__(self, batch):
        policy = random.choice(self.policy_transforms)
        images = torch.stack([policy(img) for img, _ in batch])
        targets = torch.tensor([label for _, label in batch], dtype=torch.long)
        return images, targets


def paired_t_test(baseline: Sequence[float], candidate: Sequence[float]) -> float:
    if len(baseline) != len(candidate):
        raise ValueError("paired t-test 需要相同数量的样本")
    diffs = [c - b for c, b in zip(candidate, baseline)]
    n = len(diffs)
    if n < 2:
        return 1.0
    mean_diff = sum(diffs) / n
    variance = sum((d - mean_diff) ** 2 for d in diffs) / (n - 1)
    if variance == 0:
        return 1.0
    std_diff = math.sqrt(variance)
    t_stat = mean_diff / (std_diff / math.sqrt(n))
    dist = torch.distributions.StudentT(df=n - 1)
    p = 2 * (1 - dist.cdf(torch.tensor(abs(t_stat))).item())
    return float(p)


@dataclass
class StageCConfig:
    policy_path: str
    data_root: str
    split_config: LowDataSplitConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    mixup: MixupConfig
    seeds: Sequence[int] = (0, 1, 2, 3, 4)
    delta_threshold: float = 0.3
    p_threshold: float = 0.05
    max_policies: int = 4
    output_dir: str = "artifacts/stage_c"


class StageCPolicyEnsembler:
    def __init__(self, cfg: StageCConfig) -> None:
        self.cfg = cfg
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.policies = load_stage_b_policies(cfg.policy_path)
        self.split_indices = prepare_low_data_split(cfg.split_config)
        self.val_transform = build_eval_transform()

    def _build_dataloaders(self, policy_specs: List[List[TransformSpec]]):
        dataset = datasets.CIFAR100(
            root=self.cfg.data_root,
            train=True,
            download=True,
            transform=None,
        )
        val_dataset = datasets.CIFAR100(
            root=self.cfg.data_root,
            train=True,
            download=True,
            transform=self.val_transform,
        )
        train_subset = Subset(dataset, self.split_indices["train"])
        val_subset = Subset(val_dataset, self.split_indices["val"])

        policy_transforms = [build_aug_chain(specs) for specs in policy_specs]

        collate = PolicyBatchCollator(policy_transforms)
        train_loader = DataLoader(
            train_subset,
            batch_size=256,
            shuffle=True,
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            collate_fn=collate,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=256,
            shuffle=False,
            num_workers=8,
            pin_memory=True,
        )
        return train_loader, val_loader

    def _run_training(
        self,
        policy_specs: List[List[TransformSpec]],
        seed: int,
    ) -> Dict[str, float]:
        torch.manual_seed(seed)
        np.random.seed(seed)
        random.seed(seed)

        train_loader, val_loader = self._build_dataloaders(policy_specs)

        model = build_resnet18()
        optimizer = build_optimizer(model, self.cfg.optimizer)
        scheduler = build_scheduler(optimizer, self.cfg.scheduler)
        criterion = nn.CrossEntropyLoss()

        session = TrainingSession(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            mixup_cfg=self.cfg.mixup,
            train_cfg=self.cfg.training,
            sched_cfg=self.cfg.scheduler,
            device=self.device,
        )
        session.train_until(self.cfg.training.epochs)
        return session.history[-1]

    def evaluate_policy_set(self, policy_indices: Sequence[int]) -> Dict[str, List[float]]:
        policy_specs = [self.policies[idx] for idx in policy_indices]
        metrics = {"val_top1": [], "val_top5": []}
        for seed in self.cfg.seeds:
            result = self._run_training(policy_specs, seed)
            metrics["val_top1"].append(result["val_top1"])
            metrics["val_top5"].append(result["val_top5"])
        return metrics

    def greedy_select(self) -> Dict:
        selected: List[int] = []
        best_scores: Optional[List[float]] = None
        log: List[Dict] = []

        candidate_order = list(range(len(self.policies)))

        for idx in candidate_order:
            if len(selected) >= self.cfg.max_policies:
                break
            if idx in selected:
                continue
            trial_selection = selected + [idx]
            metrics = self.evaluate_policy_set(trial_selection)
            mean_score = float(np.mean(metrics["val_top1"]))

            if not selected:
                selected = trial_selection
                best_scores = metrics["val_top1"]
                log.append(
                    {
                        "selected": selected.copy(),
                        "mean_top1": mean_score,
                        "delta": None,
                        "p_value": None,
                    }
                )
                continue

            delta = mean_score - float(np.mean(best_scores or []))
            p_value = paired_t_test(best_scores or [], metrics["val_top1"])

            log.append(
                {
                    "candidate": idx,
                    "selection": trial_selection.copy(),
                    "mean_top1": mean_score,
                    "delta": delta,
                    "p_value": p_value,
                }
            )

            if delta >= self.cfg.delta_threshold and p_value < self.cfg.p_threshold:
                selected = trial_selection
                best_scores = metrics["val_top1"]
            else:
                # 不满足增益要求，停止
                break

        output = {
            "selected_indices": selected,
            "log": log,
        }
        out_path = Path(self.cfg.output_dir) / "stage_c_selection.json"
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(output, f, indent=2)
        return output

