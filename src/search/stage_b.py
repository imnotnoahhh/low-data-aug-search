from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn as nn

from src.data.augmentations import TransformSpec, build_aug_chain
from src.data.dataset import CIFAR100DataModule, DataModuleConfig
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
from src.search.asha import ASHAConfig


def load_stage_a_candidates(paths: Sequence[str]) -> Dict[str, List[TransformSpec]]:
    """读取阶段 A 输出的 JSON，构建 transform -> specs 映射。"""

    candidates: Dict[str, List[TransformSpec]] = {}
    for path in paths:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        transform_name = Path(path).stem.replace("_topk", "")
        specs = [
            TransformSpec(name=transform_name, prob=entry["spec"]["prob"], params=entry["spec"]["params"])
            for entry in data
        ]
        candidates[transform_name] = specs
    return candidates


@dataclass
class StageBPolicySamplerConfig:
    candidate_paths: Sequence[str]
    num_policies: int = 150
    min_transforms: int = 1
    max_transforms: int = 3


@dataclass
class StageBConfig:
    sampler: StageBPolicySamplerConfig
    data: DataModuleConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    training: TrainingConfig
    mixup: MixupConfig
    asha: ASHAConfig = field(
        default_factory=lambda: ASHAConfig(
            rung_levels=[30, 60, 120],
            reduction_factor=3,
            metric_key="val_top1",
            maximize=True,
        )
    )
    output_dir: str = "artifacts/stage_b"
    device: Optional[str] = None


class StageBTuner:
    """阶段 B：在阶段 A 的候选子范围内联合调参并用 ASHA 搜索 150 组策略."""

    def __init__(self, cfg: StageBConfig) -> None:
        self.cfg = cfg
        self.cfg.asha.validate()
        self.device = torch.device(
            cfg.device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self.candidates = load_stage_a_candidates(cfg.sampler.candidate_paths)

    def _sample_policies(self) -> List[List[TransformSpec]]:
        transforms = list(self.candidates.keys())
        rng = random.Random(0)
        policies: List[List[TransformSpec]] = []
        for i in range(self.cfg.sampler.num_policies):
            k = rng.randint(
                self.cfg.sampler.min_transforms,
                min(self.cfg.sampler.max_transforms, len(transforms)),
            )
            chosen = rng.sample(transforms, k)
            policy_specs: List[TransformSpec] = []
            for name in chosen:
                specs = self.candidates[name]
                spec = rng.choice(specs)
                policy_specs.append(spec)
            policies.append(policy_specs)
        return policies

    def _build_session(self, specs: List[TransformSpec]) -> TrainingSession:
        train_transform = build_aug_chain(specs)
        data_module = CIFAR100DataModule(
            config=self.cfg.data,
            train_transform=train_transform,
        )
        train_loader = data_module.train_dataloader()
        val_loader = data_module.val_dataloader()

        model = build_resnet18()
        optimizer = build_optimizer(model, self.cfg.optimizer)
        scheduler = build_scheduler(optimizer, self.cfg.scheduler)
        criterion = nn.CrossEntropyLoss()

        return TrainingSession(
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

    def run(self) -> List[Dict]:
        policies = self._sample_policies()
        results = self._asha_loop(policies)
        self._export(results)
        return results

    def _asha_loop(self, policies: List[List[TransformSpec]]) -> List[Dict]:
        rung_levels = self.cfg.asha.rung_levels
        reduction = self.cfg.asha.reduction_factor
        metric_key = self.cfg.asha.metric_key
        maximize = self.cfg.asha.maximize

        trial_states = [
            {
                "id": f"policy_{idx}",
                "specs": specs,
                "session": self._build_session(specs),
                "alive": True,
                "results": {},
            }
            for idx, specs in enumerate(policies)
        ]
        all_records: List[Dict] = []
        active = trial_states

        for rung_idx, target_epoch in enumerate(rung_levels):
            for state in active:
                if not state["alive"]:
                    continue
                metrics = state["session"].train_until(target_epoch)
                record = {
                    "trial_id": state["id"],
                    "epoch": target_epoch,
                    "specs": state["specs"],
                    "metrics": metrics,
                }
                state["results"][target_epoch] = metrics
                all_records.append(record)
            if rung_idx == len(rung_levels) - 1:
                break
            num_keep = max(1, math.ceil(len(active) / reduction))
            sorted_states = sorted(
                active,
                key=lambda s: s["results"][target_epoch][metric_key],
                reverse=maximize,
            )
            keep = set(id(s) for s in sorted_states[:num_keep])
            new_active = []
            for state in active:
                if id(state) in keep:
                    new_active.append(state)
                else:
                    state["alive"] = False
                    state["session"].model.cpu()
                    state["session"] = None
            active = new_active

        return all_records

    def _export(self, records: List[Dict]) -> None:
        final_epoch = self.cfg.asha.rung_levels[-1]
        final_records = [r for r in records if r["epoch"] == final_epoch]
        final_records.sort(
            key=lambda r: r["metrics"].get(self.cfg.asha.metric_key, 0.0),
            reverse=self.cfg.asha.maximize,
        )
        top10 = final_records[:10]

        out_path = Path(self.cfg.output_dir) / "stage_b_top10.json"
        serializable = []
        for record in top10:
            serializable.append(
                {
                    "trial_id": record["trial_id"],
                    "val_top1": record["metrics"].get("val_top1"),
                    "specs": [
                        {"name": spec.name, "prob": spec.prob, "params": spec.params}
                        for spec in record["specs"]
                    ],
                }
            )
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)

