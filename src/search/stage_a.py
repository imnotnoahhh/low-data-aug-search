from __future__ import annotations

import csv
import json
import random
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch
import torch.nn as nn
from torch.quasirandom import SobolEngine
from torchvision import datasets, transforms as T

from src.data.augmentations import CIFAR_MEAN, CIFAR_STD, TransformSpec, build_aug_chain
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
from src.utils.visualization import save_side_by_side


def _map_range(value: float, low: float, high: float) -> float:
    return low + (high - low) * value


def sample_stage_a_configs(
    transform_name: str,
    n_samples: int,
    seed: int = 0,
) -> List[TransformSpec]:
    """根据计划中合法范围，使用 Sobol 序列采样 (p, m) 组合."""

    def sobol(dim: int) -> torch.Tensor:
        engine = SobolEngine(dim, scramble=True, seed=seed)
        return engine.draw(n_samples)

    specs: List[TransformSpec] = []

    if transform_name == "RandomResizedCrop":
        samples = sobol(3)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 1.0)
            scale = (_map_range(s[1].item(), 0.5, 1.0), 1.0)
            ratio = (_map_range(s[2].item(), 0.75, 1.33), 1.33)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"scale": scale, "ratio": ratio},
                )
            )
    elif transform_name == "RandomCrop":
        samples = sobol(2)
        padding_choices = [0, 2, 4, 8]
        for s in samples:
            prob = _map_range(s[0].item(), 0.5, 1.0)
            idx = min(int(s[1].item() * len(padding_choices)), len(padding_choices) - 1)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"padding": padding_choices[idx]},
                )
            )
    elif transform_name == "RandomRotation":
        samples = sobol(2)
        degree_choices = [5, 10, 15]
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.75)
            idx = min(int(s[1].item() * len(degree_choices)), len(degree_choices) - 1)
            deg = degree_choices[idx]
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"degrees": (-deg, deg)},
                )
            )
    elif transform_name == "RandomPerspective":
        samples = sobol(2)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.75)
            distortion = _map_range(s[1].item(), 0.05, 0.6)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"distortion_scale": distortion},
                )
            )
    elif transform_name == "RandomHorizontalFlip":
        samples = sobol(1)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.9)
            specs.append(TransformSpec(name=transform_name, prob=prob))
    elif transform_name == "ColorJitter":
        samples = sobol(5)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.9)
            brightness = _map_range(s[1].item(), 0.2, 0.6)
            contrast = _map_range(s[2].item(), 0.2, 0.6)
            saturation = _map_range(s[3].item(), 0.2, 0.6)
            hue = _map_range(s[4].item(), 0.05, 0.15)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={
                        "brightness": brightness,
                        "contrast": contrast,
                        "saturation": saturation,
                        "hue": hue,
                    },
                )
            )
    elif transform_name == "RandomGrayscale":
        samples = sobol(1)
        for s in samples:
            prob = _map_range(s[0].item(), 0.05, 0.5)
            specs.append(TransformSpec(name=transform_name, prob=prob))
    elif transform_name == "GaussianBlur":
        samples = sobol(3)
        kernel_choices = [3, 5]
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.75)
            k_idx = min(int(s[1].item() * len(kernel_choices)), len(kernel_choices) - 1)
            sigma = _map_range(s[2].item(), 0.1, 2.0)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"kernel_size": kernel_choices[k_idx], "sigma": sigma},
                )
            )
    elif transform_name == "GaussianNoise":
        samples = sobol(2)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.75)
            sigma = _map_range(s[1].item(), 0.02, 0.2)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={"sigma": sigma},
                )
            )
    elif transform_name == "RandomErasing":
        samples = sobol(4)
        for s in samples:
            prob = _map_range(s[0].item(), 0.25, 0.75)
            scale_min = _map_range(s[1].item(), 0.02, 0.25)
            scale_max = min(scale_min + _map_range(s[2].item(), 0.05, 0.4 - scale_min), 0.4)
            ratio_min = _map_range(s[3].item(), 0.3, 1.0)
            ratio_max = _map_range(s[3].item(), 1.0, 3.3)
            specs.append(
                TransformSpec(
                    name=transform_name,
                    prob=prob,
                    params={
                        "scale": (scale_min, scale_max),
                        "ratio": (ratio_min, ratio_max),
                    },
                )
            )
    else:
        raise ValueError(f"未定义 {transform_name} 的采样空间")

    return specs


@dataclass
class StageAConfig:
    transform_name: str
    n_samples: int = 32
    sobol_seed: int = 0
    seed: int = 0
    data: DataModuleConfig = field(default_factory=lambda: DataModuleConfig(root="data"))
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    scheduler: SchedulerConfig = field(
        default_factory=lambda: SchedulerConfig(max_epochs=30, warmup_epochs=5, eta_min=1e-4)
    )
    training: TrainingConfig = field(
        default_factory=lambda: TrainingConfig(
            epochs=30,
            grad_clip=1.0,
            use_amp=True,
            log_interval=5,
        )
    )
    mixup: MixupConfig = field(default_factory=lambda: MixupConfig(alpha=0.2, enabled=True))
    asha: ASHAConfig = field(
        default_factory=lambda: ASHAConfig(
            rung_levels=[10, 20, 30], reduction_factor=2, metric_key="val_top1", maximize=True
        )
    )
    output_dir: str = "artifacts/stage_a"
    visual_indices: Sequence[int] = (0, 1, 2, 3)
    visual_dirname: str = "examples"
    visual_meta_filename: str = "examples_meta.json"


class StageAScreener:
    """阶段 A：在合法范围内对单个变换进行 Sobol 采样 + ASHA 粗筛。"""

    def __init__(self, cfg: StageAConfig, device: Optional[torch.device] = None) -> None:
        self.cfg = cfg
        self.cfg.asha.validate()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        Path(self.cfg.output_dir).mkdir(parents=True, exist_ok=True)
        self._visual_dataset = datasets.CIFAR100(
            root=self.cfg.data.root,
            train=True,
            download=True,
            transform=None,
        )
        self._to_tensor = T.ToTensor()
        self._mean = torch.tensor(CIFAR_MEAN).view(3, 1, 1)
        self._std = torch.tensor(CIFAR_STD).view(3, 1, 1)
        self._visual_indices = list(self.cfg.visual_indices)

    def _build_session(self, spec: TransformSpec, seed: int) -> TrainingSession:
        self._set_seed(seed)
        train_transform = build_aug_chain([spec])
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

    @staticmethod
    def _set_seed(seed: int) -> None:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _asha_loop(self, specs: List[TransformSpec]) -> List[Dict]:
        rung_levels = self.cfg.asha.rung_levels
        reduction = self.cfg.asha.reduction_factor
        metric_key = self.cfg.asha.metric_key
        maximize = self.cfg.asha.maximize

        trial_states = [
            {
                "id": f"{self.cfg.transform_name}_{idx}",
                "spec": spec,
                "seed": self.cfg.seed + idx,
                "session": self._build_session(spec, self.cfg.seed + idx),
                "alive": True,
                "results": {},
            }
            for idx, spec in enumerate(specs)
        ]

        all_records: List[Dict] = []
        active = trial_states

        for rung_idx, target_epoch in enumerate(rung_levels):
            rung_records: List[Dict] = []
            for state in active:
                if not state["alive"]:
                    continue
                spec: TransformSpec = state["spec"]
                print(
                    f"[StageA][{self.cfg.transform_name}] Trial {state['id']} "
                    f"→ target_epoch {target_epoch}, prob={spec.prob:.2f}, params={spec.params}"
                )
                metrics = state["session"].train_until(target_epoch)
                record = {
                    "trial_id": state["id"],
                    "epoch": target_epoch,
                    "metrics": metrics,
                    "spec": state["spec"],
                }
                rung_records.append(record)
                state["results"][target_epoch] = metrics
                all_records.append(record)
            if rung_idx == len(rung_levels) - 1:
                break

            # 根据当前 rung 的指标排序，保留 Top 1/reduction
            num_keep = max(1, len(active) // reduction)
            sorted_states = sorted(
                active,
                key=lambda s: s["results"][target_epoch][metric_key],
                reverse=maximize,
            )
            keep_set = set(id(s) for s in sorted_states[:num_keep])
            new_active = []
            for state in active:
                if id(state) in keep_set:
                    new_active.append(state)
                else:
                    state["alive"] = False
                    # 释放显存
                    state["session"].model.cpu()
                    state["session"] = None
            active = new_active

        return all_records

    def run(self) -> List[Dict]:
        specs = sample_stage_a_configs(
            self.cfg.transform_name,
            n_samples=self.cfg.n_samples,
            seed=self.cfg.sobol_seed,
        )
        records = self._asha_loop(specs)
        self._export(records)
        return records

    def _export(self, records: List[Dict]) -> None:
        csv_path = Path(self.cfg.output_dir) / f"{self.cfg.transform_name}_results.csv"
        json_path = Path(self.cfg.output_dir) / f"{self.cfg.transform_name}_topk.json"
        visual_dir = Path(self.cfg.output_dir) / self.cfg.visual_dirname
        visual_dir.mkdir(parents=True, exist_ok=True)

        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "trial_id",
                    "epoch",
                    "train_top1",
                    "val_top1",
                    "train_loss",
                    "val_loss",
                    "prob",
                    "params",
                ],
            )
            writer.writeheader()
            for record in records:
                metrics = record["metrics"]
                spec: TransformSpec = record["spec"]
                writer.writerow(
                    {
                        "trial_id": record["trial_id"],
                        "epoch": record["epoch"],
                        "train_top1": metrics.get("train_top1"),
                        "val_top1": metrics.get("val_top1"),
                        "train_loss": metrics.get("train_loss"),
                        "val_loss": metrics.get("val_loss"),
                        "prob": spec.prob,
                        "params": json.dumps(spec.params),
                    }
                )

        # 导出最终 Top-4
        final_epoch = self.cfg.asha.rung_levels[-1]
        final_records = [
            r
            for r in records
            if r["epoch"] == final_epoch
        ]
        final_records.sort(
            key=lambda r: r["metrics"].get(self.cfg.asha.metric_key, 0.0),
            reverse=self.cfg.asha.maximize,
        )
        topk = final_records[:4]
        serializable = [
            {
                "trial_id": r["trial_id"],
                "val_top1": r["metrics"].get("val_top1"),
                "spec": {"prob": r["spec"].prob, "params": r["spec"].params},
            }
            for r in topk
        ]
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(serializable, f, indent=2)
        self._visualize_topk(topk, visual_dir)

    def _visualize_topk(self, topk: List[Dict], visual_dir: Path, num_examples: int = 4) -> None:
        meta: List[Dict] = []
        for rank, record in enumerate(topk, start=1):
            spec: TransformSpec = record["spec"]
            metrics = record.get("metrics", {})
            filename = visual_dir / self._build_visual_filename(rank, spec, metrics)
            self._render_examples(spec, filename, num_examples=num_examples)
            meta.append(
                {
                    "rank": rank,
                    "trial_id": record["trial_id"],
                    "file": filename.name,
                    "prob": spec.prob,
                    "params": spec.params,
                    "val_top1": metrics.get("val_top1"),
                }
            )
        meta_path = Path(self.cfg.output_dir) / self.cfg.visual_meta_filename
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

    def _build_visual_filename(self, rank: int, spec: TransformSpec, metrics: Dict) -> str:
        prob_str = f"p{spec.prob:.2f}"
        param_parts = []
        for key, value in spec.params.items():
            if isinstance(value, (list, tuple)):
                val_str = "-".join(f"{v:.2f}" for v in value)
            else:
                val_str = f"{float(value):.2f}"
            param_parts.append(f"{key}_{val_str}")
        params_str = "__".join(param_parts)
        val = metrics.get("val_top1")
        val_str = f"val{val:.2f}" if val is not None else "valNA"
        return f"rank{rank}_{prob_str}_{params_str}_{val_str}.png"

    def _render_examples(
        self,
        spec: TransformSpec,
        filepath: Path,
        num_examples: int = 4,
    ) -> None:
        transform = build_aug_chain([spec])
        originals = []
        augmented = []
        for idx in self._visual_indices[:num_examples]:
            img, _ = self._visual_dataset[idx]
            orig = self._to_tensor(img)
            aug = transform(img)
            aug = (aug * self._std + self._mean).clamp(0.0, 1.0)
            originals.append(orig)
            augmented.append(aug)
        save_side_by_side(originals, augmented, str(filepath), nrow=num_examples)

