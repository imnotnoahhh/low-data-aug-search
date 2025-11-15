from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from .mixup import MixupConfig, mixup_batch, mixup_criterion


def accuracy(output: torch.Tensor, target: torch.Tensor, topk: Sequence[int] = (1,)) -> List[float]:
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res: List[float] = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size).item())
        return res


@dataclass
class OptimizerConfig:
    lr: float = 0.2
    momentum: float = 0.9
    weight_decay: float = 5e-4


@dataclass
class SchedulerConfig:
    max_epochs: int
    warmup_epochs: int = 5
    eta_min: float = 1e-4


@dataclass
class TrainingConfig:
    epochs: int = 30
    grad_clip: float = 1.0
    use_amp: bool = True
    log_interval: int = 50


class TrainingSession:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        scheduler: torch.optim.lr_scheduler.CosineAnnealingLR,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader],
        criterion: nn.Module,
        mixup_cfg: MixupConfig,
        train_cfg: TrainingConfig,
        sched_cfg: SchedulerConfig,
        device: torch.device,
    ) -> None:
        self.model = model.to(device)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.mixup_cfg = mixup_cfg
        self.train_cfg = train_cfg
        self.sched_cfg = sched_cfg
        self.device = device
        self.scaler = GradScaler(enabled=train_cfg.use_amp)
        self.current_epoch = 0
        self.history: List[Dict[str, float]] = []

    def _set_warmup_lr(self) -> None:
        if self.current_epoch >= self.sched_cfg.warmup_epochs:
            return
        warmup_factor = (self.current_epoch + 1) / max(1, self.sched_cfg.warmup_epochs)
        base_lr = self.optimizer.defaults["lr"]
        for param_group in self.optimizer.param_groups:
            param_group["lr"] = base_lr * warmup_factor

    def _train_one_epoch(self) -> Dict[str, float]:
        self.model.train()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        self._set_warmup_lr()

        for step, (images, targets) in enumerate(self.train_loader):
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)

            with autocast(enabled=self.train_cfg.use_amp):
                inputs, targets_a, targets_b, lam = mixup_batch(images, targets, self.mixup_cfg)
                outputs = self.model(inputs)
                loss = mixup_criterion(self.criterion, outputs, targets_a, targets_b, lam)

            self.scaler.scale(loss).backward()
            if self.train_cfg.grad_clip is not None:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.train_cfg.grad_clip)
            self.scaler.step(self.optimizer)
            self.scaler.update()

            with torch.no_grad():
                prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
                batch_size = targets.size(0)
                total_loss += loss.item() * batch_size
                total_top1 += prec1 * batch_size
                total_top5 += prec5 * batch_size
                total_samples += batch_size

            if (
                self.train_cfg.log_interval
                and (step + 1) % self.train_cfg.log_interval == 0
            ):
                lr = self.optimizer.param_groups[0]["lr"]
                print(
                    f"Epoch {self.current_epoch+1} Step {step+1}: "
                    f"loss={loss.item():.4f}, top1={prec1:.2f}, lr={lr:.5f}"
                )

        epoch_loss = total_loss / total_samples
        epoch_top1 = total_top1 / total_samples
        epoch_top5 = total_top5 / total_samples
        return {
            "train_loss": epoch_loss,
            "train_top1": epoch_top1,
            "train_top5": epoch_top5,
        }

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Dict[str, float]:
        loader = loader or self.val_loader
        if loader is None:
            return {}
        self.model.eval()
        total_loss = 0.0
        total_top1 = 0.0
        total_top5 = 0.0
        total_samples = 0

        for images, targets in loader:
            images = images.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            with autocast(enabled=self.train_cfg.use_amp):
                outputs = self.model(images)
                loss = self.criterion(outputs, targets)
            prec1, prec5 = accuracy(outputs, targets, topk=(1, 5))
            batch_size = targets.size(0)
            total_loss += loss.item() * batch_size
            total_top1 += prec1 * batch_size
            total_top5 += prec5 * batch_size
            total_samples += batch_size

        return {
            "val_loss": total_loss / total_samples,
            "val_top1": total_top1 / total_samples,
            "val_top5": total_top5 / total_samples,
        }

    def train_until(self, target_epoch: int) -> Dict[str, float]:
        while self.current_epoch < target_epoch and self.current_epoch < self.sched_cfg.max_epochs:
            stats = self._train_one_epoch()
            self.current_epoch += 1
            if self.current_epoch >= self.sched_cfg.warmup_epochs:
                self.scheduler.step()
            val_stats = self.evaluate()
            stats.update(val_stats)
            stats["epoch"] = self.current_epoch
            self.history.append(stats)
        return self.history[-1] if self.history else {}


def build_optimizer(model: nn.Module, cfg: OptimizerConfig) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=cfg.lr,
        momentum=cfg.momentum,
        weight_decay=cfg.weight_decay,
        nesterov=False,
    )


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: SchedulerConfig) -> torch.optim.lr_scheduler.CosineAnnealingLR:
    t_max = max(1, cfg.max_epochs - cfg.warmup_epochs)
    return torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=t_max,
        eta_min=cfg.eta_min,
    )

