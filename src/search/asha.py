from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence


@dataclass
class ASHAConfig:
    rung_levels: Sequence[int]
    reduction_factor: int
    metric_key: str = "val_top1"
    maximize: bool = True

    def validate(self) -> None:
        if sorted(self.rung_levels) != list(self.rung_levels):
            raise ValueError("rung_levels 必须按升序排列")
        if self.reduction_factor < 2:
            raise ValueError("reduction_factor 至少为 2")

