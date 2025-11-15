from __future__ import annotations

import csv
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Sequence

import numpy as np


@dataclass
class EvalEntry:
    name: str
    seed: int
    split: str
    top1: float
    top5: float


def bootstrap_ci(values: Sequence[float], num_samples: int = 10000, alpha: float = 0.05):
    rng = np.random.default_rng(0)
    samples = []
    arr = np.array(values)
    n = len(arr)
    for _ in range(num_samples):
        idx = rng.integers(0, n, n)
        samples.append(arr[idx].mean())
    lower = np.percentile(samples, 100 * (alpha / 2))
    upper = np.percentile(samples, 100 * (1 - alpha / 2))
    return float(lower), float(upper)


@dataclass
class ResultsAggregator:
    entries: List[EvalEntry] = field(default_factory=list)

    def add(self, entry: EvalEntry) -> None:
        self.entries.append(entry)

    def summarize(self) -> Dict[str, Dict[str, float]]:
        summary: Dict[str, Dict[str, float]] = {}
        grouped: Dict[str, List[EvalEntry]] = {}
        for entry in self.entries:
            key = f"{entry.name}:{entry.split}"
            grouped.setdefault(key, []).append(entry)

        for key, vals in grouped.items():
            scores = [v.top1 for v in vals]
            mean = float(np.mean(scores))
            std = float(np.std(scores, ddof=1))
            ci_low, ci_high = bootstrap_ci(scores)
            summary[key] = {
                "mean_top1": mean,
                "std_top1": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "num_seeds": len(scores),
            }
        return summary

    def to_csv(self, path: str) -> None:
        summary = self.summarize()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "name",
                    "split",
                    "mean_top1",
                    "std_top1",
                    "ci_low",
                    "ci_high",
                    "num_seeds",
                ],
            )
            writer.writeheader()
            for key, stats in summary.items():
                name, split = key.split(":")
                writer.writerow(
                    {
                        "name": name,
                        "split": split,
                        **stats,
                    }
                )

