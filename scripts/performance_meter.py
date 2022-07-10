"""
"""
from __future__ import annotations

from abc import ABC
from typing import Dict

import torch


class PerformanceMeter(ABC):
    """This class should take care of updating the metrics during training"""

    def reset(self):
        raise NotImplementedError

    def update(self, values_dict: Dict[str, torch.Tensor]):
        raise NotImplementedError


class AverageMeter(PerformanceMeter):
    """
    Modified from:
    https://github.com/ansuini/DSSC_DL_2022/blob/main/labs/03-sgd-training.ipynb
    """

    def __init__(self, metrics):
        self.metrics = metrics
        self.val = {}
        self.count = {}
        self.sum = {}
        self.avg = {}
        self.reset()

    def _reset_single_metric(self, metric: str) -> None:
        self.val[metric] = 0  # val holds the current stat
        self.avg[metric] = 0  # avg holds the cumulative average
        self.sum[metric] = 0  # sum holds the cumulative value
        self.count[metric] = 0  # count holds the number of instances seen

    def _update_single_metric(self, val: float,
                              metric: str,
                              n: int | None = 1) -> None:
        self.val[metric] = val
        self.count[metric] += n
        self.sum[metric] += val * n
        self.avg[metric] = self.sum[metric] / self.count[metric]

    def update(self, values_dict: Dict[str, torch.Tensor], n: int | None = 1) -> None:
        for metric, val in values_dict:
            self._update_single_metric(val, metric, n)

    def reset(self) -> None:
        for metric, val in self.metrics:
            self._reset_single_metric(metric)
