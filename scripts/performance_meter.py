"""
"""
from __future__ import annotations

from abc import ABC
from typing import Dict, Iterable

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

    def __init__(self, metrics: Iterable[str]):
        self._reset_metric_dict = {
            "val": 0,
            "count": 0,
            "sum": 0,
            "avg": 0,
        }
        self.metrics = {}
        self.reset(metrics)

    def _reset_single_metric(self, metric: str) -> None:
        self.metrics[metric] = self._reset_metric_dict.copy()

    def _update_single_metric(self,
                              metric: str,
                              val: float,
                              n: int | None = 1) -> None:
        """

        :param metric: string that identifies the metric e.g. "mse"
        :param val: new value of the loss function
        :param n: number of seen examples
        """

        self.metrics[metric]["val"] = val
        self.metrics[metric]["count"] += n
        self.metrics[metric]["sum"] += val * n
        self.metrics[metric]["avg"] = self.metrics[metric]["sum"] / self.metrics[metric]["count"]

    def update(self, values_dict: Dict[str, float], n: int | None = 1) -> None:
        for metric, val in values_dict.items():
            self._update_single_metric(metric, val, n)

    def reset(self, metrics_list: Iterable[str] | None = None) -> None:
        if metrics_list is None:
            metrics_list = self.metrics.keys()
        for metric in metrics_list:
            self._reset_single_metric(metric)


def test_avg_meter():
    am = AverageMeter(["mse", "vlb"])
    print(f"Instatiated with mse and vlb: \n", am.metrics)
    am.update({"mse": 10, "vlb": 3})
    print(f"Updated with 10, and 3: \n", am.metrics)
    am.update({"mse": 30, "vlb": 5})
    print(f"Updated with 30 and 5: \n", am.metrics)
    am.reset()
    print(f"Reset: \n", am.metrics)


if __name__ == "__main__":
    test_avg_meter()
