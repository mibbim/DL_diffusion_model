from __future__ import annotations

from abc import ABC
from .utils import default_device, Device

import torch
from torch import Tensor
from numpy import pi


class VarianceSchedule(ABC):
    steps: int
    _betas: torch.Tensor
    _alphas: torch.Tensor
    _alpha_prod: torch.Tensor

    @property
    def betas(self):
        return self._betas

    def get_betas_t(self, t: Tensor):
        return self.betas.gather(-1, t).reshape(-1, 1, 1, 1)

    @property
    def alpha_prod(self):
        return self._alpha_prod

    def get_alpha_prod_t(self, t: Tensor):
        return self.alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)

    @property
    def alphas(self):
        return self._alphas

    def get_alphas_t(self, t: Tensor):
        return self._alphas.gather(-1, t).reshape(-1, 1, 1, 1)


class LinearVarianceSchedule(VarianceSchedule):
    def __init__(self,
                 lower: float | None = 0.0001,
                 upper: float | None = 0.04,
                 steps: int = 100,
                 device: Device | None = default_device):
        self.min = lower
        self.max = upper
        self.steps = steps
        self._betas = torch.linspace(self.min, self.max, self.steps).to(device)
        self._alphas = 1. - self.betas
        self._alpha_prod = torch.cumprod(self._alphas, dim=0)


class CosineVarianceSchedule(VarianceSchedule):
    def __init__(self,
                 lower: float | None = 0.0001,
                 upper: float | None = 0.04,
                 steps: int = 100,
                 device: Device | None = default_device):
        self.min = lower
        self.max = upper
        self.steps = steps
        t = torch.arange(self.steps, device=device) / self.steps
        s = 0.008
        ft = torch.cos((t + s) / (1 + s) * pi * 0.5).square()
        self._alpha_prod = ft / ft[0]

        self._alphas = torch.empty_like(self._alpha_prod)
        self._alphas[0] = self._alpha_prod[0]
        self._alphas[1:] = self._alpha_prod[1:] / self._alpha_prod[:-1]
        self._betas = 1 - self._alphas
