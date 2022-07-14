from __future__ import annotations

from abc import ABC, abstractmethod
from .utils import default_device, Device

import torch
from torch import LongTensor


class VarianceSchedule(ABC):

    @property
    @abstractmethod
    def betas(self):
        raise NotImplementedError

    @abstractmethod
    def get_betas_t(self, t: LongTensor):
        raise NotImplementedError

    @property
    @abstractmethod
    def alphas(self):
        raise NotImplementedError

    @abstractmethod
    def get_alphas_t(self, t: LongTensor):
        raise NotImplementedError

    @property
    @abstractmethod
    def alpha_prod(self):
        raise NotImplementedError

    @abstractmethod
    def get_alpha_prod_t(self, t: LongTensor):
        raise NotImplementedError


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

    @property
    def betas(self):
        return self._betas

    def get_betas_t(self, t: LongTensor):
        return self.betas.gather(-1, t).reshape(-1, 1, 1, 1)

    @property
    def alpha_prod(self):
        return self._alpha_prod

    def get_alpha_prod_t(self, t: LongTensor):
        return self.alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)

    @property
    def alphas(self):
        return self._alphas

    def get_alphas_t(self, t: LongTensor):
        return self._alphas.gather(-1, t).gather(-1, t).reshape(-1, 1, 1, 1)
