from __future__ import annotations

import torch
from torch import LongTensor, Tensor
from .variance_schedule import LinearVarianceSchedule

from typing import Tuple
from .utils import default_device, Device, IDT


class NoiseGenerator:
    def __init__(self,
                 beta: LinearVarianceSchedule = LinearVarianceSchedule(),
                 device: Device | None = default_device
                 ):
        self.device = device
        self.beta = beta
        self.max_t = self.beta.steps

    def _sample_t(self, x) -> Tensor:
        """Samples the number of diffusion steps to apply to the batch x"""
        return torch.randint(0, self.max_t, (x.shape[0],), dtype=torch.long, device=self.device)

    @staticmethod
    def sample_noise(x: IDT) -> IDT:
        """Samples the noise to add to the batch of image"""
        return torch.randn_like(x, device=x.device)

    def add_noise(self, x: IDT, t: Tensor | None = None) -> Tuple[IDT, IDT, LongTensor]:
        """Returns the noisy images, the noise, and the sampled times"""
        if t is None:
            t = self._sample_t(x)
        noise = self.sample_noise(x)
        alpha_prod_t = self.beta.get_alpha_prod_t(t)
        mean = x * alpha_prod_t.sqrt()
        std = (1 - alpha_prod_t).sqrt()
        return mean + noise.mul(std), noise, t
