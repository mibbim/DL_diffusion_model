"""
A diffusion Model object
"""
from __future__ import annotations

from typing import TypeVar, Tuple, Iterator
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch import LongTensor

import torch
import torch.nn as nn
from .Unet import Generator
from .utils import default_device, Device

IDT = TypeVar("IDT")  # Input Data Type
Loss = TypeVar("Loss")  # Loss function object


def default_model():
    """Returns default model used mainly for testing"""
    return DiffusionModel(
        noise_predictor=Generator(1, 1),
        diffusion_steps_num=1000,
        evaluation_device=default_device,
    ).to(default_device)


class DiffusionModel(nn.Module):  # Not sure should inherit
    def __init__(self,
                 noise_predictor: nn.Module,
                 diffusion_steps_num: int | None = 100,
                 evaluation_device: Device | None = default_device,
                 ) -> None:
        super().__init__()  # Not sure should inherit
        self._noise_predictor = noise_predictor
        self.max_diff_steps = diffusion_steps_num
        self.device = evaluation_device
        self.betas = torch.linspace(0.0001, 0.04, self.max_diff_steps).to(self.device)
        self._alphas = 1. - self.betas
        self._alpha_prod = torch.cumprod(self._alphas, dim=0)

    def train(self, mode: bool = True) -> DiffusionModel:
        """Forwarding the call to inner module"""
        self._noise_predictor.train(mode=mode)
        return self

    def eval(self):
        """Forwarding the call to inner module"""
        self._noise_predictor.eval()
        return self

    def to(self, *args, **kwargs):
        """Forwarding the call to inner module"""
        self._noise_predictor.to(*args, **kwargs)
        return self

    def _sample_t(self, x) -> LongTensor:
        """Samples the number of diffusion steps to apply to the batch x"""
        return torch.randint(0, self.max_diff_steps, (x.shape[0],),
                             dtype=torch.long).to(self.device)

    @staticmethod
    def _sample_noise(x: IDT) -> IDT:
        """Samples the noise to add to the batch of image"""
        return torch.randn_like(x).to(x.device)

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self._noise_predictor.parameters()

    def add_noise(self, x: IDT, t: LongTensor | None = None) -> Tuple[IDT, IDT, LongTensor]:
        """Returns the noisy images, the noise, and the sampled times"""
        if t is None:
            t = self._sample_t(x)
        noise = self._sample_noise(x)
        alpha_prod_t = self._alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)
        # self._alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)
        mean = x * alpha_prod_t.sqrt()
        std = (1 - alpha_prod_t).sqrt()
        return mean + noise.mul(std), noise, t

    def train_step(self,
                   x: IDT,
                   optimizer: Optimizer,
                   loss_fun: Loss,
                   ):
        noisy_x, noise, t = self.add_noise(x)
        predicted_noise = self._noise_predictor(x, t)
        loss = loss_fun(noise, predicted_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
