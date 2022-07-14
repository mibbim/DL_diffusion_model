"""
A diffusion Model object
"""
from __future__ import annotations

from typing import TypeVar, Tuple, Iterator, Literal
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
from torch import LongTensor

import torch
import torch.nn as nn
from .Unet import Generator
from .utils import default_device, Device
from .variance_schedule import LinearVarianceSchedule

IDT = TypeVar("IDT")  # Input Data Type
Loss = TypeVar("Loss")  # Loss function object
BetaSchedule = Literal["linear"]  # TypeVar("BetaSchedule")


def default_model():
    """Returns default model used mainly for testing"""
    return DiffusionModel(
        noise_predictor=Generator(1, 1),
        diffusion_steps_num=1000,
        evaluation_device=default_device,
    ).to(default_device)


# class LinearBeta:
#     def __init__(self,
#                  lower: float | None = 0.0001,
#                  upper: float | None = 0.04,
#                  steps: int = 100,
#                  device: Device | None = default_device):
#         self.min = lower
#         self.max = upper
#         self.steps = steps
#         self._betas = torch.linspace(self.min, self.max, self.steps).to(device)
#         self._alphas = 1. - self.betas
#         self._alpha_prod = torch.cumprod(self._alphas, dim=0)
#
#     @property
#     def betas(self):
#         return self._betas
#
#     def get_betas_t(self, t: LongTensor):
#         return self.betas.gather(-1, t).reshape(-1, 1, 1, 1)
#
#     @property
#     def alpha_prod(self):
#         return self._alpha_prod
#
#     def get_alpha_prod(self, t: torch.LongTensor):
#         return self.alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)
#
#     @property
#     def alphas(self):
#         return self._alphas
#
#     def get_alphas(self, t: torch.LongTensor):
#         return self._alphas.gather(-1, t).gather(-1, t).reshape(-1, 1, 1, 1)


class NoiseGenerator:
    def __init__(self,
                 beta: LinearVarianceSchedule = LinearVarianceSchedule(),
                 device: Device | None = default_device
                 ):
        self.device = device
        self.beta = beta
        self.max_t = self.beta.steps

    def _sample_t(self, x) -> LongTensor:
        """Samples the number of diffusion steps to apply to the batch x"""
        return torch.randint(0, self.max_t, (x.shape[0],),
                             dtype=torch.long).to(self.device)

    @staticmethod
    def _sample_noise(x: IDT) -> IDT:
        """Samples the noise to add to the batch of image"""
        return torch.randn_like(x).to(x.device)

    def add_noise(self, x: IDT, t: LongTensor | None = None) -> Tuple[IDT, IDT, LongTensor]:
        """Returns the noisy images, the noise, and the sampled times"""
        if t is None:
            t = self._sample_t(x)
        noise = self._sample_noise(x)
        alpha_prod_t = self.beta.get_alpha_prod_t(t)
        mean = x * alpha_prod_t.sqrt()
        std = (1 - alpha_prod_t).sqrt()
        return mean + noise.mul(std), noise, t


class DiffusionModel(nn.Module):  # Not sure should inherit
    def __init__(self,
                 noise_predictor: nn.Module,
                 noise_generator: NoiseGenerator | None = NoiseGenerator(),
                 diffusion_steps_num: int | None = 100,
                 evaluation_device: Device | None = default_device,
                 ) -> None:
        super().__init__()  # Not sure should inherit
        self._noise_generator = noise_generator
        self._noise_predictor = noise_predictor
        self.max_diff_steps = diffusion_steps_num
        self.device = evaluation_device

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

    def parameters(self, recurse: bool = True) -> Iterator[Parameter]:
        return self._noise_predictor.parameters()

    def train_step(self,
                   x: IDT,
                   optimizer: Optimizer,
                   loss_fun: Loss,
                   ):
        noisy_x, noise, t = self._noise_generator.add_noise(x)
        predicted_noise = self._noise_predictor(x, t)
        loss = loss_fun(noise, predicted_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss

    @torch.no_grad()
    def val_step(self, x: IDT, validation_metric):
        noisy_x, noise, t = self._noise_generator.add_noise(x)
        predicted_noise = self._noise_predictor(x, t)
        loss = validation_metric(noise, predicted_noise)
        return loss

    # @torch.no_grad()
    # def generate(self,
    #              n: int,
    #              image_dim: Tuple[int, int] = (28, 28)):
    #     raise NotImplementedError
    #     x = torch.randn(n, 1, *image_dim)
    #     for t in reversed(range(self.max_diff_steps)):
    #         noise = self._noise_predictor(x, torch.tensor([t for _ in range(n)], dtype=torch.long))
    #         alpha_t = self.beta.get_alphas(t)
    #         x = x - noise
