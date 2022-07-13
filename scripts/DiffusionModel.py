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


class LinearBeta:
    def __init__(self,
                 lower: float | None = 0.0001,
                 upper: float | None = 0.04,
                 steps: int = 100,
                 device: Device | None = default_device):
        self.min = lower
        self.max = upper
        self.steps = steps
        self.betas = torch.linspace(self.min, self.max, self.steps).to(device)
        self._alphas = 1. - self.betas
        self.alpha_prod = torch.cumprod(self._alphas, dim=0)


class NoiseGenerator:
    def __init__(self,
                 max_diff_steps: int | None = 100,
                 beta: LinearBeta = LinearBeta(),
                 # min_beta: float | None = 0.0001,
                 # max_beta: float | None = 0.04,
                 # beta_schedule: BetaSchedule | None = "linear",
                 device: Device | None = default_device
                 ):
        self.device = device
        self.beta = beta
        self.max_diff_steps = self.beta.steps

    def _sample_t(self, x) -> LongTensor:
        """Samples the number of diffusion steps to apply to the batch x"""
        return torch.randint(0, self.max_diff_steps, (x.shape[0],),
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
        alpha_prod_t = self.beta.alpha_prod.gather(-1, t).reshape(-1, 1, 1, 1)
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
        self.betas = torch.linspace(0.0001, 0.04, self.max_diff_steps).to(
            self.device)  # try later cosine
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

    def val_step(self, x, validation_metric):
        noisy_x, noise, t = self._noise_generator.add_noise(x)
        predicted_noise = self._noise_predictor(x, t)
        loss = validation_metric(noise, predicted_noise)
        return loss
