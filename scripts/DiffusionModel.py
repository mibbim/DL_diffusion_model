"""
A diffusion Model object
"""
from __future__ import annotations

from typing import TypeVar, Iterator, Literal, Tuple
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer

import torch
import torch.nn as nn
from .Unet import Generator
from .utils import default_device, Device, IDT
from .noiseGenerator import NoiseGenerator

Loss = TypeVar("Loss")  # Loss function object
BetaSchedule = Literal["linear"]  # TypeVar("BetaSchedule")


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

    @torch.no_grad()
    def generate(self,
                 n: int = 1,
                 image_dim: Tuple[int, int] = (28, 28)):
        x = torch.randn(n, 1, *image_dim)
        for t in reversed(range(self.max_diff_steps)):
            pred_noise = self._noise_predictor(x, torch.tensor([t for _ in range(n)],
                                                               dtype=torch.long))
            beta_t = self.beta.get_betas_t(t)
            alpha_t = self.beta.get_alphas_t(t)
            alpha_prod_t = self.beta.get_alpha_prod_t(t)
            pred_noise_term = (1 - alpha_t) * torch.div(pred_noise, (1 - alpha_prod_t).sqrt())
            noise_term = self._noise_generator.sample_noise(x).mul(beta_t)
            x = x - pred_noise_term + noise_term
            x = torch.div(x, alpha_t.sqrt())

        raise NotImplementedError
