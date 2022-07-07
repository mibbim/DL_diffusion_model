"""
A diffusion Model object
"""
from __future__ import annotations
from torch.optim.optimizer import Optimizer
from typing import TypeVar, Tuple

import torch.nn as nn

IDT = TypeVar("IDT")  # Input Data Type
Loss = TypeVar("Loss")  # Loss function object


class DiffusionModel(nn.Module):  # Not sure should inherit
    def __init__(self,
                 noise_predictor: nn.Module,
                 ) -> None:
        super().__init__()  # Not sure should inherit
        self.noise_predictor = noise_predictor
        raise NotImplementedError

    def train(self, mode: bool = True):
        """Forwarding the coll to inner module"""
        self.noise_predictor.train(mode=mode)
        return self

    def eval(self):
        """Forwarding the coll to inner module"""
        self.noise_predictor.eval()
        return self

    def add_noise(self, x: IDT) -> Tuple[IDT, IDT, int]:
        """Returns The noisy images, the noise, and the sampled times"""
        raise NotImplementedError

    def train_step(self,
                   x: IDT,
                   optimizer: Optimizer,
                   loss_fun: Loss,
                   ):
        noisy_x, noise, t = self.add_noise(x)
        predicted_noise = self.noise_predictor(x, t)
        loss = loss_fun(noise, predicted_noise)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
