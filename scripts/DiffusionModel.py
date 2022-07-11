"""
A diffusion Model object
"""
from __future__ import annotations
from typing import TypeVar, Tuple, Iterator

from typing import Literal

import matplotlib.pyplot as plt
from torch.nn.parameter import Parameter
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch import LongTensor
import torch

from Unet import Generator

IDT = TypeVar("IDT")  # Input Data Type
Loss = TypeVar("Loss")  # Loss function object
Device = Literal["cuda", "cpu"]

default_device: Device
if torch.cuda.is_available():
    default_device = "cuda"
else:
    default_device = "cpu"


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


def test_noise():
    from torchvision.transforms import ToPILImage
    import numpy as np
    total_steps = 1000
    model = DiffusionModel(
        noise_predictor=Generator(1, 1),
        diffusion_steps_num=total_steps,
        evaluation_device=default_device,
    ).to(default_device)

    torch.manual_seed(8)
    train, _ = load_data(1, 1, 1000)
    x, y = next(iter(train))
    img = x[0]
    for t in np.geomspace(1, total_steps - 1, 10):
        x_t, noise, t = model.add_noise(
            x.to(default_device),
            torch.tensor(int(t), dtype=torch.long).to(default_device)
        )
        img = torch.cat((img, x_t[0].cpu()), 2)
        plt.figure(t)
        plt.hist(x_t.cpu().flatten(), density=True)
        plt.title(f"{t=}")
    plt.show()

    ToPILImage()(img).show()


def test_trainstep():
    torch.manual_seed(8)

    model = DiffusionModel(
        noise_predictor=Generator(1, 1),
        diffusion_steps_num=2,
        evaluation_device=default_device,
    ).to(default_device)
    model.train()
    train, _ = load_data(2, 1, 1000)
    x, y = next(iter(train))
    losses = []
    for _ in range(100):
        l = model.train_step(x.to(model.device),
                             torch.optim.Adam(model.parameters()),
                             mse_loss,
                             )
        losses.append(l.item())
    plt.plot(losses)
    plt.show()


if __name__ == "__main__":
    from import_dataset import load_data
    from torch.nn.functional import mse_loss

    test_noise()
    test_trainstep()
