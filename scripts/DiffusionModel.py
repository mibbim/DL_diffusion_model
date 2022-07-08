"""
A diffusion Model object
"""
from __future__ import annotations
from typing import TypeVar, Tuple

from typing import Literal
from torch.optim.optimizer import Optimizer
import torch.nn as nn
from torch import LongTensor
import torch

IDT = TypeVar("IDT")  # Input Data Type
Loss = TypeVar("Loss")  # Loss function object
Device = Literal["cuda", "cpu"]


class DiffusionModel(nn.Module):  # Not sure should inherit
    def __init__(self,
                 noise_predictor: nn.Module,
                 diffusion_steps_num: int | None = 100,
                 evaluation_device: Device | None = "cpu",
                 ) -> None:
        super().__init__()  # Not sure should inherit
        self.noise_predictor = noise_predictor
        self.max_diff_steps = diffusion_steps_num
        self.betas = torch.linspace(0.0001, 0.04, self.max_diff_steps).to(evaluation_device)
        self._alphas = 1. - self.betas
        self._alpha_prod = torch.cumprod(self._alphas, dim=0)

    def train(self, mode: bool = True):
        """Forwarding the call to inner module"""
        self.noise_predictor.train(mode=mode)
        return self

    def eval(self):
        """Forwarding the call to inner module"""
        self.noise_predictor.eval()
        return self

    def _sample_t(self, x) -> LongTensor:
        """Samples the number of diffusion steps to apply to the batch x"""
        return torch.randint(0, self.max_diff_steps, (x.size[0],),
                             dtype=torch.long).to(self.evaluation_device)

    @staticmethod
    def _sample_noise(x: IDT) -> IDT:
        """Samples the noise to add to the batch of image"""
        return torch.randn_like(x).to(x.device)

    def add_noise(self, x: IDT) -> Tuple[IDT, IDT, LongTensor]:
        """Returns the noisy images, the noise, and the sampled times"""
        t = self._sample_t(x)
        noise = self._sample_noise(x)
        mean = x * self._alpha_prod.gather(0, t).sqrt()
        var = 1 - self._alpha_prod.gather(0, t)
        return mean + noise.mul(var.sqrt()), noise, t

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


def main():
    # from Data.import_dataset import load_data
    # import torchvision.transforms as T
    #
    # torch.manual_seed(8)
    # train, test = load_data(5, 1, 1000)  # load
    # for x, y in train:
    #     img = T.ToPILImage()(x[0])
    #     img.show()
    #     pass
    # print()
    pass


if __name__ == "__main__":
    main()
