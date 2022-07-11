"""
This file contains the trainer class
"""

from __future__ import annotations

from typing import Literal

import torch.optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from tqdm.auto import trange

from .DiffusionModel import DiffusionModel, default_model
from .DiffusionModel import Loss
from .utils import default_device, Device

from .performance_meter import AverageMeter

optimizers_dict = {"Adam": torch.optim.Adam}


class Trainer:
    def __init__(self,
                 model: DiffusionModel = default_model(),
                 optimizer: Literal["Adam"] | None = "Adam",
                 learning_rate: float | None = 1e-3,

                 ) -> None:
        self.model = model
        self.opt = optimizers_dict[optimizer](params=model.parameters(), lr=learning_rate)
        self.history = {"loss": []}
        # self.lr = learning_rate

    def train(self,
              n_epochs: int,
              train_dataloader: DataLoader,
              device: Device = default_device,
              loss_function: Loss = mse_loss
              ) -> None:

        # Set the module in training mode
        self.model.train()

        epochs = trange(n_epochs, desc="Training epoch")

        average_meter = AverageMeter(["train_mse"])
        for epoch in epochs:
            for x, y in train_dataloader:
                x = x.to(device)  # Batch of Images
                loss = self.model.train_step(x, self.opt, loss_function)
                average_meter.update({"train_mse": loss.item()}, n=x.shape[0])

            self.history["loss"].append(average_meter.metrics["train_mse"]["avg"])
