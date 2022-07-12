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
from .performance_meter import AverageMeter
from .utils import default_device, Device

optimizers_dict = {"Adam": torch.optim.Adam}


class Trainer:
    def __init__(self,
                 model: DiffusionModel = default_model(),
                 optimizer: Literal["Adam"] | None = "Adam",
                 learning_rate: float | None = 1e-3,

                 ) -> None:
        self.model = model
        self.opt = optimizers_dict[optimizer](params=model.parameters(), lr=learning_rate)
        self.history = {"train_loss": []}
        # self.lr = learning_rate

    def train(self,
              n_epochs: int,
              train_dataloader: DataLoader,
              val_dataloader: DataLoader,
              device: Device = default_device,
              loss_function: Loss = mse_loss,
              validation_metric: Loss | None = None,
              valid_each: int | None = 1,  # Epochs between evaluation
              ) -> None:

        # Set the module in training mode
        self.model.train()
        if validation_metric is None:
            validation_metric = loss_function

        epochs = trange(n_epochs, desc="Training epoch")

        average_meter = AverageMeter(["train_mse"])
        for epoch in epochs:
            for x, y in train_dataloader:
                x = x.to(device)  # Batch of Images
                loss = self.model.train_step(x, self.opt, loss_function)
                average_meter.update({"train_mse": loss.item()}, n=x.shape[0])

            self.history["train_loss"].append((epoch, average_meter.metrics["train_mse"]["avg"]))
            average_meter.reset()
            if epoch % valid_each == 0:
                self.model.eval()
                self.validate(val_dataloader, validation_metric, epoch)
                self.model.train()

    def validate(self, data_loader: DataLoader, validation_metric: Loss, epoch: int):
        validation_meter = AverageMeter(["val_mse"])
        for x, y in data_loader:
            x = x
            val_loss = self.model.validation_step(x, validation_metric)
            validation_meter.update({"val_mse": val_loss.item()}, n=x.shape[0])
        self.history["val_loss"].append((epoch, validation_meter.metrics["val_loss"]["avg"]))
        validation_meter.reset()
