"""
This file contains the trainer class
"""

from __future__ import annotations

from typing import Literal
from torch.optim.optimizer import Optimizer
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from tqdm.auto import trange
from DiffusionModel import DiffusionModel
from torch.nn.functional import mse_loss

from DiffusionModel import Loss


class Trainer:
    def __init__(self,
                 optimizer: Optimizer,
                 learning_rate: float,
                 ) -> None:
        self.opt = optimizer
        self.lr = learning_rate

    def train(self,
              diff_model: DiffusionModel,
              n_epochs: int,
              training_data: Dataset,
              batch_size: int,
              device: Literal["cuda", "cpu"] = "cuda",
              loss_function: Loss = mse_loss
              ) -> None:

        # Set the module in training mode
        diff_model.train()

        train_dataloader = DataLoader(training_data, batch_size=batch_size, shuffle=True)
        epochs = trange(n_epochs, desc="Training epoch")
        losses = []
        for epoch in epochs:
            loss = 0
            for x, y in train_dataloader:
                x = x.to(device)  # Batch of Images
                loss += diff_model.train_step(x, self.opt, loss_function)

            losses.append(loss)
