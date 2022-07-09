"""
This file contains the trainer class
"""

from __future__ import annotations

from typing import Literal

import torch.optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from tqdm.auto import trange

from DiffusionModel import DiffusionModel
from DiffusionModel import Loss, Device
from DiffusionModel import default_device

optimizers_dict = {"Adam": torch.optim.Adam}


class Trainer:
    def __init__(self,
                 optimizer: Literal["Adam"] | None = "Adam",
                 learning_rate: float | None = 1e-3,
                 ) -> None:

        self.opt = optimizers_dict[optimizer](learning_rate)
        # self.lr = learning_rate

    def train(self,
              diff_model: DiffusionModel,
              n_epochs: int,
              train_dataloader: DataLoader,
              device: Device = default_device,
              loss_function: Loss = mse_loss
              ) -> None:

        # Set the module in training mode
        diff_model.train()

        epochs = trange(n_epochs, desc="Training epoch")
        losses = []
        for epoch in epochs:
            loss = 0
            for x, y in train_dataloader:
                x = x.to(device)  # Batch of Images
                loss += diff_model.train_step(x, self.opt, loss_function)

            losses.append(loss)


# def test_train():
#     """Tests that the train function works, if the model works correctly"""
#
#     class DiffModelStub(DiffusionModel):
#         # def __init__(self):
#             # super(DiffModelStub, self).__init__(None, 3, )
#
#         from DiffusionModel import IDT
#
#         def train_step(self,
#                        x: IDT,
#                        optimizer: Optimizer,
#                        loss_fun: Loss,
#                        ):
#             return 1
#
#     my_trainer = Trainer(torch.optim.Adam([torch.tensor([1, 2])], 1e-3))
#     my_trainer.train(DiffModelStub(), )


if __name__ == "__main__":
    pass
    # test_train()
