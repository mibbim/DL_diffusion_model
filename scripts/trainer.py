"""
This file contains the trainer class
"""

from __future__ import annotations

from typing import Literal, Dict, Tuple

from tqdm.auto import trange
from pathlib import Path

import torch.optim
from torch.nn.functional import mse_loss
from torch.utils.data import DataLoader

from .DiffusionModel import DiffusionModel, default_model
from .DiffusionModel import Loss
from .performance_meter import AverageMeter
from .utils import default_device, Device

from torch.utils.tensorboard import SummaryWriter
from torch import Tensor

optimizers_dict = {"Adam": torch.optim.Adam,
                   "AdamW": torch.optim.AdamW}
script_dir = Path(__file__).resolve().parent


class MetricDumper:
    def __init__(self,
                 log_dir: Path | None = None,
                 # tb_writer: SummaryWriter | None = SummaryWriter(),
                 ):
        self.tb_writer = SummaryWriter(log_dir=str(log_dir / "tb"))

    def dump_scalars(self, scalar_dict: Dict[str, Tuple[int, float]]) -> None:
        """
        :param scalar_dict: {scalar_tag : (step, value)}
        """
        for k, (step, v) in scalar_dict.items():
            self.tb_writer.add_scalar(tag=k, scalar_value=v, global_step=step)

    def dump_image(self, image_dict: Dict[str, Tuple[int, Tensor]]) -> None:
        """
        :param image_dict: {image_tag : (step, image)}
        """
        for k, (step, v) in image_dict.items():
            self.tb_writer.add_image(tag=k, img_tensor=v, global_step=step)


class Trainer:
    def __init__(self,
                 model: DiffusionModel = default_model(),
                 optimizer: Literal["Adam", "AdamW"] | None = "AdamW",
                 learning_rate: float | None = 2e-4,
                 metric_dumper: MetricDumper | None = None,
                 device: torch.device | None = default_device,
                 out_path: Path | None = None
                 ) -> None:
        self.out_path = out_path
        if self.out_path is not None:
            self.out_path.resolve().mkdir(parents=True, exist_ok=True)

        if metric_dumper is None:
            metric_dumper = MetricDumper(log_dir=self.out_path)
        self.device = device
        self.model = model.to(device)
        self.opt = optimizers_dict[optimizer](params=model.parameters(), lr=learning_rate)
        self.history = {"train_loss": [],
                        "val_loss": []}
        self.dumper = metric_dumper

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
            for data in train_dataloader:
                if isinstance(data, (tuple, list)):
                    x, y = data
                else:
                    x = data
                x = x.to(device)  # Batch of Images
                xt, noise, t = self.model._noise_generator.add_noise(x)
                pred_noise = self.model.predict_noise(xt, t)
                # loss = self.model.train_step(x, self.opt, loss_function)
                loss = loss_function(noise, pred_noise)
                average_meter.update({"train_mse": loss.item()}, n=x.shape[0])
                self.opt.zero_grad()
                loss.backward()
                self.opt.step()

            self.history["train_loss"].append((epoch, average_meter.metrics["train_mse"]["avg"]))
            average_meter.reset()
            if epoch % valid_each == 0:
                self.model.eval()
                self.validate(val_dataloader, validation_metric, epoch)
                self.model.train()

        self.model.eval()
        self.validate(val_dataloader, validation_metric, n_epochs)

    def _clear_history(self):
        self.history = {k: [] for k in self.history}

    def dump_history(self):
        for k, v_list in self.history.items():
            for epoch_and_val in v_list:
                self.dumper.dump_scalars({k: epoch_and_val})

    def _eval_val_loss(self,
                       data_loader: DataLoader,
                       validation_metric: Loss,
                       epoch: int):
        validation_meter = AverageMeter(["val_mse"])
        best_loss = torch.inf
        for data in data_loader:
            if isinstance(data, (tuple, list)):
                x, y = data
            else:
                x = data
            x = x.to(self.device)
            val_loss = self.model.val_step(x, validation_metric)
            if val_loss.item() < best_loss:
                self.store_state(epoch)
            validation_meter.update({"val_mse": val_loss.item()}, n=x.shape[0])
        return validation_meter

    def _eval_img_reconstruction(self,
                                 data_loader: DataLoader,
                                 epoch: int):
        x, _ = next(iter(data_loader))
        x, noisy, denoised = self.model.forward_and_backward_img(x)

    def validate(self, data_loader: DataLoader, validation_metric: Loss, epoch: int):
        # validation_meter = AverageMeter(["val_mse"])
        # best_loss = torch.inf
        # for x, y in data_loader:
        #     x = x.to(self.device)
        #     val_loss = self.model.val_step(x, validation_metric)
        #     if val_loss.item() < best_loss:
        #         self.store_state(epoch)
        #     validation_meter.update({"val_mse": val_loss.item()}, n=x.shape[0])
        validation_meter = self._eval_val_loss(data_loader, validation_metric, epoch)
        self.history["val_loss"].append((epoch, validation_meter.metrics["val_mse"]["avg"]))
        self.dump_history()
        self._clear_history()
        validation_meter.reset()

    def store_state(self, epoch):
        checkpoint_dict = {
            "parameters": self.model.state_dict(),
            "optimizer": self.opt.state_dict(),
            "epoch": epoch,
        }
        torch.save(checkpoint_dict, self.out_path / "checkpoint.pt")

    def load_state(self, file: Path | None = None):
        if file is None:
            file = self.out_path / "checkpoint.pt"
        checkpoint_dict = torch.load(file)
        self.model.load_state_dict(checkpoint_dict["parameters"])
        self.opt.load_state_dict(checkpoint_dict["optimizer"])
