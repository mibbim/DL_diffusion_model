from datetime import datetime
from pathlib import Path

import torch
from torchvision.transforms import ToPILImage

from scripts.DiffusionModel import DiffusionModel
from scripts.Unet_valeria import UNet as UNet_valeria
from scripts.aiaiart_unet import UNet as Ai_unet
from scripts.noiseGenerator import NoiseGenerator
from scripts.trainer import Trainer
from scripts.import_dataset import load_data_CSSD, load_data_CIFAR10
from scripts.utils import default_device
from scripts.variance_schedule import LinearVarianceSchedule

script_dir = Path(__file__).resolve().parent

device = default_device

n_steps = 100
beta_min, beta_max = 0.0001, 0.04,
beta = torch.linspace(beta_min, beta_max, n_steps).to(device)
alpha = (1. - beta).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device)


def test_train():
    """
    A template for traininf a DiffusionModel with a Trainer
    """
    out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(
        now=datetime.now())
    noise_generator = NoiseGenerator(beta=LinearVarianceSchedule(beta_min,
                                                                 beta_max,
                                                                 n_steps,
                                                                 device=device))
    model = DiffusionModel(noise_predictor=Ai_unet(n_channels=32).to(device),
                           noise_generator=noise_generator,
                           )
    trainer = Trainer(model=model,
                      optimizer="AdamW",
                      learning_rate=2e-4,
                      out_path=out_path)

    train_loader, test_loader = load_data_CIFAR10(train_batch_size=32,
                                                  test_batch_size=32,
                                                  ratio_data=100, )
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=1,
                  )

    x, y = next(iter(test_loader))
    x = x.to(device)
    noised_x, noise, t = model._noise_generator.add_noise(x, torch.tensor(99, dtype=torch.long,
                                                                          device=device))
    img = model.generate_from(noised_x)[0]
    ToPILImage()(img).show()


def test_train_Unet_valeria(n_epochs=10,valid_each=1,train_batch_size=32,test_batch_size=32, ratio_data=100, n_time_steps = n_steps, learning_rate=2e-4, dropout=None):
    # out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(now=datetime.now())
    # trainer = Trainer(model=DiffusionModel(noise_predictor=UNet(3, 3)),
    #                   out_path=out_path,
    #                   )
    # train_loader, test_loader = load_data_CSSD(train_batch_size=64,
    #                                            test_batch_size=128,
    #                                            ratio_test=0.2)
    # trainer.train(n_epochs=50,
    #               train_dataloader=train_loader,
    #               val_dataloader=test_loader,
    #               valid_each=5,
    #               )
    out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(
        now=datetime.now())
    noise_generator = NoiseGenerator(beta=LinearVarianceSchedule(beta_min,
                                                                 beta_max,
                                                                 n_time_steps,
                                                                 device=device))
    model = DiffusionModel(noise_predictor=UNet_valeria(n_classes=3, dropout=dropout).to(device), #is_attn=(False, False, True), dropout=0.1
                           noise_generator=noise_generator,
                           )
    trainer = Trainer(model=model,
                      optimizer="AdamW",
                      learning_rate=learning_rate,
                      out_path=out_path)

    train_loader, test_loader = load_data_CIFAR10(train_batch_size=train_batch_size,
                                                  test_batch_size=test_batch_size,
                                                  ratio_data=ratio_data, )
    trainer.train(n_epochs=n_epochs,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=valid_each,
                  )

    x, y = next(iter(test_loader))
    x = x.to(device)
    noised_x, noise, t = model._noise_generator.add_noise(x, torch.tensor(99, dtype=torch.long,
                                                                          device=device))
    img = model.generate_from(noised_x)[0]
    ToPILImage()(img).show()

    # for m in trainer.history.keys():
    #     plot_single(trainer.history, m)
    # plt.legend()
    # plt.show()


if __name__ == "__main__":
    #test_train()
    test_train_Unet_valeria()
