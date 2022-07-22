from datetime import datetime
from pathlib import Path

import torch
from torchvision.transforms import ToPILImage

from scripts.DiffusionModel import DiffusionModel
from scripts.Unet_valeria import UNet as UNet_valeria
from scripts.aiaiart_unet import UNet as Ai_unet
from scripts.noiseGenerator import NoiseGenerator
from scripts.trainer import Trainer
from scripts.import_dataset import load_data_CIFAR10
from scripts.utils import default_device
from scripts.variance_schedule import LinearVarianceSchedule, CosineVarianceSchedule

script_dir = Path(__file__).resolve().parent

device = default_device

n_steps = 100
beta_min, beta_max = 0.0001, 0.04,
beta = torch.linspace(beta_min, beta_max, n_steps).to(device)
alpha = (1. - beta).to(device)
alpha_bar = torch.cumprod(alpha, dim=0).to(device)


def test_train_with_other_unet():
    """
    A template for training a DiffusionModel with a Trainer
    """
    out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(
        now=datetime.now())
    noise_generator = NoiseGenerator(beta=LinearVarianceSchedule(beta_min,
                                                                 beta_max,
                                                                 1000,
                                                                 device=device))
    model = DiffusionModel(
        noise_predictor=Ai_unet(n_channels=32).to(device),
        noise_generator=NoiseGenerator(beta=CosineVarianceSchedule(n_steps,
                                                                   device=device)),
    )
    trainer = Trainer(model=model,
                      optimizer="AdamW",
                      learning_rate=2e-4,
                      out_path=out_path)

    train_loader, test_loader = load_data_CIFAR10(train_batch_size=32,
                                                  test_batch_size=256,
                                                  ratio_data=100, )
    trainer.train(n_epochs=10,
                  train_dataloader=train_loader,
                  val_dataloader=test_loader,
                  valid_each=2,
                  )

    x, y = next(iter(test_loader))
    x = x.to(device)
    noised_x, noise, t = model._noise_generator.add_noise(x, torch.tensor(99, dtype=torch.long,
                                                                          device=device))
    img = torch.cat((x[0], noised_x[0], model.generate_from(noised_x[:1])[0]), dim=2)
    ToPILImage()(img).show()
    noise = torch.randn_like(x[:1])
    img = torch.cat((noise[0], model.generate_from(noise)[0]), dim=2)
    ToPILImage()(img).show()


def test_train_Unet_valeria(
        n_epochs=10,
        valid_each=1,
        train_batch_size=32,
        test_batch_size=32,
        ratio_data=100,
        n_time_steps=n_steps,
        learning_rate=2e-4,
        dropout=None,
        n_conv_filters=64):
    out_path = script_dir / "out" / "{now:%Y-%m-%d}_{now:%H:%M:%S}".format(
        now=datetime.now())
    noise_generator = NoiseGenerator(beta=LinearVarianceSchedule(beta_min,
                                                                 beta_max,
                                                                 n_steps,
                                                                 device=device))
    model = DiffusionModel(noise_predictor=UNet_valeria(n_classes=3, dropout=dropout,
                                                        n_conv_filters=n_conv_filters).to(device),
                           # is_attn=(False, False, True), dropout=0.1
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
    img = torch.cat((x[0], noised_x[0], model.generate_from(noised_x)[0]), dim=2)
    pil = ToPILImage()(img)
    pil.show()
    pil.save(out_path / "fwd_bkwd.jpg")
    noise = torch.randn_like(x)
    img = torch.cat((noise[0], model.generate_from(noise)[0]), dim=2)
    pil = ToPILImage()(img)
    pil.show()
    pil.save(out_path / "from_noise.jpg")


if __name__ == "__main__":
    test_train_with_other_unet()
    # test_train_Unet_valeria()
