import torch
from matplotlib import pyplot as plt


def test_noise():
    from torchvision.transforms import ToPILImage
    import numpy as np

    total_steps = 1000
    model = default_model()
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
    model = default_model()
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
    from scripts.import_dataset import load_data
    from torch.nn.functional import mse_loss
    from scripts.DiffusionModel import default_model
    from scripts.Unet import Generator
    from scripts.utils import default_device

    test_noise()
    test_trainstep()
