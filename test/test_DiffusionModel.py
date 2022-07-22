import torch
from matplotlib import pyplot as plt


def test_noise():
    from scripts.variance_schedule import LinearVarianceSchedule
    import numpy as np

    total_steps = 1000
    noise_generator = NoiseGenerator(
        beta=LinearVarianceSchedule(steps=total_steps)
    )
    torch.manual_seed(8)
    train, _ = load_data_CIFAR10(1, 1, 1000)
    x, y = next(iter(train))
    img = x[0]
    for t in np.geomspace(1, total_steps - 1, 10):
        x_t, noise, t = noise_generator.add_noise(
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
    """Removed method"""
    torch.manual_seed(8)
    model = default_model()
    model.train()
    train, _ = load_data_CIFAR10(2, 1, 1000)
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


def test_generate():
    torch.manual_seed(8)
    model = default_model()
    res = model.generate(3)
    ToPILImage()(res[0]).show()


# Feature to be fixed
# def test_fwd_and_bkwd():
#     train, _ = load_data_CIFAR10(2, 1, 1000)
#     x, _ = next(iter(train))
#     model = default_model()
#     x, noisy, denoised = model.forward_and_backward_img(x)
#     img = torch.cat((x, noisy, denoised))
#     ToPILImage()(img).show()


if __name__ == "__main__":
    from torchvision.transforms import ToPILImage
    from scripts.import_dataset import load_data_CIFAR10
    from torch.nn.functional import mse_loss
    from scripts.DiffusionModel import default_model, NoiseGenerator
    from scripts.utils import default_device

    test_generate()
    test_noise()
    # test_trainstep()
    # test_fwd_and_bkwd()
