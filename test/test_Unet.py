import torch

from scripts.aiaiart_unet import UNet
from scripts.utils import default_device
from scripts.import_dataset import load_data_CIFAR10


def test_random_image():
    img_channels = 1
    img_size = 32
    x = torch.randn((2, img_channels, img_size, img_size))
    unet = UNet(img_channels, 32)
    t = torch.tensor([1, 23], dtype=torch.long, device=x.device)
    assert unet(x, t).shape == x.shape


def test_cifar():
    img_channels = 3

    torch.manual_seed(8)
    train, _ = load_data_CIFAR10(5, 1, 1000)
    unet = UNet(img_channels, 32)

    for x, y in train:
        t = torch.tensor([1, 23, 12, 1, 1], dtype=torch.long, device=x.device)
        in_shape = x.shape
        out_shape = unet(x, t).shape
        assert (in_shape == out_shape)


def test_device(device=default_device):
    img_channels = 3
    print(f"running on {device}")
    unet = UNet(img_channels, 32).to(device)
    for p in unet.parameters():
        try:
            assert p.device == device
        except AssertionError:
            print(p.device, device)


if __name__ == "__main__":
    test_random_image()
    test_cifar()
    test_device()
    if torch.cuda.is_available():
        test_device(torch.device("cpu"))
