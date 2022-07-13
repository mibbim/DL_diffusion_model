import torch

from scripts.Unet import Generator
from scripts.utils import default_device
from scripts.import_dataset import load_MNIST


def test_random_image():
    img_channels = 1
    img_size = 28
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, img_channels)
    print(gen(x, 1).shape)


def test_mnist():
    img_channels = 1

    torch.manual_seed(8)
    train, _ = load_MNIST(5, 1, 1000)
    gen = Generator(img_channels, img_channels)

    for x, y in train:
        in_shape = x.shape
        out_shape = gen(x, 1).shape
        assert (in_shape == out_shape)


def test_device(device=default_device):
    img_channels = 1
    print(f"running on {device}")
    gen = Generator(img_channels, img_channels).to(device)
    for p in gen.parameters():
        try:
            assert p.device == device
        except AssertionError:
            print(p.device, device)


if __name__ == "__main__":
    test_random_image()
    test_mnist()
    test_device()
    if torch.cuda.is_available():
        test_device(torch.device("cpu"))
