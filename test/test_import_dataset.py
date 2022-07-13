from scripts.import_dataset import load_MNIST


def test_import():
    import torchvision.transforms as T
    import torch

    torch.manual_seed(8)
    train_loader, test_loader = load_MNIST(1, 1, 10000, verbose=True)  # load
    x, y = next(iter(train_loader))
    # [1, 1, 28, 28] 1 is the number of img in the batch, 1 is the number of channel, 28x28 pixels
    print(f"x is a {type(x)} of shape {x.shape}")
    print(f"y is a {type(y)} of shape {y.shape}")
    img = T.ToPILImage()(x[0])
    img.show()
    pass


if __name__ == "__main__":
    test_import()
