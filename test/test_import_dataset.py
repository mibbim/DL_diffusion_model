# python -m test_import_dataset

from scripts.utils.import_dataset import load_data_CIFAR10


def test_import():
    import torchvision.transforms as T
    import torch

    torch.manual_seed(8)
    train_loader, test_loader = load_data_CIFAR10(1, 1, 100, verbose=True)  # 139 * 139 * 3
    x, y = next(iter(train_loader))
    print(f"x is a {type(x)} of shape {x.shape}")
    img = T.ToPILImage()(x[0])
    img.show()
    pass


if __name__ == "__main__":
    test_import()
