# import sys
# sys.path.append('../')

from scripts.import_dataset import load_MNIST
from scripts.import_dataset import load_data_CSSD


def test_import():
    import torchvision.transforms as T
    import torch

    torch.manual_seed(8)
    # train_loader, test_loader = load_MNIST(1, 1, 10000, verbose=True)  # [1, 1, 28, 28] 1 is the number of img in the batch, 1 is the number of channel, 28x28 pixels
    train_loader, test_loader = load_data_CSSD(1, 1, 0.20, verbose=True) 
    x = next(iter(train_loader))
    print(f"x is a {type(x)} of shape {x.shape}") 
    img = T.ToPILImage()(x[0])
    img.show()
    pass



if __name__ == "__main__":
    test_import()
