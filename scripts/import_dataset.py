from torchvision.datasets import MNIST
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset


# ratio_data to define the ratio of the total number of sample for train and test
# e.g. if ratio_data=100, there will (total_number_sample_mnist / 100) number of sample
def load_data(train_batch_size, test_batch_size, ratio_data=1, verbose=False):
    data_train_full = MNIST(root='', train=True,
                            transform=transforms.ToTensor(),
                            download=True)
    data_test_full = MNIST(root='', train=False,
                           transform=transforms.ToTensor(),
                           download=True)

    # Take only one part of the dataset
    data_train_less = Subset(data_train_full, range(0, len(data_train_full) // ratio_data))
    data_test_less = Subset(data_test_full, range(0, len(data_test_full) // ratio_data))

    if verbose:
        # Only to study, then we will remove this part
        print("Number of element of train: ", len(data_train_less))
        print("Number of element of test: ", len(data_test_less))
        print("Type of one image: ", type(data_train_less[0][0]))
        print("Shape of an image: ", data_train_less[0][0].shape)

    train_loader = DataLoader(data_train_less, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(data_test_less, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader


# How to use: 
# data_train, data_test = load_data(train_batch_size=32, test_batch_size=32, ratio_data=100)

# Every call to the dataset iterator will return batch of images of size batch_size
# The main difference to work with DataLoader is that this is non-indexable (no dataset[6])
def test():
    import torchvision.transforms as T
    import torch

    torch.manual_seed(8)
    train_loader, test_loader = load_data(1, 1, 10000, verbose=True)  # load
    for x, y in train_loader:
        print(f"x is a {type(x)} of shape {x.shape}")
        print(f"y is a {type(y)} of shape {y.shape}")
        img = T.ToPILImage()(x[0])
        img.show()
        pass


if __name__ == "__main__":
    test()

