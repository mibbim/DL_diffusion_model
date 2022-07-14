from torchvision.datasets import MNIST
from dataset import CSSDdataset

from torchvision import transforms
from torch.utils.data import DataLoader, Subset
import requests, zipfile, io, os
import torch
import torchvision.transforms as T


# ratio_data to define the ratio of the total number of sample for train and test
# e.g. if ratio_data=100, there will (total_number_sample_mnist / 100) number of sample
def load_MNIST(train_batch_size, test_batch_size, ratio_data=1, verbose=False):
    """
    :param train_batch_size:
    :param test_batch_size: 
    :param ratio_data: 
    :param verbose: if True, print some information about the dataset
    :return:

    Iterating over the result DataLoader gives a (a, b, c, d) tensor.
    - a is the batch size
    - b is the channel of the image
    - c, d are the image pixels
    """
    data_train_full = MNIST(root='', train=True,
                            transform=transforms.ToTensor(),
                            download=True)
    data_test_full = MNIST(root='', train=False,
                           transform=transforms.ToTensor(),
                           download=True)

    print(type(data_test_full))
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


def load_data_CSSD(train_batch_size, test_batch_size, ratio_test=0.20, verbose=False):
    """
    :param train_batch_size:
    :param test_batch_size:
    :param ratio_test: ratio of the total number of sample for train and test
    :param verbose: if True, print some information about the dataset
    :return:

    """

    if not os.path.exists('./CSSD'):
        dataset_url = "http://www.cse.cuhk.edu.hk/leojia/projects/hsaliency/data/CSSD/images.zip"
        r = requests.get(dataset_url)
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall("./CSSD")

    train_data = CSSDdataset("./CSSD/images", transform=transforms.ToTensor(), test=False,
                             ratio_test=ratio_test)
    test_data = CSSDdataset("./CSSD/images", transform=transforms.ToTensor(), test=True,
                            ratio_test=ratio_test)
    # if verbose:
    #     print(data)

    train_loader = DataLoader(train_data, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=test_batch_size, shuffle=True)

    return train_loader, test_loader

# How to use: 
# data_train, data_test = load_data(train_batch_size=32, test_batch_size=32, ratio_data=100)


def test():

    torch.manual_seed(8)
    # train_loader, test_loader = load_MNIST(1, 1, 10000, verbose=True)  # [1, 1, 28, 28] 1 is the number of img in the batch, 1 is the number of channel, 28x28 pixels
    train_loader, test_loader = load_data_CSSD(1, 1, 0.20, verbose=True) 
    x = next(iter(train_loader))
    print(f"x is a {type(x)} of shape {x.shape}") 
    img = T.ToPILImage()(x[0])
    img.show()
    pass


if __name__ == "__main__":
    test()

# >>>>>>> new_data
