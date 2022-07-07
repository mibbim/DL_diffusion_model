from torchvision.datasets import MNIST
import matplotlib.pyplot as plot
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
import matplotlib.pyplot as plot

# The main difference to work with DataLoader is that this is non-indexable (no dataset[6])
def load_data(train_batch_size, test_batch_size, ratio_data = 1):
    data_train_full = MNIST(root='', train=True, 
                        transform=transforms.ToTensor(), 
                        download=True)
    data_test_full = MNIST(root='', train=False, 
                        transform=transforms.ToTensor(), 
                        download=True)

    # Take only one part of the dataset
    data_train_less = Subset(data_train_full, range(0, len(data_train_full) // ratio_data))
    data_test_less = Subset(data_test_full, range(0, len(data_test_full)// ratio_data))

    # Only to study, then we will remove this part
    print("Number of element of train: ", len(data_train_less))
    print("Number of element of test: ", len(data_test_less))
    print("Type of one image: ", type(data_train_less[0][0]))
    print("Shape of an image: ", data_train_less[0][0].shape)

    
    train_loader = DataLoader(data_train_less, batch_size=train_batch_size, shuffle=True)
    test_loader = DataLoader(data_test_less, batch_size=test_batch_size, shuffle=True)
    
    return train_loader, test_loader


# How to use: 
data_train, data_test = load_data(train_batch_size=32, test_batch_size=32, ratio_data=100)
