import torch

from scripts.Unet_valeria import UNet
#from scripts.utils import default_device


def test_random_image():
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=1)
    data = torch.rand(1, 3, 572, 572)
    output = net(data)
    print("Shape of output: ", output.shape)
    print("UNet number of param: ", sum(p.numel() for p in net.parameters() if p.requires_grad)) #31037698 aka 31 milioni di trainable parameters


# def test_mnist():
#     img_channels = 1

#     torch.manual_seed(8)
#     train, _ = load_data(5, 1, 1000)
#     gen = Generator(img_channels, img_channels)

#     for x, y in train:
#         in_shape = x.shape
#         out_shape = gen(x, 1).shape
#         assert (in_shape == out_shape)


# def test_device(device=default_device):
#     img_channels = 1
#     print(f"running on {device}")
#     gen = Generator(img_channels, img_channels).to(device)
#     for p in gen.parameters():
#         try:
#             assert p.device == device
#         except AssertionError:
#             print(p.device, device)


if __name__ == "__main__":
    test_random_image()
    # test_mnist()
    # test_device()
    # if torch.cuda.is_available():
    #     test_device(torch.device("cpu"))
