import torch

from scripts.Unet_valeria import UNet


def test_random_image():
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=1)
    data = torch.rand(1, 3, 572, 572)
    output = net(data)
    print("Shape of output: ", output.shape)
    print("UNet number of param: ", sum(p.numel() for p in net.parameters() if
                                        p.requires_grad))  # 31037698 aka 31 milioni di trainable parameters


if __name__ == "__main__":
    test_random_image()
