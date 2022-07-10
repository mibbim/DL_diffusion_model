"""
This file contains the implementation of a U-net module,
used in DiffusionModel to predict the noise
"""

import torch
import torch.nn as nn


class Generator(nn.Module):
    """
    This implementation is only temporary, used to test the other units.
    The original authors are Davide Basso and Andres Bermeo Marinelli, taken from
    their GAN Project on github:
    https://github.com/dbasso98/GANs/blob/main/CycleGANs/generator.py

    """

    def __init__(self, in_channels=3, output_nc=3, n_filter=64, n_blocks=9, use_dropout=False):

        assert (n_blocks >= 0)
        super(Generator, self).__init__()
        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(in_channels, n_filter, kernel_size=7, padding=0),
                 nn.InstanceNorm2d(n_filter),
                 nn.ReLU(True)]

        layers = 2
        for i in range(layers):  # add downsampling layers
            mult = 2 ** i
            model += [
                nn.Conv2d(n_filter * mult, n_filter * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(n_filter * mult * 2),
                nn.ReLU(True)]

        mult = 2 ** layers
        for i in range(n_blocks):  # add ResNet blocks

            model += [ResnetBlock(n_filter * mult, use_dropout=use_dropout)]

        for i in range(layers):  # add upsampling layers
            mult = 2 ** (layers - i)
            model += [nn.ConvTranspose2d(n_filter * mult, int(n_filter * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1),
                      nn.InstanceNorm2d(int(n_filter * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(n_filter, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input, t):
        return self.model(input)


class ResnetBlock(nn.Module):

    def __init__(self, dim, use_dropout):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, use_dropout)

    def build_conv_block(self, dim, use_dropout):
        conv_block = []
        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0), nn.InstanceNorm2d(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        conv_block += [nn.ReflectionPad2d(1)]

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=0), nn.InstanceNorm2d(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


def test_random_image():
    img_channels = 1
    img_size = 28
    x = torch.randn((2, img_channels, img_size, img_size))
    gen = Generator(img_channels, img_channels)
    print(gen(x).shape)


def test_mnist():
    img_channels = 1
    img_size = 28

    from import_dataset import load_data
    torch.manual_seed(8)
    train, _ = load_data(5, 1, 1000)
    gen = Generator(img_channels, img_channels)

    for x, y in train:
        in_shape = x.shape
        out_shape = gen(x).shape
        print(f"{in_shape=}, {out_shape=}")


if __name__ == "__main__":
    test_random_image()
    test_mnist()
