"""
This file contains the implementation of a U-net module,
used in DiffusionModel to predict the noise
"""
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
        model = nn.ModuleList([nn.ReflectionPad2d(3),
                               nn.Conv2d(in_channels, n_filter, kernel_size=7, padding=0),
                               nn.InstanceNorm2d(n_filter),
                               nn.ReLU(True)])

        layers = 2
        for i in range(layers):  # add downsampling layers
            mult = 2 ** i
            model += nn.ModuleList([
                nn.Conv2d(n_filter * mult, n_filter * mult * 2, kernel_size=3, stride=2, padding=1),
                nn.InstanceNorm2d(n_filter * mult * 2),
                nn.ReLU(True)])

        mult = 2 ** layers
        for i in range(n_blocks):  # add ResNet blocks 0 1 2 3 4 5 6 7 8 

            model += nn.ModuleList([ResnetBlock(n_filter * mult, use_dropout=use_dropout)])

        for i in range(layers):  # add upsampling layers
            mult = 2 ** (layers - i)
            model += nn.ModuleList([nn.ConvTranspose2d(n_filter * mult, int(n_filter * mult / 2),
                                                       kernel_size=3, stride=2,
                                                       padding=1, output_padding=1),
                                    nn.InstanceNorm2d(int(n_filter * mult / 2)),
                                    nn.ReLU(True)])
        model += nn.ModuleList([nn.ReflectionPad2d(3)])
        model += nn.ModuleList([nn.Conv2d(n_filter, output_nc, kernel_size=7, padding=0)])
        model += nn.ModuleList([nn.Tanh()])

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
