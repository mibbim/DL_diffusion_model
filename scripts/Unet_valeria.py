"""
This file contains the implementation of Valeria's U-net module,
used in DiffusionModel to predict the noise
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal
from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters

ActivationType = Literal["ReLU", "LeakyReLU", "PReLU", "SiLU", "ELU", "none"]

# Custom Activation function
class ActivationFunc(nn.Module):
    """
    ### Custom actiavation function to choose among ReLU, LeakyReLU, ParametricReLU, Sigmoid Linear Unit, identity.
    """
    def __init__(self, activation_type: ActivationType = "ReLU"):
        super().__init__()
        # # initialize alpha parameter for ParametricReLU and ELU
        # if activation_type == Literal[ "PReLU", "ELU"]:
        #     self.alpha = Parameter(torch.rand(alpha)) # create a tensor out of alpha
        # else:
        #     self.alpha = Parameter(torch.tensor(0.0)) # create a tensor empty
        # self.alpha.requiresGrad = True # set requiresGrad to true!

        if activation_type == 'ReLU':
            self.activation = nn.ReLU() # Should I use inplace=True?
        elif activation_type == 'LeakyReLU':
            self.activation = nn.LeakyReLU() # 0.01 default parameter
        elif activation_type == 'PReLU': # to access the alpha parameter learnt use activation.weight
            self.activation = nn.PReLU()
            #self.activation = nn.PReLU(init=alpha)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ELU': # Exponential Linear Unit (computationally more expensive than ReLU)
            self.activation = nn.ELU()
            #self.activation = nn.ELU(alpha=alpha)
        elif activation_type == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError('Unknown activation type')
        

    def forward(self, x):
        return self.activation(x)

# Double convolutional layer for Unet
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type: ActivationType = "ReLU"):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False), # stride=1,padding=0,bias set false since usage of BatchNorm
            nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
            ActivationFunc(activation_type),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type)
        )
    def forward(self, data):
        return self.conv(data)

# Module for downsampling
class ConvBlockDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = ConvBlock(in_channels,out_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2) #the stride default value is kernel_size
        
    
    def forward(self, data):
        out_for_upsample = self.conv(data)
        out = self.downsample(out_for_upsample)
        return out, out_for_upsample

# Module for upsampling
class ConvBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels,out_channels)
        
    
    def forward(self, data, data_from_downsample:torch.Tensor):
        out = self.upsample(data)
        # data has dimension ch x h x w
        # data_from_downsample has dimension ch x H x W
        # with H > h, W > w
        h, w = out.shape[2], out.shape[3]
        H, W = data_from_downsample.shape[2], data_from_downsample.shape[3]
        # do a center crop of data_from_downsample 
        # (starting from H//2, W//2, the center pixel of the larger image)
        cropped_data_from_downsample = data_from_downsample[:, :, H//2-h//2 : H//2+(h//2 + h%2), W//2-w//2 : W//2+(w//2 + w%2)]
        out = torch.cat([out, cropped_data_from_downsample], dim=1)
        return self.conv(out)


class OutConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    def __init__(self, n_channels, n_classes):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = ConvBlockDownsample(n_channels, 64)
        self.down2 = ConvBlockDownsample(64, 128)
        self.down3 = ConvBlockDownsample(128, 256)
        self.down4 = ConvBlockDownsample(256, 512)
        self.ground = ConvBlock(512, 1024)
        self.up1 = ConvBlockUpsample(1024, 512)
        self.up2 = ConvBlockUpsample(512, 256)
        self.up3 = ConvBlockUpsample(256, 128)
        self.up4 = ConvBlockUpsample(128, 64)
        self.outc = OutConv(64, n_classes)

    def forward(self, x):
        x1, up1 = self.down1(x)
        x2, up2 = self.down2(x1)
        x3, up3 = self.down3(x2)
        x4, up4 = self.down4(x3)
        x5 = self.ground(x4)
        x = self.up1(x5, up4)
        x = self.up2(x, up3)
        x = self.up3(x, up2)
        x = self.up4(x, up1)
        logits = self.outc(x)
        return logits

if __name__ == "__main__":
    # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=2)
    data = torch.rand(1, 3, 572, 572)
    output = net(data)
    print("Shape of output: ", output.shape)

    # state_dict stores both parameters and persistent buffers (e.g., BatchNorm's running mean and var). 
    # There's no way (AFAIK) to tell them apart from the state_dict itself, 
    # you'll need to load them into the model and use sum(p.numel() for p in model.parameters() to count only the parameters.
    print("UNet number of param: ", sum(p.numel() for p in net.parameters() if p.requires_grad)) #31037698 aka 31 milioni di trainable parameters