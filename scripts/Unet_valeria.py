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
    def __init__(self, activation_type: ActivationType = "ReLU", alpha= None):
        super().__init__()
                # initialize alpha parameter for ParametricReLU and ELU
        if activation_type == Literal["ReLU", "LeakyReLU", "SiLU", "none"]:
            self.alpha = Parameter(torch.tensor(0.0)) # create a tensor out of alpha
        else:
            self.alpha = Parameter(torch.rand(1)) # create a tensor
            
        self.alpha.requiresGrad = True # set requiresGrad to true!

        if activation_type == 'ReLU':
            self.activation = nn.ReLU()
        elif activation_type == 'LeakyReLU':
            self.activation = nn.LeakyReLU() # 0.01 default parameter
        elif activation_type == 'PReLU':
            self.activation = nn.PReLU(init=alpha)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ELU': # Exponential Linear Unit (computationally more expensive than ReLU)
            self.activation = nn.ELU(alpha=alpha)
        elif activation_type == 'none':
            self.activation = lambda x: x
        else:
            raise ValueError('Unknown activation type')
        

    def forward(self, x):
        return self.activation(x)

class ConvBlockDownsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type: ActivationType = "ReLU", alpha= None):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3), #padding and stride are set to 1, no bias
            nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
            ActivationFunc(activation_type, alpha),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type, alpha)
        )
        self.downsample = nn.MaxPool2d(kernel_size=2)
        
    
    def forward(self, data):
        out = self.conv(data)
        out2 = self.downsample(out)
        return out, out2


class ConvBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels, activation_type: ActivationType = "ReLU", alpha= None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type, alpha),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type, alpha)
        )
        
    
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
        


if __name__ == "__main__":
    block = ConvBlockDownsample(1, 64, "ELU", 0.3)
    output_for_upsample, output = block(torch.rand(1, 1, 28, 28)) #batch, channels, size, size
    print("Shape of output: ", output.shape) #([1, 64, 12, 12])
    print("Shape of output for upsample: ", output_for_upsample.shape) #([1, 64, 24, 24])

    block2 = ConvBlockUpsample(128, 64)
    data_before_upsample = torch.rand((1, 128, 12, 12))
    # we do the forward pass by output_for_upsample from before
    output2 = block2(data_before_upsample, output_for_upsample)
    print("Shape of output2: ", output2.shape) #([1, 64, 20, 20])