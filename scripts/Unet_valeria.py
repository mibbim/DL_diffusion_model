"""
This file contains the implementation of Valeria's U-net module,
used in DiffusionModel to predict the noise
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal
#from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import math

ActivationType = Literal["ReLU", "LeakyReLU", "PReLU", "SiLU", "ELU", "none"]

# Custom Activation function
class ActivationFunc(nn.Module):
    """
    ### Custom actiavation function to choose among:
    #### ReLU, LeakyReLU, ParametricReLU, Sigmoid Linear Unit, identity.
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

# Time Embedding position encoder
class SinusoidalPositionEmbeddings(nn.Module):
    # \begin{align}
    # PE^{(1)}_{t,i} &= sin\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg) \\
    # PE^{(2)}_{t,i} &= cos\Bigg(\frac{t}{10000^{\frac{i}{d - 1}}}\Bigg)
    # \end{align}
    #
    # where $d$ is `half_dim`
    """
    ### Sinusoidal position embeddings to encode time-step t, inspired by the Transformer (Vaswani et al., 2017).

    It takes a tensor of shape (batch_size, 1) as input (i.e. the noise levels of several noisy images in a batch), 
    and turns this into a tensor of shape (batch_size, dim), with dim being the dimensionality of the position embeddings. 
    This is then added to each residual block, as we will see further.

    """
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1) #Concatenates the given sequence of seq tensors in the given dimension: -1 is along last dimension
        return embeddings

# Double convolutional layer for Unet
class ConvBlock(nn.Module):
    """
    #### Double convolutional module with batch normalization
    Specify custom activation function to use, other than ReLU

    ----------
    #### Parameters
    in_channels: number
            the number of convolution filters.
    out_channels: number
            the number in output of channels.
    activation_type: ActivationType literal.
            optional choice for activation function.
    dropout: number, default is None.
            optional choice for dropout rate (USE 0.1).

    """
    def __init__(self, in_channels, out_channels, activation_type: ActivationType = "ReLU", dropout=None):
        super().__init__()

        layers = [nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False), # stride=1,padding=0,bias set false since usage of BatchNorm
            nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
            ActivationFunc(activation_type)]
        if dropout:
            layers.append(nn.Dropout(dropout))
        layers += [nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type)]

        self.conv = nn.Sequential(*layers)

        # self.conv = nn.Sequential(
        #     nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False), # stride=1,padding=0,bias set false since usage of BatchNorm
        #     nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
        #     ActivationFunc(activation_type),
        #     nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False),
        #     nn.BatchNorm2d(out_channels),
        #     ActivationFunc(activation_type)
        # )
        #nn.init.kaiming_uniform_(w, mode='fan_in', nonlinearity='relu')
    def forward(self, data):
        return self.conv(data)

# Module for downsampling
class ConvBlockDownsample(nn.Module):
    """
    ### Module for downsampling:
    #### Calls for ConvBlock and then performs downsampling

    ----------
    #### Return type
    out_for_upsample: 
            data passed only through ConvBlock that will be later used in Upsampling.
    out: 
            data already processed with max-polling, ready to be used for another downsampling layer.

    """
    def __init__(self, in_channels, out_channels, dropout=None):
        super().__init__()
        self.conv = ConvBlock(in_channels,out_channels, dropout=dropout)
        self.downsample = nn.MaxPool2d(kernel_size=2) #the stride default value is kernel_size!
        
    
    def forward(self, data):
        out_for_upsample = self.conv(data)
        if out_for_upsample.size(dim=2)%2 != 0:
            out_for_upsample = nn.ReplicationPad2d(0,1,0,1) #padding_left,padding_right,padding_top,padding_bottom
        out = self.downsample(out_for_upsample)
        return out, out_for_upsample

# Module for upsampling
class ConvBlockUpsample(nn.Module):
    """
    ### Module for upsampling:
    #### Calls for upsampling (center crop on data_from_downsample) and then performs ConvBlock
    ----------
    #### Parameters
        dropout: number, default is None
    """
    def __init__(self, in_channels, out_channels, dropout=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = ConvBlock(in_channels,out_channels, dropout=dropout)
        
    
    def forward(self, data, data_from_downsample:torch.Tensor):
        out = self.upsample(data)
        # data has dimension ch x h x w
        # data_from_downsample has dimension ch x H x W
        # with H > h, W > w
        h, w = out.shape[2], out.shape[3]
        if h%2 != 0:
            data = nn.ReplicationPad2d(0,1,0,1) #padding_left,padding_right,padding_top,padding_bottom
        H, W = data_from_downsample.shape[2], data_from_downsample.shape[3]
        # do a center crop of data_from_downsample 
        # (starting from H//2, W//2, the center pixel of the larger image)
        cropped_data_from_downsample = data_from_downsample[:, :, H//2-h//2 : H//2+(h//2 + h%2), W//2-w//2 : W//2+(w//2 + w%2)]
        out = torch.cat([out, cropped_data_from_downsample], dim=1)

        return self.conv(out)

#Final convolution layer with 1x1 filter
class OutConv(nn.Module):
    """
    ### Final Unet layer with conv 1x1:
    ----------
    #### Parameters
    in_channels: number
            the number of convolution filters.
    n_classes: number
            the number of probabilities you want to get per pixel (num of output image's channels)
    
    """
    def __init__(self, in_channels, out_channels):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    ### Unet network:
    ----------
    #### Parameters
    n_channels: number
            the number of convolution filters for start.
    n_classes: number, default is 1 (regression problem)
            the number of probabilities you want to get per pixel (num of output image's channels)
    dropout: number, default is None 
            Use 0.1 rate as paper https://arxiv.org/pdf/2006.11239.pdf suggested - implementation https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py)

    """
    def __init__(self, n_channels, n_classes=1, dropout=None):
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        self.down1 = ConvBlockDownsample(n_channels, 64, dropout=dropout)
        self.down2 = ConvBlockDownsample(64, 128, dropout=dropout)
        self.down3 = ConvBlockDownsample(128, 256, dropout=dropout)
        self.down4 = ConvBlockDownsample(256, 512, dropout=dropout)
        self.ground = ConvBlock(512, 1024)
        self.up1 = ConvBlockUpsample(1024, 512, dropout=dropout)
        self.up2 = ConvBlockUpsample(512, 256, dropout=dropout)
        self.up3 = ConvBlockUpsample(256, 128, dropout=dropout)
        self.up4 = ConvBlockUpsample(128, 64, dropout=dropout)
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
    # m = nn.MaxPool2d(kernel_size=2)
    # input = torch.rand(1, 1, 3, 3)
    # print(input)
    # output = m(input)
    # print(output.shape)
    # print(output)
    # SinusoidalPositionEmbeddings
    # # n_classes is the number of probabilities you want to get per pixel
    net = UNet(n_channels=3, n_classes=1, dropout=0.1)
    # data = torch.rand(1, 3, 572, 572)
    # output = net(data)
    # print("Shape of output: ", output.shape)
    print("UNet number of param: ", sum(p.numel() for p in net.parameters() if p.requires_grad)) #31037698 aka 31 milioni di trainable parameters