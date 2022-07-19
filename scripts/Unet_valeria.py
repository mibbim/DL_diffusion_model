"""
This file contains the implementation of Valeria's U-net module,
used in DiffusionModel to predict the noise
"""
import torch.nn as nn
import torch
import torch.nn.functional as F
from typing import Literal, Optional
#from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
import math

ActivationType = Literal["ReLU", "LeakyReLU", "PReLU", "SiLU", "ELU", "none"]

# Custom Activation function
class ActivationFunc(nn.Module):
    """
    #### Custom actiavation function to choose among:
    ReLU, LeakyReLU, ParametricReLU, Sigmoid Linear Unit, Exponential Linear Unit, identity.

    ----------
    #### Parameters
    * `activation_type`: ActivationType Literal
        the name of the custom activation function.
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
class DoubleConvBlock(nn.Module):
    """
    #### Double convolutional block has two convolution layers with batch normalization
    Specify custom activation function to use, other than default ReLU

    ----------
    #### Parameters
    * `in_channels`: number
            the number of convolution filters.
    * `out_channels`: number
            the number in output of channels.
    * `activation_type`: ActivationType literal.
            optional choice for activation function.
    * `dropout`: number, default is None.
            optional choice for dropout rate (use 0.1).
    * `time_channels`: number
            the number channels in the time step ($t$) embeddings.

    """
    def __init__(self, in_channels: int, out_channels: int, activation_type: ActivationType = "ReLU", dropout=None, time_channels=None):
        super().__init__()

        # First convolution layer and batch normalization with optional dropout
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3, bias=False), # stride=1,padding=0,bias set false since usage of BatchNorm
            nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
            ActivationFunc(activation_type)])
        if dropout:
            layers+= nn.ModuleList(nn.Dropout(dropout))
        self.conv_block1 = nn.Sequential(*layers)

        # Second convolution layer and batch normalization
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type))

        # MPL for time embeddings
        self.mlp = (
            nn.Sequential(ActivationFunc(activation_type), nn.Linear(time_channels, out_channels)) #order?
            if time_channels
            else None
        )

    def forward(self, data, time_emb=None):
        """
        * `data` has shape `[batch_size, in_channels, height, width]`
        * `time_emb` has shape `[batch_size, time_channels]`
        """
        x = self.conv_block1(data)

        if self.mlp and time_emb:
            time_emb = self.mlp(time_emb)
            # Add time embeddings
            x += time_emb[:, :, None, None]

        return self.conv_block2(x)

# Attention block
class AttentionBlock(nn.Module):
    """
    ### Attention block from https://colab.research.google.com/drive/1NFxjNI-UIR7Ku0KERmv7Yb_586vHQW43?usp=sharing#scrollTo=aHwkcmvkRLH0
    """

    def __init__(self, in_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `in_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = in_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, in_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(in_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, in_channels)
        # Scale for dot-product attention
        self.scale = d_k ** -0.5
        #
        self.n_heads = n_heads
        self.d_k = d_k

    def forward(self, x: torch.Tensor, t: Optional[torch.Tensor] = None):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # `t` is not used, but it's kept in the arguments because for the attention layer function signature
        # to match with `ResidualBlock`.
        _ = t
        # Get shape
        batch_size, in_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, in_channels]`
        x = x.view(batch_size, in_channels, -1).permute(0, 2, 1)
        # Get query, key, and values (concatenated) and shape it to `[batch_size, seq, n_heads, 3 * d_k]`
        qkv = self.projection(x).view(batch_size, -1, self.n_heads, 3 * self.d_k)
        # Split query, key, and values. Each of them will have shape `[batch_size, seq, n_heads, d_k]`
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        # Calculate scaled dot-product $\frac{Q K^\top}{\sqrt{d_k}}$
        attn = torch.einsum('bihd,bjhd->bijh', q, k) * self.scale
        # Softmax along the sequence dimension $\underset{seq}{softmax}\Bigg(\frac{Q K^\top}{\sqrt{d_k}}\Bigg)$
        attn = attn.softmax(dim=1)
        # Multiply by values
        res = torch.einsum('bijh,bjhd->bihd', attn, v)
        # Reshape to `[batch_size, seq, n_heads * d_k]`
        res = res.view(batch_size, -1, self.n_heads * self.d_k)
        # Transform to `[batch_size, seq, in_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, in_channels, height, width)

        #
        return res

# Module for downsampling
class ConvBlockDownsample(nn.Module):
    """
    ### Module for downsampling:
    #### Calls for ConvBlock and then performs downsampling

    ----------
    #### Parameters
    * `in_channels`: number
            the number of convolution filters.
    * `out_channels`: number
            the number in output of channels.
    ----------
    #### Return type
    * `out_for_upsample`: 
            data passed only through ConvBlock that will be later used in Upsampling.
    * `out`: 
            data already processed with max-polling, ready to be used for another downsampling layer.

    """
    def __init__(self, in_channels: int, out_channels: int, dropout=None):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels,out_channels, dropout=dropout)
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
    * `in_channels`: number
        the number of convolution filters.
    * `out_channels`: number
        the number in output of channels.
    * `dropout`: number, default is None
        the dropout rate.
    """
    def __init__(self, in_channels: int, out_channels: int, dropout=None):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels,out_channels, dropout=dropout)
        
    
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
    * `in_channels`: number
            the number of convolution filters.
    * `n_classes`: number
            the number of probabilities you want to get per pixel (num of output image's channels)
    
    """
    def __init__(self, in_channels, out_channels: int):
        super(OutConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """
    ### UNet network:
    ----------
    #### Parameters
    * `n_channels`: number, default is 3
            the number of input image's channels.
    * `n_classes`: number, default is 1 (regression problem)
            the number of probabilities you want to get per pixel aka  the number of output image's channels.
    * `n_conv_filters`: number, default is 64
            the number of convolutional filters for starting UNet block.
    * `n_unet_blocks`: number, default is 9
            the number of double convolutional blocks.
    * `dropout`: number, default is None 
            dropout rate. Use 0.1 as paper https://arxiv.org/pdf/2006.11239.pdf suggested - implementation https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py)

    """
    def __init__(self, n_channels: int=3, n_classes: int=1, n_conv_filters: int=64, n_unet_blocks: int=9, dropout=None):
        assert (n_unet_blocks >= 0 and n_unet_blocks%2 == 1), "n_unet_blocks must be an odd number"
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # # First block
        # model, res = [ConvBlockDownsample(n_channels, n_conv_filters, dropout=dropout)]

        # # Downsample blocks
        # n_downsample_blocks = n_unet_blocks // 2 #4
        # for i in range(n_downsample_blocks - 1): # 0,1,2
        #     mult = 2 ** i# 1 2 4
        #     model, res += [ConvBlockDownsample(n_conv_filters * mult, n_conv_filters * mult * 2, dropout=dropout)]

        # # Middle blocks
        # mult_mid_block= 2 ** (n_downsample_blocks -1)
        # model += DoubleConvBlock(n_conv_filters * mult_mid_block, n_conv_filters* mult_mid_block * 2)

        # # Upsample blocks
        # for i in range(n_downsample_blocks,0,-1): # 0,1,2,3
        #     mult = 2 ** i# 1 2 4 8
        #     model, res += [ConvBlockUpsample(n_channels, n_conv_filters * mult, dropout=dropout)]

        # # Final block
        # model += [OutConv(n_conv_filters, n_classes)]
        # self.model = nn.Sequential(*model)

        # Downsample blocks
        self.down1 = ConvBlockDownsample(n_channels, n_conv_filters, dropout=dropout)
        self.down2 = ConvBlockDownsample(n_conv_filters, n_conv_filters*2, dropout=dropout)
        self.down3 = ConvBlockDownsample(n_conv_filters*2, n_conv_filters*4, dropout=dropout)
        self.down4 = ConvBlockDownsample(n_conv_filters*4, n_conv_filters*8, dropout=dropout)

        # Middle blocks
        self.ground = DoubleConvBlock(n_conv_filters*8, n_conv_filters*16)

        # Upsample blocks
        self.up1 = ConvBlockUpsample(n_conv_filters*16, n_conv_filters*8, dropout=dropout)
        self.up2 = ConvBlockUpsample(n_conv_filters*8, n_conv_filters*4, dropout=dropout)
        self.up3 = ConvBlockUpsample(n_conv_filters*4, n_conv_filters*2, dropout=dropout)
        self.up4 = ConvBlockUpsample(n_conv_filters*2, n_conv_filters, dropout=dropout)
        # Final block
        self.outc = OutConv(n_conv_filters, n_classes)

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
        # self.model(x)
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
    data = torch.rand(1, 3, 572, 572)
    output = net(data)
    # print("Shape of output: ", output.shape)
    print("UNet number of param: ", sum(p.numel() for p in net.parameters() if p.requires_grad)) #31037698 aka 31 milioni di trainable parameters