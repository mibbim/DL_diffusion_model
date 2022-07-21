"""
This file contains the implementation of Valeria's U-net module,
used in DiffusionModel to predict the noise
"""
import torch.nn as nn
import torch
from torchvision.transforms import CenterCrop
from typing import List, Literal, Optional, Tuple, Union
# from torch.nn.parameter import Parameter # import Parameter to create custom activations with learnable parameters
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
            self.activation = nn.ReLU()  # Should I use inplace=True?
        elif activation_type == 'LeakyReLU':
            self.activation = nn.LeakyReLU()  # 0.01 default parameter
        elif activation_type == 'PReLU':  # to access the alpha parameter learnt use activation.weight
            self.activation = nn.PReLU()
            # self.activation = nn.PReLU(init=alpha)
        elif activation_type == 'SiLU':
            self.activation = nn.SiLU()
        elif activation_type == 'ELU':  # Exponential Linear Unit (computationally more expensive than ReLU)
            self.activation = nn.ELU()
            # self.activation = nn.ELU(alpha=alpha)
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

    ----------
    #### Parameters
    * `dim`: int
        the dimensionality of the position embeddings.
    * `activation_type`: ActivationType Literal
        the name of the custom activation function.

    """

    def __init__(self, dim, activation_type: ActivationType = "ReLU"):
        super().__init__()
        self.dim = dim  # is the number of dimensions in the embedding

        self.mlp = nn.Sequential(
            nn.Linear(self.dim // 4, self.dim),
            ActivationFunc(activation_type),
            nn.Linear(self.dim, self.dim)
        )

    def forward(self, t: torch.Tensor):
        half_dim = self.dim // 8
        emb = math.log(10_000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * -emb)
        emb = t[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()),
                        dim=1)  # Concatenates the given sequence of seq tensors in the given dimension
        return self.mlp(emb)


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
            the number channels in the time step ($t$) embeddings (time dim).

    """

    def __init__(self, in_channels: int, out_channels: int,
                 activation_type: ActivationType = "ReLU", dropout: float = None,
                 time_channels=None):
        super().__init__()

        # First convolution layer and batch normalization with optional dropout
        layers = nn.ModuleList([
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3,
                      bias=False, padding=(1, 1)), #SAME padding
            # stride=1,padding=0,bias set false since usage of BatchNorm
            nn.BatchNorm2d(out_channels),  # dimensionality of the incoming data
            ActivationFunc(activation_type)])
        if dropout:
            layers += nn.ModuleList([nn.Dropout(dropout), ])
        self.conv_block1 = nn.Sequential(*layers)

        # Second convolution layer and batch normalization
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3,
                      bias=False, padding=(1, 1)), #SAME padding
            nn.BatchNorm2d(out_channels),
            ActivationFunc(activation_type))

        # Linear layer for time embeddings
        self.time_emb = (
            nn.Sequential(nn.Linear(time_channels, out_channels))
            if time_channels
            else None
        )

    def forward(self, data: torch.Tensor, t: torch.Tensor):
        """
        * `data` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size, time_channels]`
        """
        # First convolution layer 
        x = self.conv_block1(data)

        # Add time embeddings
        if self.time_emb:
            t = self.time_emb(t)
            # Add time embeddings
            # x += t[:, :, None, None]
            x = x + t[:, :, None, None]

        return self.conv_block2(x)

# Ahh yes, magical attention...
class AttentionBlock(nn.Module):
    """
    ### Attention block.
    """

    def __init__(self, n_channels: int, n_heads: int = 1, d_k: int = None, n_groups: int = 32):
        """
        * `n_channels` is the number of channels in the input
        * `n_heads` is the number of heads in multi-head attention
        * `d_k` is the number of dimensions in each head
        * `n_groups` is the number of groups for [group normalization](../../normalization/group_norm/index.html)
        """
        super().__init__()

        # Default `d_k`
        if d_k is None:
            d_k = n_channels
        # Normalization layer
        self.norm = nn.GroupNorm(n_groups, n_channels)
        # Projections for query, key and values
        self.projection = nn.Linear(n_channels, n_heads * d_k * 3)
        # Linear layer for final transformation
        self.output = nn.Linear(n_heads * d_k, n_channels)
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
        batch_size, n_channels, height, width = x.shape
        # Change `x` to shape `[batch_size, seq, n_channels]`
        x = x.view(batch_size, n_channels, -1).permute(0, 2, 1)
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
        # Transform to `[batch_size, seq, n_channels]`
        res = self.output(res)

        # Add skip connection
        res += x

        # Change to shape `[batch_size, in_channels, height, width]`
        res = res.permute(0, 2, 1).view(batch_size, n_channels, height, width)

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
    * `dropout`: number, default is None.
            optional choice for dropout rate (use 0.1).
    * `time_channels`: number
            the number channels in the time step ($t$) embeddings (time dim).
    * `has_attn`: boolean, default is False.
    ----------
    #### Return type
    * `out_for_upsample`: 
            data passed only through ConvBlock that will be later used in Upsampling.
    * `out`: 
            data already processed with max-polling, ready to be used for another downsampling layer.

    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = None,
                 time_channels: int = None, has_attn: bool = False):
        super().__init__()
        self.conv = DoubleConvBlock(in_channels, out_channels, dropout=dropout,
                                    time_channels=time_channels)
        self.downsample = nn.MaxPool2d(kernel_size=2)  # the stride default value is kernel_size!
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, data: torch.Tensor, upsample_list: List[torch.Tensor], t: torch.Tensor, ):
        # Double convolution layer with attention block
        out_for_upsample = self.conv(data, t)
        out_for_upsample = self.attn(out_for_upsample)

        # Add to list
        upsample_list.append(out_for_upsample)

        # Resize data to avoid loss in MaxPooling
        if out_for_upsample.size(dim=2) % 2 != 0:
            out_for_upsample = nn.ReplicationPad2d((0, 1, 0, 1))(
                out_for_upsample)  # padding_left,padding_right,padding_top,padding_bottom

        # MaxPooling with stride=2, kernel=2
        out = self.downsample(out_for_upsample)

        # Return out and out_for_upsample as a List
        return out, upsample_list


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
    * `time_channels`: number
            the number channels in the time step ($t$) embeddings (time dim).
    * `has_attn`: boolean, default is False.
    """

    def __init__(self, in_channels: int, out_channels: int, dropout: float = None,
                 time_channels: int = None, has_attn: bool = False):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels,
                                           kernel_size=2, stride=2)
        self.conv = DoubleConvBlock(in_channels, out_channels, dropout=dropout,
                                    time_channels=time_channels)
        if has_attn:
            self.attn = AttentionBlock(out_channels)
        else:
            self.attn = nn.Identity()

    def forward(self, data: torch.Tensor, data_from_downsample: List[torch.Tensor],
                t: torch.Tensor):
        # Upsample
        out = self.upsample(data)

        # data has dimension ch x h x w
        # data_from_downsample has dimension ch x H x W
        # with H > h, W > w
        h, w = out.shape[2], out.shape[3]

        if h % 2 != 0:
            # padding_left,padding_right,padding_top,padding_bottom
            out = nn.ReplicationPad2d((0, 1, 0, 1))(out)

        data_previous = data_from_downsample.pop()
        H, W = data_previous.shape[2], data_previous.shape[3]

        # Do a center crop of data_from_downsample 
        # (starting from H//2, W//2, the center pixel of the larger image)
        if data_previous.shape != out.shape:
            cropper = CenterCrop((out.shape[-2], out.shape[-1]))
            data_previous = cropper(data_previous)
            # data_previous = data_previous[:, :,
            #                 H // 2 - h // 2: H // 2 + (h // 2 + h % 2),
            #                 W // 2 - w // 2: W // 2 + (w // 2 + w % 2)]
        out = torch.cat([out, data_previous], dim=1)

        # Double convolution layer and Attention block
        out = self.conv(out, t)
        return self.attn(out)


# Final convolution layer with 1x1 filter
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


class MiddleBlock(nn.Module):
    """
    ### Middle block
    It combines a `DoubleConvBlock`, `AttentionBlock`, followed by another `DoubleConvBlock`.
    This block is applied at the lowest resolution of the U-Net.

    ----------
    #### Parameters
    * `in_channels`: number
            the number of convolution filters.
    * `activation_type`: ActivationType literal.
            optional choice for activation function.
    * `dropout`: number, default is None.
            optional choice for dropout rate (use 0.1).
    * `time_channels`: number, default is None.
            the number channels in the time step ($t$) embeddings (time dim).
    """

    def __init__(self, n_channels: int, time_channels: int=None,
                 activation_type: ActivationType = "ReLU", dropout: float = None):
        super().__init__()
        self.res1 = DoubleConvBlock(n_channels, n_channels*2, activation_type=activation_type, dropout=dropout,
                                      time_channels=time_channels)
        self.attn = AttentionBlock(n_channels*2)
        self.res2 = DoubleConvBlock(n_channels*2, n_channels*2, activation_type=activation_type, dropout=dropout,
                                      time_channels=time_channels)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        x = self.res1(x, t)
        x = self.attn(x)
        x = self.res2(x, t)
        return x


class UNet(nn.Module):
    """
    ### UNet network:
    ----------
    #### Parameters
    * `n_channels`: number, default is 3
            the number of input image's channels.
    * `n_classes`: number, default is 1 (regression problem)
            the number of probabilities you want to get per pixel aka  the number of output image's channels.
    * `n_conv_filters`: number, default is 32
            the number of convolutional filters for starting UNet block.
    * `n_unet_blocks`: number, default is 7
            the number of double convolutional blocks.
    * `dropout`: number, default is None 
            dropout rate. Use 0.1 as paper https://arxiv.org/pdf/2006.11239.pdf suggested - implementation https://github.com/hojonathanho/diffusion/blob/master/diffusion_tf/models/unet.py)
    * `is_attn`: List o booleans, default is (False, True, True)
            List of booleans that indicate whether to use attention at each resolution
    """

    def __init__(self, n_channels: int = 3, n_classes: int = 1, n_conv_filters: int = 32,
                 n_unet_blocks: int = 7, dropout: float = None, is_attn: Union[Tuple[bool, ...], List[int]] = (False, True, True)):
        assert (
                n_unet_blocks >= 0 and n_unet_blocks % 2 == 1), "n_unet_blocks must be an odd number"
        super(UNet, self).__init__()
        self.n_channels = n_channels
        self.n_classes = n_classes

        # # First block
        # model = [ConvBlockDownsample(n_channels, n_conv_filters, dropout=dropout)]

        # # Downsample blocks
        # n_downsample_blocks = n_unet_blocks // 2 #4
        # for i in range(n_downsample_blocks - 1): # 0,1,2
        #     mult = 2 ** i# 1 2 4
        #     model += [ConvBlockDownsample(n_conv_filters * mult, n_conv_filters * mult * 2, dropout=dropout)]

        # # Middle blocks
        # mult_mid_block= 2 ** (n_downsample_blocks -1)
        # model += DoubleConvBlock(n_conv_filters * mult_mid_block, n_conv_filters* mult_mid_block * 2)

        # # Upsample blocks
        # for i in range(n_downsample_blocks,0,-1): # 0,1,2,3
        #     mult = 2 ** i# 1 2 4 8
        #     model += [ConvBlockUpsample(n_channels, n_conv_filters * mult, dropout=dropout)]

        # # Final block
        # model += [OutConv(n_conv_filters, n_classes)]
        # self.model = nn.Sequential(*model)

        # Time embedding layer. Time embedding has `n_channels * 4` channels
        self.time_emb = SinusoidalPositionEmbeddings(n_conv_filters * 4)

        # Downsample blocks
        self.down1 = ConvBlockDownsample(n_channels, n_conv_filters, dropout=dropout,
                                         time_channels=n_conv_filters * 4, has_attn=is_attn[0])
        self.down2 = ConvBlockDownsample(n_conv_filters, n_conv_filters * 2, dropout=dropout,
                                         time_channels=n_conv_filters * 4, has_attn=is_attn[1])
        self.down3 = ConvBlockDownsample(n_conv_filters * 2, n_conv_filters * 4, dropout=dropout,
                                         time_channels=n_conv_filters * 4, has_attn=is_attn[2])
        # self.down4 = ConvBlockDownsample(n_conv_filters * 4, n_conv_filters * 8, dropout=dropout,
        #                                  time_channels=n_conv_filters * 4)

        # Middle blocks
        # self.ground = DoubleConvBlock(n_conv_filters * 8, n_conv_filters * 16, dropout=dropout,
        #                               time_channels=n_conv_filters * 4)
        # self.ground = DoubleConvBlock(n_conv_filters * 4, n_conv_filters * 8, dropout=dropout,
        #                               time_channels=n_conv_filters * 4)
        self.ground = MiddleBlock(n_conv_filters * 4, dropout=dropout,
                                         time_channels=n_conv_filters * 4)

        # Upsample blocks
        # self.up1 = ConvBlockUpsample(n_conv_filters * 16, n_conv_filters * 8, dropout=dropout,
        #                              time_channels=n_conv_filters * 4)
        # self.up2 = ConvBlockUpsample(n_conv_filters * 8, n_conv_filters * 4, dropout=dropout,
        #                              time_channels=n_conv_filters * 4, has_attn=is_attn[2])
        # self.up3 = ConvBlockUpsample(n_conv_filters * 4, n_conv_filters * 2, dropout=dropout,
        #                              time_channels=n_conv_filters * 4, has_attn=is_attn[1])
        # self.up4 = ConvBlockUpsample(n_conv_filters * 2, n_conv_filters, dropout=dropout,
        #                              time_channels=n_conv_filters * 4, has_attn=is_attn[0])
        self.up2 = ConvBlockUpsample(n_conv_filters * 8, n_conv_filters * 4, dropout=dropout,
                                     time_channels=n_conv_filters * 4, has_attn=is_attn[2])
        self.up3 = ConvBlockUpsample(n_conv_filters * 4, n_conv_filters * 2, dropout=dropout,
                                     time_channels=n_conv_filters * 4, has_attn=is_attn[1])
        self.up4 = ConvBlockUpsample(n_conv_filters * 2, n_conv_filters, dropout=dropout,
                                     time_channels=n_conv_filters * 4, has_attn=is_attn[0])
        # Final block
        self.outc = OutConv(n_conv_filters, n_classes)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        * `x` has shape `[batch_size, in_channels, height, width]`
        * `t` has shape `[batch_size]`
        """
        img_size = (x.shape[-2], x.shape[-1])
        # Get time-step embeddings
        t = self.time_emb(t)
        up = list()
        x1, up = self.down1(x, up, t)
        x2, up = self.down2(x1, up, t)
        x, up = self.down3(x2, up, t)
        # x4, up = self.down4(x3, up, t)
        x = self.ground(x, t)
        # x = self.up1(x5, up, t)
        x = self.up2(x, up, t)
        x = self.up3(x, up, t)
        x = self.up4(x, up, t)
        x = self.outc(x)
        x = CenterCrop(img_size)(x)
        return x


if __name__ == "__main__":
    # m = nn.MaxPool2d(kernel_size=2)
    # input = torch.rand(1, 1, 3, 3)
    # print(input)
    # output = m(input)
    # print(output.shape)
    # print(output)
    # SinusoidalPositionEmbeddings
    # # n_classes is the number of probabilities you want to get per pixel

    # Let's see it in action on dummy data:
    # A dummy batch of 1 3-channel 572px images
    x = torch.randn(1, 3, 32, 32)

    # 't' - what timestep are we on
    t = torch.tensor([50], dtype=torch.long)

    # Define the unet model
    unet = UNet(n_classes=1)

    # The foreward pass (takes both x and t)
    model_output = unet(x, t)

    # The output shape matches the input.
    print("Shape of output: ", model_output.shape)

    # net = UNet(n_channels=3, n_classes=1, dropout=0.5)
    # data = torch.rand(1, 3, 572, 572)
    # output = net(data)
    # print("Shape of output: ", output.shape)
    # print("UNet number of param: ", sum(p.numel() for p in unet.parameters() if p.requires_grad)) #31876673 aka 31 milioni di trainable parameters
