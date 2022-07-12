"""
This file contains the implementation of Valeria's U-net module,
used in DiffusionModel to predict the noise
"""
import torch.nn as nn
import torch
import torch.nn.functional as F

class ConvBlockDownsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3), #padding and stride are set to 1, no bias
            nn.BatchNorm2d(out_channels), #dimensionality of the incoming data
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            
        )
        self.downsample = nn.MaxPool2d(kernel_size=2)
    
    def forward(self, data):
        out = self.conv(data)
        out2 = self.downsample(out)
        return out, out2


class ConvBlockUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.upsample = nn.ConvTranspose2d(in_channels=in_channels, out_channels=out_channels, kernel_size=2, stride=2)
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=3),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
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
    block = ConvBlockDownsample(1, 64) 
    output_for_upsample, output = block(torch.rand(1, 1, 28, 28)) #batch, channels, size, size
    print("Shape of output: ", output.shape) #([1, 64, 12, 12])
    print("Shape of output for upsample: ", output_for_upsample.shape) #([1, 64, 24, 24])

    block2 = ConvBlockUpsample(128, 64)
    data_before_upsample = torch.rand((1, 128, 12, 12))
    # we do the forward pass by output_for_upsample from before
    output2 = block2(data_before_upsample, output_for_upsample).shape
    print("Output2: ", output2) #([1, 64, 20, 20])