import torch

from scripts.Unet_valeria import ConvBlockDownsample, ConvBlockUpsample
#from scripts.utils import default_device


def test_random_image():
    # img_channels = 1
    # img_size = 28
    # x = torch.randn((2, img_channels, img_size, img_size))
    # gen = Generator(img_channels, img_channels)
    # print(gen(x, 1).shape)
    block2 = ConvBlockUpsample(128, 64)
    data_before_upsample = torch.rand((1, 128, 12, 12))
    # we do the forward pass by output_for_upsample from before
    output2 = block2(data_before_upsample, output_for_upsample)
    print("Shape of output2: ", output2.shape) #([1, 64, 20, 20])


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
