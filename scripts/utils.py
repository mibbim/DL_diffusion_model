from __future__ import annotations
# from Unet import Generator
# from DiffusionModel import Device, DiffusionModel
import torch

Device = torch.device

default_device: Device
if torch.cuda.is_available():
    default_device = torch.device("cuda")
else:
    default_device = torch.device("cpu")

# def default_model():
#     """Returns default model used mainly for testing"""
#     DiffusionModel(
#         noise_predictor=Generator(1, 1),
#         diffusion_steps_num=1000,
#         evaluation_device=default_device,
#     ).to(default_device)
