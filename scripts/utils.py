from __future__ import annotations
import torch

Device = torch.device

default_device: Device
if torch.cuda.is_available():
    default_device = torch.device("cuda:0")
else:
    default_device = torch.device("cpu")
