from __future__ import annotations

from typing import TypeVar

import torch

Device = torch.device
IDT = TypeVar("IDT")  # Input Data Type

default_device: Device
if torch.cuda.is_available():
    default_device = torch.device("cuda:0")
else:
    default_device = torch.device("cpu")
