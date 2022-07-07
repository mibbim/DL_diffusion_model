"""
A diffusion Model object
"""

from __future__ import annotations
import torch.nn as nn


class DiffusionModel(nn.Module):  # Not sure should inherit
    def __init__(self,
                 noise_predictor: nn.Module,
                 ) -> None:
        super().__init__()  # Not sure should inherit
        self.noise_predictor = noise_predictor
        raise NotImplementedError

    def train(self, mode: bool = True):
        """Forwarding the coll to inner module"""
        self.noise_predictor.train(mode=mode)
        return self

    def eval(self):
        """Forwarding the coll to inner module"""
        self.noise_predictor.eval()
        return self
