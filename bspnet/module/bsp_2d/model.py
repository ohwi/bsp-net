import torch.nn as nn

from .encoder import Encoder
from .generator import Generator


class BSP2dModel(nn.Module):
    def __init__(self, base_channels, num_planes, num_primitives, phase, resolution):
        super().__init__()

        self.encoder = Encoder(
            in_channels=1,
            num_planes=num_planes,
            base_channels=base_channels,
        )

        self.generator = Generator(
            num_planes=num_planes,
            num_primitives=num_primitives,
            phase=phase,
            resolution=resolution,
        )

        self.phase = phase

    def forward(self, x):
        weights, bias = self.encoder(x)
        output = self.generator(weights, bias)
        return output

    @property
    def concave_weights(self):
        return self.generator.concave_weights

    @property
    def convex_weights(self):
        return self.generator.convex_weights
