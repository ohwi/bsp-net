import numpy as np

import torch
import torch.nn as nn


class Generator(nn.Module):
    def __init__(self, num_planes, num_primitives, phase, resolution):
        super().__init__()

        self.phase = phase

        # convex weight of level 2
        self.convex_weights = torch.nn.Parameter(torch.randn([num_primitives, num_planes]))
        nn.init.normal_(self.convex_weights, mean=0.0, std=0.02)

        # level 3
        if phase == 0:
            self.concave_weights = torch.nn.Parameter(torch.randn([1, num_primitives]))
            nn.init.normal_(self.concave_weights, mean=1e-5, std=0.02)
        else:
            self.concave_weights = None

        # TODO; use meshgrid
        coords = np.zeros([resolution, resolution, 2], np.float32)
        for i in range(resolution):
            for j in range(resolution):
                coords[i, j, 0] = i
                coords[i, j, 1] = j

        # normalization and flatten
        coords = (coords + 0.5) / resolution - 0.5
        coords = np.reshape(coords, [1, resolution * resolution, 2])
        self.register_buffer("coords", torch.from_numpy(coords))

    def forward(self, weights, bias, coords=None):
        if coords is None:
            coords = self.coords

        if self.phase == 0:
            return self.forward_phase_zero(weights, bias, coords)
        else:
            # TODO; add phase one
            raise NotImplementedError

    def forward_phase_zero(self, weights, bias, coords):
        # level 1
        x = torch.matmul(coords, weights) + bias
        x = torch.clamp(x, min=0)   # C+(x) in eq. 2

        # level 2
        x = torch.einsum("b x p, c p -> b x c", x, self.convex_weights)
        x = torch.clamp(1-x, min=0, max=1)  # [1 - C+(x)] in eq. 4

        # level 3
        x = torch.einsum("b x c, z c -> b x", x, self.concave_weights)  # z as 1
        x = torch.clamp(x, max=1)
        x = torch.clamp(x, min=0)   # S+(x) in eq. 4

        return x
