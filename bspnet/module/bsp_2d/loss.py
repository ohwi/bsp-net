import torch
import torch.nn as nn

from .model import BSP2dModel


class BSP2dLoss(nn.Module):
    def __init__(self, model):
        super().__init__()
        model: BSP2dModel
        self.model = model
        self.phase = model.phase

    def forward(self, pred, target):
        if self.phase == 0:
            loss_sp = torch.mean((pred - target) ** 2)
            loss_concave = torch.sum(torch.abs(self.model.concave_weights - 1.))
            loss_convex = torch.sum(
                torch.clamp(self.model.convex_weights - 1, min=0) - torch.clamp(self.model.convex_weights, max=0)
            )

            loss = loss_sp + loss_convex + loss_concave

        else:
            raise NotImplementedError

        return loss
