import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FocalLoss(nn.Module):
    def __init__(self, weight=None, gamma=0.):
        super(FocalLoss, self).__init__()
        assert gamma >= 0
        self.gamma = gamma
        self.weight = weight

    def forward(self, input, target):
        loss = None
        ce_inputs = F.cross_entropy(input, target, reduction='none', weight=self.weight)
        p = torch.exp(-ce_inputs)
        loss_array = (1 - p) ** self.gamma * ce_inputs
        loss = loss_array.mean()

        return loss