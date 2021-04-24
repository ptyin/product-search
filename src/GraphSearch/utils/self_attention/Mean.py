import torch
from torch import nn


class Mean(nn.Module):
    def __init__(self, dim):
        super(Mean, self).__init__()
        self.dim = dim

    def reset_parameters(self):
        pass

    def forward(self, tensor: torch.Tensor):
        return torch.mean(tensor, dim=self.dim)
