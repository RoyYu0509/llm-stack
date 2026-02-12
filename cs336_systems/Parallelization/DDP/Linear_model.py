import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import SGD
import os
from torch.utils.data import TensorDataset

class LinearModel(nn.Module):
    def __init__(self, in_dim, out_dim, device, dtype) -> None:
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim).to(device=device, dtype=dtype)

    def forward(self, x):
        return self.linear(x)
    

