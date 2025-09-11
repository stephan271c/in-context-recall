import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Callable, Dict
import torch.nn.functional as F

import torchopt

class WeightModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.Sigmoid()
        )

        self._init_weights()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
    
    def _init_weights(self):
        # self.modules() iterates through all modules in the network
        # using xavier since we use sigmoid activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class LinearModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)
    
class TTTMLP(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        intermediate_dim = 4 * self.input_dim

        # First linear layer parameters
        self.W1 = nn.Parameter(torch.normal(0, 0.02, size=(self.input_dim, intermediate_dim)))
        self.b1 = nn.Parameter(torch.zeros(1, intermediate_dim))

        # Second linear layer parameters
        self.W2 = nn.Parameter(torch.normal(0, 0.02, size=(intermediate_dim, self.output_dim)))
        self.b2 = nn.Parameter(torch.zeros(1, self.output_dim))


    def forward(self, x):
        x = torch.matmul(x, self.W1) + self.b1
        x = F.gelu(x)
        x = torch.matmul(x, self.W2) + self.b2
        return x
    