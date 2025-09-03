import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Callable, Dict

# You'll need to install torchopt: pip install meta-torchopt
import torchopt

class weightModel(nn.Module):
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
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

class LinearModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, x):
        return self.linear(x)