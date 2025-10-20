import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Callable, Dict
import torch.nn.functional as F


class OuterProductMemory(nn.Module):
    """Simple outer product memory module that accumulates key-value associations."""

    def __init__(self, key_dim: int, val_dim: int, device: torch.device):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.device = device
        # Initialize cumulative matrix as a buffer (not a parameter)
        self.register_buffer('cumulative_matrix', torch.zeros(val_dim, key_dim, device=device))

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict values for given keys using accumulated outer products."""
        # keys shape: (batch_size, key_dim) or (seq_len, key_dim)
        # Return predictions: (batch_size, val_dim) or (seq_len, val_dim)
        return (self.cumulative_matrix @ keys.T).T

    def update(self, key: torch.Tensor, value: torch.Tensor):
        """Update the cumulative matrix with a new key-value pair."""
        # Accumulate outer product: M += value @ key.T
        self.cumulative_matrix = self.cumulative_matrix + torch.outer(value, key)

    def reset(self):
        """Reset the cumulative matrix to zeros."""
        self.cumulative_matrix = torch.zeros(self.val_dim, self.key_dim, device=self.device)

class HyperparamModel(nn.Module):
    """A generic model to predict a single hyperparameter."""
    def __init__(self, key_dim: int, initial_bias: float = 0.0):
        super().__init__()
        self.scaler = nn.Linear(key_dim, 1)
        torch.nn.init.constant_(self.scaler.bias, initial_bias)

    def forward(self, current_key: torch.Tensor) -> torch.Tensor:
        # Sigmoid ensures output is between (0, 1)
        return torch.sigmoid(self.scaler(current_key)).squeeze(-1)

class WeightModel(nn.Module):
    """input: key vector, output: weight vector of length context_dim, 
    for weighting loss over window"""
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
    
class TTT(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 2):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        intermediate_dim = 4 * self.input_dim

        # Build layer dimensions: input -> hidden(s) -> output.
        dims = [self.input_dim]
        if self.num_layers > 1:
            dims.extend([intermediate_dim] * (self.num_layers - 1))
        dims.append(self.output_dim)

        self.weights = nn.ParameterList([
            nn.Parameter(torch.normal(0, 0.02, size=(in_dim, out_dim)))
            for in_dim, out_dim in zip(dims[:-1], dims[1:])
        ])
        self.biases = nn.ParameterList([
            nn.Parameter(torch.zeros(1, out_dim))
            for out_dim in dims[1:]
        ])

    def forward(self, x):
        for idx, (weight, bias) in enumerate(zip(self.weights, self.biases)):
            x = torch.matmul(x, weight) + bias
            if idx < self.num_layers - 1:
                x = F.gelu(x)
        return x
    
