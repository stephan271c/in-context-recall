import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Callable, Dict, Sequence
import torch.nn.functional as F


class LinearAttentionMemory(nn.Module):
    """Simple outer product memory module that accumulates key-value associations."""

    def __init__(self, key_dim: int, val_dim: int):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim
        # Initialize cumulative matrix as a buffer (not a parameter)
        self.register_buffer('cumulative_matrix', torch.zeros(val_dim, key_dim))

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict values for given keys using accumulated outer products."""
        # keys shape: (batch_size, key_dim) or (seq_len, key_dim)
        # Return predictions: (batch_size, val_dim) or (seq_len, val_dim)
        outputs = (self.cumulative_matrix @ keys.T).T
        return outputs

    def update(self, key: torch.Tensor, value: torch.Tensor):
        """Update the cumulative matrix with a new key-value pair."""
        # Allow callers to pass either a single vector or a context window.
        with torch.no_grad():
            key_tensor = key
            value_tensor = value

            if key_tensor.ndim == 2 and value_tensor.ndim == 2:
                if key_tensor.shape[0] != value_tensor.shape[0]:
                    raise ValueError(
                        f"Key/value window length mismatch: {key_tensor.shape[0]} vs {value_tensor.shape[0]}"
                    )
                key_tensor = key_tensor[-1]
                value_tensor = value_tensor[-1]
            elif key_tensor.ndim != 1 or value_tensor.ndim != 1:
                raise ValueError(
                    f"Expected 1D key/value tensors or 2D context windows, got {key_tensor.shape} and {value_tensor.shape}"
                )

            if key_tensor.numel() != self.key_dim:
                raise ValueError(f"Key has {key_tensor.numel()} elements, expected {self.key_dim}")
            if value_tensor.numel() != self.val_dim:
                raise ValueError(f"Value has {value_tensor.numel()} elements, expected {self.val_dim}")

            # Accumulate outer product: M += value @ key.T
            self.cumulative_matrix += torch.outer(value_tensor, key_tensor)

    def reset(self):
        """Reset the cumulative matrix to zeros."""
        self.cumulative_matrix.zero_()

class MesaLayerMemory(nn.Module):
    """Mesa-layer memory implementing a discounted Sherman-Morrison update."""

    def __init__(
        self,
        key_dim: int,
        val_dim: int,
        lam: float = 1.0,
        epsilon: float = 1e-6,
    ):
        super().__init__()
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.lambda_init = lam
        self.epsilon = epsilon

        self.register_buffer('R_matrix', lam * torch.eye(key_dim))
        self.register_buffer('S_matrix', torch.zeros(val_dim, key_dim))
        self.register_buffer('phi_matrix', torch.zeros(val_dim, key_dim))

    def forward(self, keys: torch.Tensor) -> torch.Tensor:
        """Predict values for the provided keys using the current Mesa memory."""
        return (self.phi_matrix @ keys.T).T

    def update(self, key: torch.Tensor, value: torch.Tensor, gamma: torch.Tensor):
        """Update the memory with a new (key, value, gamma) triple."""
        with torch.no_grad():
            key_vec = key.reshape(-1)
            value_vec = value.reshape(-1)
            gamma_scalar = gamma.reshape(())

            if key_vec.numel() != self.key_dim:
                raise ValueError(f"key has {key_vec.numel()} elements, expected {self.key_dim}")
            if value_vec.numel() != self.val_dim:
                raise ValueError(f"value has {value_vec.numel()} elements, expected {self.val_dim}")

            Rk = self.R_matrix @ key_vec
            denom = 1.0 + torch.dot(key_vec, Rk)
            denom = denom + self.epsilon
            outer_Rk = torch.outer(Rk, Rk)
            self.R_matrix = self.R_matrix - outer_Rk / denom

            self.S_matrix = gamma_scalar * self.S_matrix + torch.outer(value_vec, key_vec)
            self.phi_matrix = self.S_matrix @ self.R_matrix

    def reset(self):
        """Reset the internal memory state."""
        with torch.no_grad():
            self.R_matrix.copy_(self.lambda_init * torch.eye(self.key_dim))
            self.S_matrix.zero_()
            self.phi_matrix.zero_()

class HyperparamModel(nn.Module):
    """A generic model to predict a single hyperparameter."""
    def __init__(self, key_dim: int, initial_bias: float = 0.0):
        super().__init__()
        self.scaler = nn.Linear(key_dim, 1)
        torch.nn.init.constant_(self.scaler.bias, initial_bias)

    def forward(self, current_key: torch.Tensor) -> torch.Tensor:
        # Sigmoid ensures output is between (0, 1)
        return torch.sigmoid(self.scaler(current_key)).squeeze(-1)

class LearnableHyperparam(nn.Module):
    """A learnable scalar hyperparameter."""
    def __init__(self, initial_value: float = -2.2):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(initial_value))

    def forward(self) -> torch.Tensor:
        # Ignore current_key; just return the learned parameter
        return torch.sigmoid(self.param)

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

