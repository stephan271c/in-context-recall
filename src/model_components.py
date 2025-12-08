from typing import Union

import torch
import torch.nn as nn
import torch.nn.functional as F


class TTT(nn.Module):
    """Test-Time Training MLP module.

    A differentiable memory module for test-time training.

    For 1 layer: f(x) = Wx (simple linear transformation)
    For >1 layers: f(x) = Wx + g(x) where g is an MLP with GELU activations.

    Args:
        input_dim: Dimension of input features (key dimension).
        output_dim: Dimension of output features (value dimension).
        num_layers: Number of linear layers (must be >= 1).
        init_var: Standard deviation for weight initialization.

    Raises:
        ValueError: If num_layers < 1.
    """

    HIDDEN_DIM_MULTIPLIER = 4

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        num_layers: int = 2,
        init_var: float = 0.02,
    ):
        super().__init__()
        if num_layers < 1:
            raise ValueError("num_layers must be at least 1")

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.num_layers = num_layers
        intermediate_dim = self.HIDDEN_DIM_MULTIPLIER * self.input_dim

        # Linear projection W for f(x) = Wx (or the Wx part of Wx + g(x))
        self.linear_weight = nn.Parameter(
            torch.normal(0, init_var, size=(input_dim, output_dim))
        )

        # Neural network g(x) for multi-layer case
        if self.num_layers > 1:
            # Build g(x) as an MLP with (num_layers - 1) hidden layers
            # Architecture: input -> [hidden -> gelu] * (num_layers - 1) -> output
            g_layers = []
            in_dim = input_dim
            for i in range(self.num_layers - 1):
                g_layers.append(nn.Linear(in_dim, intermediate_dim))
                g_layers.append(nn.GELU())
                in_dim = intermediate_dim
            g_layers.append(nn.Linear(intermediate_dim, output_dim))
            self.g_network = nn.Sequential(*g_layers)

            for module in self.g_network.modules():
                if isinstance(module, nn.Linear):
                    nn.init.normal_(module.weight, 0, init_var)
                    nn.init.zeros_(module.bias)
        else:
            self.g_network = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Linear transformation Wx
        linear_out = torch.matmul(x, self.linear_weight)

        if self.g_network is not None:
            # f(x) = Wx + g(x)
            return linear_out + self.g_network(x)
        else:
            # f(x) = Wx
            return linear_out


class HyperparamHeadWrapper:
    """
    Normalizes LR or Weight heads so the main loop doesn't need if/else checks.
    Always returns: Tensor of shape (Batch_Size, ...)
    """

    def __init__(
        self, head: Union[nn.Module, torch.Tensor, float], device: torch.device
    ):
        # Keep heads on the same device as the main model to avoid device mismatches
        if isinstance(head, nn.Module):
            head = head.to(device)
        self.head = head
        self.device = device

    def __call__(self, current_keys: torch.Tensor, batch_size: int) -> torch.Tensor:
        # Case 1: Neural Network Head
        if isinstance(self.head, nn.Module):
            # Check if it takes input or is a purely learnable parameter
            try:
                out = self.head(current_keys)
            except TypeError:
                # Fallback to parameter based (e.g. LearnableHyperparam)
                out = self.head()
            out = out.to(self.device)

            # Ensure it has batch dim
            if out.dim() == 0:
                out = out.expand(batch_size)
            elif out.dim() == 1 and out.shape[0] != batch_size:
                # Assume shape is (feature_dim,) -> expand to (B, feature_dim)
                out = out.unsqueeze(0).expand(batch_size, -1)
            return out

        # Case 2: Static Tensor
        elif torch.is_tensor(self.head):
            t = self.head.to(self.device)
            if t.dim() == 0:
                return t.expand(batch_size)
            if t.dim() == 1 and t.shape[0] != batch_size:
                return t.unsqueeze(0).expand(batch_size, -1)
            return t

        # Case 3: Float
        else:
            return torch.full((batch_size,), self.head, device=self.device)


class HyperparamModel(nn.Module):
    """A generic model to predict a single hyperparameter."""

    def __init__(self, key_dim: int, initial_bias: float = 0.0):
        super().__init__()
        self.scaler = nn.Linear(key_dim, 1)
        torch.nn.init.constant_(self.scaler.bias, initial_bias)

    def forward(self, current_key: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.scaler(current_key)).squeeze(-1)  # shape (B,)


class LearnableHyperparam(nn.Module):
    """A learnable scalar hyperparameter.

    The initial value is set so that the sigmoid output defaults to 0.1
    """

    def __init__(self, initial_value: float = -2.1972):
        super().__init__()
        self.param = nn.Parameter(torch.tensor(initial_value))

    def forward(self) -> torch.Tensor:
        return torch.sigmoid(self.param)


class WeightModel(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(input_dim, output_dim), nn.Sigmoid())

        self._init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def _init_weights(self):
        # using xavier since we use sigmoid activations
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
