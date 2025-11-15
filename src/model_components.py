import torch
import torch.nn as nn


class HyperparamModel(nn.Module):
    """A generic model to predict a single hyperparameter."""
    def __init__(self, key_dim: int, initial_bias: float = 0.0):
        super().__init__()
        self.scaler = nn.Linear(key_dim, 1)
        torch.nn.init.constant_(self.scaler.bias, initial_bias)

    def forward(self, current_key: torch.Tensor) -> torch.Tensor:
        # Sigmoid ensures output is between (0, 1)
        return torch.sigmoid(self.scaler(current_key)).squeeze(-1) # shape (B,)


class LearnableHyperparam(nn.Module):
    """A learnable scalar hyperparameter.
    
    The initial value is set so that the sigmoid output defaults to 0.1
    """
    def __init__(self, initial_value: float = -2.1972):
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