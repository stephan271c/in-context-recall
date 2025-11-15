import torch
import torch.nn as nn
from torch import vmap
from torch.func import functional_call
from typing import Callable, Dict, Sequence
import torch.nn.functional as F
from abc import ABC, abstractmethod


class LinearRNN(ABC):
    """
    Abstract base class for manually defined linear RNNs.

    All concrete implementations must implement the following methods:
    - forward: Forward pass computation
    - update: State update mechanism
    - init_state: Initialize the RNN state
    """

    @abstractmethod
    def forward(self, *args, **kwargs):
        """
        Forward pass of the RNN.
        """
        pass

    @abstractmethod
    def update(self, *args, **kwargs):
        """
        Update the RNN state based on new input.
        """
        pass

    @abstractmethod
    def init_state(self, *args, **kwargs):
        """
        Initialize the RNN state.
        """
        pass


class LinearAttentionMemory(LinearRNN):
    """linear attention memory module that accumulates key-value associations."""

    def __init__(self, key_dim: int, val_dim: int, batch_size: int):
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.batch_size = batch_size

    # Functional forward: takes cumulative_matrix as input
    @staticmethod
    def forward(cumulative_matrix: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict values for given keys using accumulated outer products.

        Args:
            cumulative_matrix: (batch_size, val_dim, key_dim)
            keys: (batch_size, seq_len, key_dim)

        Returns:
            outputs: (batch_size, seq_len, val_dim)
        """
        # Use einsum for batch-aware matrix multiplication
        if keys.ndim == 3:  # across the sequence
            return torch.einsum("bvk, btk -> btv", cumulative_matrix, keys)
        elif keys.ndim == 2:  # single timestep
            return torch.einsum("bvk, bk -> bv", cumulative_matrix, keys)

    # Functional update: takes cumulative_matrix as input, returns updated
    @staticmethod
    def update(
        cumulative_matrix: torch.Tensor, key: torch.Tensor, value: torch.Tensor
    ) -> torch.Tensor:
        """Update the cumulative matrix with a new key-value pair.

        Args:
            cumulative_matrix: (batch_size, val_dim, key_dim)
            key: (batch_size, key_dim)
            value: (batch_size, val_dim)

        Returns:
            updated_cumulative_matrix: Same shape as input
        """
        if cumulative_matrix.ndim == 3:  # Batched
            # Vectorized outer product: value[:, None, :] @ key[:, :, None] but simplified
            outer = torch.einsum("bv, bk -> bvk", value, key)
            return cumulative_matrix + outer
        else:
            raise ValueError(
                "cumulative_matrix must be batched (3D tensor) for update_fn"
            )

    @staticmethod
    def init_state(batch_size: int, val_dim: int, key_dim: int) -> torch.Tensor:
        """Initialize batched cumulative matrices."""
        return torch.zeros(batch_size, val_dim, key_dim)


class MesaLayerMemory(LinearRNN):
    """Batched Mesa-layer memory implementing discounted Sherman-Morrison updates."""

    def __init__(
        self, key_dim: int, val_dim: int, lam: float = 1.0, epsilon: float = 1e-6
    ):
        """Initialize the batched Mesa memory parameters.

        Args:
            key_dim: Dimensionality of keys
            val_dim: Dimensionality of values
            batch_size: Size of the batch
            lam: Initial lambda for R matrix initialization
            epsilon: Small constant for numerical stability
        """
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.lambda_init = lam
        self.epsilon = epsilon

    @staticmethod
    def forward(phi_matrix: torch.Tensor, keys: torch.Tensor) -> torch.Tensor:
        """Forward pass: predict values for given keys using current phi matrix.

        Args:
            phi_matrix: (batch_size, val_dim, key_dim)
            keys: (batch_size, seq_len, key_dim) or (batch_size, key_dim)

        Returns:
            outputs: (batch_size, seq_len, val_dim) or (batch_size, val_dim)
        """
        if phi_matrix.ndim != 3:
            raise ValueError("phi_matrix must be batched (3D tensor)")

        if keys.ndim == 3:  # (batch_size, seq_len, key_dim)
            return torch.einsum("bvk, btk -> btv", phi_matrix, keys)
        elif keys.ndim == 2:  # (batch_size, key_dim)
            return torch.einsum("bvk, bk -> bv", phi_matrix, keys)
        else:
            raise ValueError("keys must be 2D or 3D tensor")

    @staticmethod
    def update(
        R_matrix: torch.Tensor,
        S_matrix: torch.Tensor,
        phi_matrix: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        gamma: torch.Tensor,
        epsilon: float = 1e-6,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Update the memory matrices with a new (key, value, gamma) triple.

        Args:
            R_matrix: (batch_size, key_dim, key_dim)
            S_matrix: (batch_size, val_dim, key_dim)
            phi_matrix: (batch_size, val_dim, key_dim)
            key: (batch_size, key_dim)
            value: (batch_size, val_dim)
            gamma: (batch_size,) or scalar
            epsilon: Small constant for numerical stability

        Returns:
            (R_matrix, S_matrix, phi_matrix): Updated matrices
        """
        if R_matrix.ndim != 3 or S_matrix.ndim != 3 or phi_matrix.ndim != 3:
            raise ValueError("All matrices must be batched (3D tensors)")

        if key.ndim != 2 or value.ndim != 2:
            raise ValueError("key and value must be 2D tensors")

        device = key.device
        R_matrix = R_matrix.to(device)
        S_matrix = S_matrix.to(device)
        phi_matrix = phi_matrix.to(device)
        if isinstance(gamma, torch.Tensor):
            gamma = gamma.to(device)
        else:
            gamma = torch.tensor(gamma, device=device, dtype=key.dtype)

        batch_size = key.shape[0]
        if gamma.ndim == 0:
            gamma_batch = gamma.expand(batch_size)
        elif gamma.ndim == 1 and gamma.shape[0] == batch_size:
            gamma_batch = gamma
        else:
            raise ValueError("gamma must be a scalar or a batch-sized tensor")

        gamma_matrix = gamma_batch.unsqueeze(-1).unsqueeze(-1)

        # Compute Rk for each batch element: (batch_size, key_dim)
        Rk = torch.einsum("bij, bj -> bi", R_matrix, key)

        # Compute denominator: (batch_size,)
        denom = gamma_batch + torch.einsum("bk, bk -> b", key, Rk)
        denom = denom + epsilon

        # Compute outer product of Rk: (batch_size, key_dim, key_dim)
        outer_Rk = torch.einsum("bk, bl -> bkl", Rk, Rk)

        # Update R_matrix, see eq (17) of https://arxiv.org/pdf/2309.05858
        R_matrix = (
            R_matrix - outer_Rk / denom.unsqueeze(-1).unsqueeze(-1)
        ) / gamma_matrix

        # Update S_matrix: gamma * S + outer_product(value, key)
        S_matrix = gamma_matrix * S_matrix + torch.einsum("bv, bk -> bvk", value, key)

        # Update phi_matrix: S @ R
        phi_matrix = torch.einsum("bvk, bkj -> bvj", S_matrix, R_matrix)

        return R_matrix, S_matrix, phi_matrix

    def init_state(
        self, batch_size: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Initialize the three memory matrices for a batch.

        Args:
            batch_size: Size of the batch

        Returns:
            (R_matrix, S_matrix, phi_matrix): Initialized matrices
        """

        R_matrix = (
            self.lambda_init
            * torch.eye(self.key_dim).unsqueeze(0).expand(batch_size, -1, -1).clone()
        )
        S_matrix = torch.zeros(batch_size, self.val_dim, self.key_dim)
        phi_matrix = torch.zeros(batch_size, self.val_dim, self.key_dim)

        return R_matrix, S_matrix, phi_matrix
