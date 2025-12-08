"""Utility functions for tensor broadcasting and preparation in differentiable optimization loops."""

from typing import Any, Dict

import torch
import torch.nn as nn
from torch.utils._pytree import tree_map


def expand_tensor(x: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Replicate a tensor across the batch dimension with independent storage.

    Args:
        x: Input tensor of shape (...).
        batch_size: Target batch size.

    Returns:
        Tensor of shape (batch_size, ...) with cloned data.
    """
    return x.unsqueeze(0).expand(batch_size, *x.shape).clone()


def prepare_initial_params(
    module: nn.Module, batch_size: int
) -> Dict[str, torch.Tensor]:
    """Extract parameters from a module and broadcast them to (batch, ...) with gradients enabled.

    Args:
        module: PyTorch module to extract parameters from.
        batch_size: Target batch size for broadcasting.

    Returns:
        Dictionary mapping parameter names to batched parameter tensors with requires_grad=True.
    """
    params = dict(module.named_parameters())
    return tree_map(lambda p: expand_tensor(p, batch_size).requires_grad_(True), params)


def prepare_optimizer_state(state: Any, batch_size: int, device: torch.device) -> Any:
    """Broadcast optimizer state to the batch using PyTree utilities.

    Handles tensors, booleans, integers, and floats by expanding them
    to have a batch dimension.

    Args:
        state: Optimizer state (nested dict/list structure with tensors and scalars).
        batch_size: Target batch size for broadcasting.
        device: Target device for created tensors.

    Returns:
        Batched optimizer state with the same structure as input.
    """

    def _map_fn(x: Any) -> Any:
        if isinstance(x, torch.Tensor):
            return expand_tensor(x.to(device), batch_size)
        if isinstance(x, bool):
            return torch.full((batch_size,), x, dtype=torch.bool, device=device)
        if isinstance(x, int):
            return torch.full((batch_size,), x, dtype=torch.long, device=device)
        if isinstance(x, float):
            return torch.full(
                (batch_size,), x, dtype=torch.get_default_dtype(), device=device
            )
        return x

    return tree_map(_map_fn, state)


def normalize_loss_weights(
    weights: torch.Tensor, batch_size: int, length: int
) -> torch.Tensor:
    """Ensure loss weights have shape (batch_size, length).

    HyperparamHeadWrapper already guarantees a batch dimension; here we only
    fix the time/window dimension.

    Args:
        weights: Input weights tensor (scalar, 1D, or 2D).
        batch_size: Expected batch size.
        length: Expected window/sequence length.

    Returns:
        Tensor of shape (batch_size, length).

    Raises:
        ValueError: If weights have incompatible dimensions or shapes.
    """
    if weights.dim() == 0:
        weights = weights.expand(batch_size).unsqueeze(-1)
    if weights.dim() == 1:
        weights = weights.unsqueeze(-1).expand(batch_size, length)
    elif weights.dim() == 2:
        if weights.shape[1] == 1:
            weights = weights.expand(-1, length)
        elif weights.shape[1] != length:
            raise ValueError(
                f"loss_weight last dimension must be {length}, got {weights.shape[1]}"
            )
    else:
        raise ValueError(
            f"loss_weight must be 1D or 2D after batching, got {weights.dim()} dimensions"
        )
    return weights


def normalize_lr(lr: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure learning rate is a (batch_size,) vector.

    Args:
        lr: Input learning rate tensor (scalar, 1D, or 2D with last dim = 1).
        batch_size: Expected batch size.

    Returns:
        Tensor of shape (batch_size,).

    Raises:
        ValueError: If learning rate has incompatible shape.
    """
    if lr.dim() == 0:
        lr = lr.expand(batch_size)
    elif lr.dim() == 2 and lr.shape[1] == 1:
        lr = lr.squeeze(1)
    elif lr.dim() > 1 and lr.shape[-1] != 1:
        raise ValueError(
            f"learning_rate must be scalar per batch; got shape {lr.shape}"
        )
    return lr
