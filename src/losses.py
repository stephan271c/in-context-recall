import torch
import torch.nn as nn
from torch.func import functional_call
from functools import wraps
from typing import Callable, Dict, Optional

# --- Validation Decorator ---
def validate_loss_inputs(func):
    """
    A decorator that validates shapes for loss functions expecting
    (predictions, targets, weights) as the first three arguments.
    """
    @wraps(func)
    def wrapper(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None, *args, **kwargs):
        assert predictions.dim() == 2, f"Expected predictions to be a 2D tensor, but got {predictions.dim()} dimensions."
        assert predictions.shape == targets.shape, \
            f"Predictions and targets must have the same shape, but got {predictions.shape} and {targets.shape}."

        if weights is not None:
            assert weights.dim() == 1, f"Expected weights to be a 1D tensor, but got {weights.dim()} dimensions."
            assert weights.shape[0] == predictions.shape[1], \
                f"Weights length ({weights.shape[0]}) must match the window_size dimension of predictions ({predictions.shape[1]})."
        
        # If all checks pass, call the original function
        return func(predictions, targets, weights, *args, **kwargs)
    return wrapper

@validate_loss_inputs
def windowed_p_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None, p: int = 2) -> torch.Tensor:
    """
    computes weighted L_p^p-loss over a window.

    args:
        predictions: tensor of shape (vec_length, window_size)
        targets: tensor of shape (vec_length, window_size)
        weights: tensor of shape (window_size)
        p: int
    """
    if weights is not None:
        assert predictions.shape[1] == targets.shape[1] == weights.shape[0], "window size must align"
    # --- Compute Loss ---
    powered_diff = torch.pow(torch.abs(predictions - targets), p)
    if weights is not None:
        weighted_powered_diff = powered_diff * weights
    else:
        weighted_powered_diff = powered_diff

    final_loss = torch.sum(weighted_powered_diff)
    return final_loss

@validate_loss_inputs
def windowed_inner_product_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None) -> float:
    """
    computes weighted L_p^p-loss over a window.

    args:
        predictions: tensor of shape (vec_length, window_size)
        targets: tensor of shape (vec_length, window_size)
        weights: tensor of shape (window_size)
        p: int
    """
            
    return 0.0


def windowed_recall_cross_entropy(
    model: "torch.nn.Module",
    params: Dict[str, torch.Tensor],
    all_keys: torch.Tensor,
    all_values: torch.Tensor,
    time_index: int,
    window_size: int = 1,
    loss_fn: Optional[Callable[[torch.Tensor, torch.Tensor], torch.Tensor]] = None,
) -> torch.Tensor:
    """Computes a windowed recall loss that scores previous key/value pairs.

    For a given ``time_index``, this helper evaluates the model on the most recent
    ``window_size`` keys (including the current key) and applies a classification
    loss against every stored value vector. The scores are computed by taking the
    dot-product between each predicted value and the matrix of known value
    vectors, mirroring the original outer-loss computation.

    Args:
        model: The differentiable memory model to evaluate.
        params: Functional parameters for ``model`` (e.g., from ``functional_call``).
        all_keys: Tensor of shape (sequence_length, key_dim) containing every key.
        all_values: Tensor of shape (sequence_length, value_dim) containing every value.
        time_index: Index of the current timestep (0-based).
        window_size: Number of timesteps to include in the recall window. The
            window always includes the current timestep; values larger than the
            number of seen steps are automatically clamped.
        loss_fn: Callable used to score the logits. Defaults to
            ``nn.CrossEntropyLoss`` if omitted.

    Returns:
        A scalar tensor containing the averaged recall loss over the window.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if time_index < 0 or time_index >= all_keys.shape[0]:
        raise IndexError(f"time_index {time_index} is out of bounds for keys of length {all_keys.shape[0]}")
    if not params:
        raise ValueError("params dictionary must not be empty")

    start_index = max(0, time_index - window_size + 1)
    key_window = all_keys[start_index: time_index + 1]
    value_matrix = all_values

    params_device = next(iter(params.values())).device
    key_window = key_window.to(params_device)
    value_matrix = value_matrix.to(params_device)

    predictions = functional_call(model, params, key_window)
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)

    logits = predictions @ value_matrix.T
    target_indices = torch.arange(start_index, time_index + 1, device=params_device)

    loss_fn = loss_fn or nn.CrossEntropyLoss()
    return loss_fn(logits, target_indices)
