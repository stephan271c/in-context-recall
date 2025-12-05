from functools import wraps
from typing import Dict

import torch
import torch.nn.functional as F
from torch.func import functional_call


def validate_loss_inputs(func):
    """
    A decorator that validates shapes for loss functions expecting
    (predictions, targets, weights) as the first three arguments.
    """

    @wraps(func)
    def wrapper(
        predictions: torch.Tensor,
        targets: torch.Tensor,
        weights: torch.Tensor | None = None,
        *args,
        **kwargs,
    ):
        if predictions.dim() != 2:
            raise ValueError(
                f"Expected predictions to be a 2D tensor, but got {predictions.dim()} dimensions."
            )
        if predictions.shape != targets.shape:
            raise ValueError(
                f"Predictions and targets must have the same shape, but got {predictions.shape} and {targets.shape}."
            )

        if weights is not None:
            if weights.dim() != 1:
                raise ValueError(
                    f"Expected weights to be a 1D tensor, but got {weights.dim()} dimensions."
                )
            if weights.shape[0] != predictions.shape[1]:
                raise ValueError(
                    f"Weights length ({weights.shape[0]}) must match the window_size dimension of predictions ({predictions.shape[1]})."
                )

        return func(predictions, targets, weights, *args, **kwargs)

    return wrapper


@validate_loss_inputs
def windowed_p_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
    p: int = 2,
    reg_coef: float = 0.0,
) -> torch.Tensor:
    """
    computes weighted L_p^p-loss over a window with optional L2 regularization on weights.

    args:
        predictions: tensor of shape (vec_length, window_size)
        targets: tensor of shape (vec_length, window_size)
        weights: tensor of shape (window_size)
        p: int
        reg_coef: regularization coefficient for L2 regularization on weights (default: 0.0)
    """
    powered_diff = torch.pow(torch.abs(predictions - targets), p)
    if weights is not None:
        weighted_powered_diff = powered_diff * weights
    else:
        weighted_powered_diff = powered_diff

    data_loss = torch.sum(weighted_powered_diff)

    # regularization: Frobenius norm squared for 1D tensor
    if weights is not None and reg_coef > 0:
        reg_term = reg_coef * torch.sum(
            weights**2
        )  
        final_loss = data_loss + reg_term
    else:
        final_loss = data_loss

    return final_loss


@validate_loss_inputs
def windowed_inner_product_loss(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    weights: torch.Tensor | None = None,
) -> float:
    """
    computes inner-loss over a window.

    args:
        predictions: tensor of shape (vec_length, window_size)
        targets: tensor of shape (vec_length, window_size)
        weights: tensor of shape (window_size)
        p: int
    """

    return 0.0


# for outer loss
def windowed_recall_cross_entropy(
    model: "torch.nn.Module",
    params: Dict[str, torch.Tensor],
    all_keys: torch.Tensor,
    all_values: torch.Tensor,
    time_index: int,
    window_size: int = 1,
    offset: int = 0,
) -> torch.Tensor:
    """Computes a windowed recall loss that scores previous key/value pairs.

    For a given ``time_index``, this helper evaluates the model on the most recent
    ``window_size`` keys shifted back by ``offset`` steps and applies a classification
    loss against every stored value vector. The scores are computed by taking the
    dot-product between each predicted value and the matrix of known value
    vectors.

    Args:
        model: The differentiable memory model to evaluate.
        params: Functional parameters for ``model`` (e.g., from ``functional_call``).
        all_keys: Tensor of shape (sequence_length, key_dim) containing every key.
        all_values: Tensor of shape (sequence_length, value_dim) containing every value.
        time_index: Index of the current timestep (0-based).
        window_size: Number of timesteps to include in the recall window. values 
            larger than the number of seen steps are automatically clamped.
        offset: Number of steps to shift the recall window backwards. The window
            ends at ``time_index - offset``.

    Returns:
        A scalar tensor containing the averaged recall loss over the window. If
        ``time_index - offset <= 0`` (i.e., not enough prior steps), returns a
        zero loss on the correct device/dtype.
    """
    if window_size <= 0:
        raise ValueError("window_size must be a positive integer")
    if offset < 0:
        raise ValueError("offset must be a non-negative integer")
    if time_index < 0 or time_index >= all_keys.shape[0]:
        raise IndexError(
            f"time_index {time_index} is out of bounds for keys of length {all_keys.shape[0]}"
        )
    if not params:
        raise ValueError("params dictionary must not be empty")

    example_param = next(iter(params.values()))
    params_device = example_param.device
    params_dtype = example_param.dtype

    window_end = time_index - offset
    if window_end <= 0:
        return torch.zeros((), device=params_device, dtype=params_dtype)

    start_index = max(0, window_end - window_size)
    key_window = all_keys[start_index:window_end]
    value_matrix = all_values

    key_window = key_window.to(params_device)
    value_matrix = value_matrix.to(params_device)

    predictions = functional_call(model, params, key_window)
    if predictions.dim() == 1:
        predictions = predictions.unsqueeze(0)

    logits = predictions @ value_matrix.T
    target_indices = torch.arange(start_index, window_end, device=params_device)

    return F.cross_entropy(logits, target_indices)
