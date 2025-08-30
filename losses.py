import torch
from functools import wraps

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
def windowed_p_loss(predictions: torch.Tensor, targets: torch.Tensor, weights: torch.Tensor | None = None, p: int = 2) -> float:
    """
    computes weighted L_p^p-loss over a window.

    args:
        predictions: tensor of shape (vec_length, window_size)
        targets: tensor of shape (vec_length, window_size)
        weights: tensor of shape (window_size)
        p: int
    """

    # --- Compute Loss ---
    powered_diff = torch.pow(torch.abs(predictions - targets), p)
    if weights is not None:
        weighted_powered_diff = powered_diff * weights
    else:
        weighted_powered_diff = powered_diff

    final_loss = torch.sum(weighted_powered_diff).item()
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