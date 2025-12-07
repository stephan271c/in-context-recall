from typing import Dict, List, Optional, Tuple, Union

import torch
import torch.nn as nn
from torch.func import functional_call, grad, vmap

from src.losses import windowed_p_loss, windowed_recall_cross_entropy
from src.meta_optimizers import MetaOptimizer
from src.model_components import HyperparamHeadWrapper
from src.synthetic_datasets import InContextRecallDataset
from src.utils import (
    normalize_loss_weights,
    normalize_lr,
    prepare_initial_params,
    prepare_optimizer_state,
)


def inner_optimization_forward(
    memory_module: nn.Module,
    dataset: InContextRecallDataset,
    inner_optimizer: MetaOptimizer,
    inner_lr_head: Union[nn.Module, torch.Tensor, float],
    inner_loss_weight_head: Union[nn.Module, torch.Tensor],
    inner_optimizer_kwargs: Optional[Dict[str, torch.Tensor]] = None,
    outer_window_size: int = 1,
    offset: int = 0,
    eval_mode: bool = False,
) -> Tuple[torch.Tensor, List[torch.Tensor]]:
    """Perform inner optimization forward pass, unrolling over the sequence.
    
    This function implements the forward pass in a sequence: for each timestep,
    it computes gradients of an inner loss and updates the memory module parameters
    using the provided meta-optimizer. An outer loss is accumulated over all timesteps
    for meta-learning the inner loop hyperparameters.
    
    Args:
        memory_module: The differentiable memory model (e.g., TTT MLP).
        dataset: InContextRecallDataset providing key-value pairs with windowing.
        inner_optimizer: Meta-optimizer implementing differentiable parameter updates.
        inner_lr_head: Learning rate source - can be a neural network (context-aware),
            a learnable scalar, a fixed tensor, or a float.
        inner_loss_weight_head: Loss weight source for weighting the inner loss
            across the context window.
        inner_optimizer_kwargs: Additional keyword arguments for the optimizer
            (e.g., beta1, beta2 for Adam). Should not include 'lr'.
        outer_window_size: Number of timesteps to include in outer loss computation.
        offset: Backward offset for the outer loss window.
        eval_mode: If True, also collect predictions at each timestep for evaluation.
        
    Returns:
        A tuple of:
            - outer_loss: Scalar tensor containing the accumulated outer loss.
            - predictions: List of prediction tensors (empty if eval_mode=False).
    """
    try:
        device = next(memory_module.parameters()).device
    except StopIteration:
        device = torch.device("cpu")

    dataset = dataset.to(device)
    inputs = dataset.inputs
    targets = dataset.targets

    batch_size, seq_len, _ = inputs.shape

    # Initialize batched parameters and optimizer states
    theta = prepare_initial_params(memory_module, batch_size)
    base_params = dict(memory_module.named_parameters())
    states = prepare_optimizer_state(
        inner_optimizer.init_states(base_params), batch_size, device
    )

    get_lr = HyperparamHeadWrapper(inner_lr_head, device)
    get_loss_weight = HyperparamHeadWrapper(inner_loss_weight_head, device)

    # Prepare static hyperparameters (everything except lr)
    if inner_optimizer_kwargs is None:
        static_hparams = {}
    else:
        static_hparams = {
            k: (val.to(device) if torch.is_tensor(val) else val)
            for k, val in inner_optimizer_kwargs.items()
            if k != "lr"
        }
    
    def single_inner_loss(
        params: Dict[str, torch.Tensor],
        key_window: torch.Tensor,
        value_window: torch.Tensor,
        weight_vec: torch.Tensor,
    ) -> torch.Tensor:
        """Compute inner loss for a single batch element."""
        predictions = functional_call(memory_module, params, key_window)
        return windowed_p_loss(predictions.T, value_window.T, weight_vec)

    grad_fn = vmap(grad(single_inner_loss), in_dims=(0, 0, 0, 0))

    def inner_step(params, grad_dict, state, lr_scalar):
        """Single optimizer step for one batch element."""
        return inner_optimizer.step(
            params, grad_dict, state, lr=lr_scalar, **static_hparams
        )

    vmap_inner_step = vmap(inner_step, in_dims=(0, 0, 0, 0))

    def single_forward(params, inp):
        """Forward pass for one batch element (used in eval mode)."""
        return functional_call(memory_module, params, (inp,))

    vmap_batch_forward = vmap(single_forward, in_dims=(0, 0))

    def outer_loss_fn(params, seq_keys, seq_values, time_idx):
        """Compute outer loss for one batch element at given timestep."""
        return windowed_recall_cross_entropy(
            memory_module,
            params,
            seq_keys,
            seq_values,
            time_index=time_idx,
            window_size=outer_window_size,
            offset=offset
        )

    vmap_outer_loss = vmap(outer_loss_fn, in_dims=(0, 0, 0, None))
    
    outer_loss = torch.zeros((), device=device, dtype=inputs.dtype)
    predictions: List[torch.Tensor] = []

    for t in range(seq_len):
        key_window, value_window = dataset[t]
        # Shapes: (B, ctx_window, key_dim), (B, ctx_window, val_dim)
        
        window_length = key_window.shape[1]
        current_keys = key_window[:, -1]

        loss_weights = normalize_loss_weights(
            get_loss_weight(current_keys, batch_size), batch_size, window_length
        )
        lr_values = normalize_lr(get_lr(current_keys, batch_size), batch_size)

        grads = grad_fn(theta, key_window, value_window, loss_weights)
        theta, states = vmap_inner_step(theta, grads, states, lr_values)

        per_sample_outer = vmap_outer_loss(theta, inputs, targets, t)
        outer_loss = outer_loss + per_sample_outer.mean()

        # Collect predictions in eval mode
        if eval_mode:
            keys, _ = dataset[: t + 1]  # keys shape (B, t+1, key_dim)
            output_preds = vmap_batch_forward(theta, keys)
            predictions.append(output_preds)

    return outer_loss, predictions
