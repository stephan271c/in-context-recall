import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
from typing import Callable, Dict, Sequence, Iterable, Any, Mapping
import torch.nn.functional as F
from abc import ABC, abstractmethod
from meta_optimizers import MetaOptimizer
from losses import windowed_p_loss, windowed_recall_cross_entropy
from synthetic_datasets import BatchedInContextRecallDataset
from func_memory_module import LearnableHyperparam


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

    

# Helper functions for parameter dict manipulation

def _stack_param_dict(params: Mapping[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
    """Replicate a parameter dict across the batch dimension."""
    stacked: Dict[str, torch.Tensor] = {}
    for name, tensor in params.items():
        expanded = tensor.unsqueeze(0).expand(batch_size, *tensor.shape)
        # creates independent copies across batch dimension and allows gradient tracking
        stacked[name] = expanded.clone().requires_grad_(True)
    return stacked

def _broadcast_state_tree(state: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[str, Any]:
    """Broadcast optimizer state so each batch element has an independent copy."""
    if isinstance(state, dict):
        return {k: _broadcast_state_tree(v, batch_size, device) for k, v in state.items()}
    if torch.is_tensor(state):
        expanded = state.unsqueeze(0).expand(batch_size, *state.shape)
        return expanded.clone()
    if isinstance(state, bool):
        return torch.full((batch_size,), state, dtype=torch.bool, device=device)
    if isinstance(state, int):
        return torch.full((batch_size,), state, dtype=torch.long, device=device)
    if isinstance(state, float):
        return torch.full((batch_size,), state, dtype=torch.get_default_dtype(), device=device)
    raise TypeError(f"Unsupported optimizer state type: {type(state)}")

def _ensure_batch_vector(value: torch.Tensor | float, batch_size: int, length: int, device: torch.device, name: str) -> torch.Tensor:
    """Ensure a tensor/buffer is shaped (batch_size, length)."""
    if isinstance(value, float):
        tensor = torch.full((batch_size, length), value, device=device)
        return tensor

    if not torch.is_tensor(value):
        raise TypeError(f"{name} must be a torch.Tensor or float.")

    tensor = value.to(device)
    if tensor.dim() == 0:
        tensor = tensor.expand(batch_size, length)
    elif tensor.dim() == 1:
        if tensor.shape[0] == length:
            tensor = tensor.unsqueeze(0).expand(batch_size, -1)
        elif tensor.shape[0] == batch_size:
            tensor = tensor.unsqueeze(1).expand(-1, length)
        elif tensor.shape[0] == 1:
            tensor = tensor.expand(batch_size, length)
        else:
            raise ValueError(f"{name} has incompatible length {tensor.shape[0]}; expected {length} or batch size.")
    elif tensor.dim() == 2:
        if tensor.shape != (batch_size, length):
            raise ValueError(f"{name} shape {tensor.shape} must match (batch_size, {length}).")
    else:
        raise ValueError(f"{name} must be at most 2D; got {tensor.dim()} dimensions.")

    return tensor

# forward pass unrolling sequence
@torch.enable_grad()
def inner_optimization_forward(
    memory_module: nn.Module,   # e.g., TTT; its weights are the outer-learned initialization
    dataset: BatchedInContextRecallDataset,
    inner_opt: MetaOptimizer,
    lr_head: nn.Module | torch.Tensor | float,          # inner learning rate head
    loss_weight_head: nn.Module | torch.Tensor, # weights for inner loss
    inner_opt_kwargs: Dict[str, torch.Tensor] | None = None, # inner optimizer hyperparameters
    outer_window_size: int = 1,
    eval_mode = False
):
    # Get device from the first parameter of the module
    first_param = next(memory_module.parameters())
    device = first_param.device

    # Move dataset tensors to device in place
    dataset.inputs = dataset.inputs.to(device)
    dataset.targets = dataset.targets.to(device)

    inputs = dataset.inputs
    targets = dataset.targets

    # we expect inputs and targets to be 3D tensors: (B, seq_len, key/val-dim)
    if inputs.dim() != 3 or targets.dim() != 3:
        raise ValueError(f"Expected dataset inputs to have 3 dims, got {inputs.dim()}.")

    effective_batch = inputs.shape[0]
    seq_len = inputs.shape[1]

    base_params = {name: param for name, param in memory_module.named_parameters()}
    theta = _stack_param_dict(base_params, effective_batch)
    states = _broadcast_state_tree(inner_opt.init_states(base_params), effective_batch, device)

    # these are inner optimizer hyperparameters that do not change per step. lr can change
    if inner_opt_kwargs is None:
        static_hparams = {}
    else:
        static_hparams = {k: v for k, v in inner_opt_kwargs.items() if k != 'lr'}
        static_hparams = {
        k: val.to(device) if torch.is_tensor(val) else val # move to device. Some hyperparams are trainable tensors
        for k, val in static_hparams.items()
        }

    def single_inner_loss(params: Dict[str, torch.Tensor], key_window: torch.Tensor, value_window: torch.Tensor, weight_vec: torch.Tensor) -> torch.Tensor:
        predictions = functional_call(memory_module, params, key_window)
        return windowed_p_loss(predictions.T, value_window.T, weight_vec)

    grad_fn = vmap(grad(single_inner_loss))

    outer_loss = torch.zeros((), device=device)

    full_inputs = inputs
    full_targets = targets

    predictions = []

    for t in range(seq_len):
        key_window, value_window = dataset[t] # of shapes (B, ctx_window, key_dim), (B, ctx_window, val_dim)
        
        window_length = key_window.shape[1]
        current_keys = key_window[:, -1]

        if isinstance(loss_weight_head, nn.Module):
            loss_weight = loss_weight_head(current_keys) # shape (B, context_window)
        else:
            loss_weight = loss_weight_head # shape (context_window,)
            
        loss_weights = _ensure_batch_vector(loss_weight, effective_batch, window_length, device, "loss_weight")

        grads = grad_fn(theta, key_window, value_window, loss_weights)

        if isinstance(lr_head, nn.Module):
            if isinstance(lr_head, LearnableHyperparam):
                lr_values = lr_head() # shape (1,)
            else:
                lr_values = lr_head(current_keys) # shape (B,)
        else:
            lr_values = lr_head # float
        # of shape (B,)
        lr_values = _ensure_batch_vector(lr_values, effective_batch, 1, device, "learning_rate").squeeze(-1)

        def inner_step(params, grad_dict, state, lr_scalar):
            return inner_opt.step(params, grad_dict, state, lr=lr_scalar, **static_hparams)

        theta, states = vmap(inner_step)(theta, grads, states, lr_values)

        def outer_loss_fn(params, seq_keys, seq_values):
            return windowed_recall_cross_entropy(
                memory_module,
                params,
                seq_keys,
                seq_values,
                time_index=t,
                window_size=outer_window_size,
                loss_fn=F.cross_entropy
            )

        per_sample_outer = vmap(outer_loss_fn)(theta, full_inputs, full_targets)
        outer_loss = outer_loss + per_sample_outer.mean()

        def batch_functional_call(module, batched_params, inputs):
            # The lambda function has TWO arguments: `params` and `inp`.
            mapped_fn = lambda params, inp: functional_call(module, params, (inp,))

            # We tell vmap to map over the 0-th axis of BOTH arguments.
            return vmap(mapped_fn, in_dims=(0, 0))(batched_params, inputs)
                
        if eval_mode:
            keys, _ = dataset[:t+1] # keys shape (B, t+1, key_dim)
            output_preds = batch_functional_call(memory_module, theta, keys) # subbatching over time
            predictions.append(output_preds)

    return outer_loss, predictions
