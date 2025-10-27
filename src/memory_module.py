import torch
import torch.nn as nn
from torch.func import functional_call, vmap, grad
from typing import Callable, Dict, Sequence, Iterable, Any
import torch.nn.functional as F
from abc import ABC, abstractmethod

class MemoryModule(nn.Module, ABC):
    @abstractmethod
    def forward(self, keys, values, **kwargs):
        """"Produces next hidden state of memory module. may involve metaoptimization.

        Args:
            keys: Input tensor(s)
            values: Previous hidden state(s)

        Returns:
            Next hidden state(s)
        """
        raise NotImplementedError

    @abstractmethod
    def output(self, x):
        """Produces output from hidden state of memory module"""
        raise NotImplementedError

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

    def update(self, keys: torch.Tensor, values: torch.Tensor, params, loss_fn: Callable, inner_opt,
              state: Dict = None, **kwargs):
        """
        Update TTT module parameters using gradient-based inner loop optimization.

        Args:
            keys: Input key tensors of size (context_size, key_dim)
            values: Target of value tensors of size (context_size, value_dim)
            params: Current parameters of the module
            loss_fn: Loss function that takes predictions and targets, returns loss
            inner_opt: Meta optimizer with step() method for parameter updates
            state: Optimizer state dictionary
            **kwargs: Additional optimizer-specific parameters (e.g., lr, momentum, etc.)

        Returns:
            updated_params: Updated parameters dictionary
            updated_state: Updated optimizer state
        """
        if state is None:
            state = {}

        # Forward pass to get predictions. This makes no sense
        predictions = functional_call(self, params, keys)

        # Compute loss
        loss = loss_fn(predictions.T, values.T)

        # Compute gradients
        param_items = list(params.items())
        grads_tuple = torch.autograd.grad(
            loss,
            tuple(p for _, p in param_items),
            retain_graph=True,
            allow_unused=True,
        )

        # Handle None gradients
        grads = {}
        for (name, param), grad in zip(param_items, grads_tuple):
            grads[name] = torch.zeros_like(param) if grad is None else grad

        # Update parameters using inner optimizer
        updated_params, updated_state = inner_opt.step(
            params,
            grads,
            state,
            **kwargs
        )

        # Update the module parameters in-place
        with torch.no_grad():
            for name, param in self.named_parameters():
                if name in updated_params:
                    param.copy_(updated_params[name])

        return updated_params, updated_state
    
# this is deprecated. use the unroll_with_inner_param_dict function instead
class MetaRNN(nn.Module):
    def __init__(
        self,
        memory_module: MemoryModule,
        lr_model: nn.Module | float,
        loss_weight_model: nn.Module | torch.Tensor,
        trainable_init_state: bool = False
        ):
        super().__init__()
        self.memory_module = memory_module
        self.lr_model = lr_model # inner learning rate model or fixed lr
        self.loss_weight_model = loss_weight_model
        self.trainable_init_state = trainable_init_state
        # Store hidden size as an integer (this is pretty ugly)
        hidden_size = getattr(memory_module, 'hidden_size', None)
        if hidden_size is None:
            raise ValueError("Memory module must have a 'hidden_size' attribute")
        if not isinstance(hidden_size, int):
            hidden_size = int(hidden_size)
        self.hidden_size = hidden_size

        if trainable_init_state:
            self.init_hidden_state = nn.Parameter(torch.zeros(1, self.hidden_size))
        else:
            # For non-trainable, we'll generate fresh random states per batch in forward
            self.init_hidden_state = None

    def forward(self, x, probe_inputs = None):
    # Note that the learning rate has to be calculated in sequence.
    # we will internally slice the probe inputs if provided
        batch_size = x.size(0) # picks out dimensions
        seq_len = x.size(1)

        # Initialize hidden state - different random init for each batch example
        if self.trainable_init_state:
            h_prev = self.init_hidden_state.expand(batch_size, self.hidden_size)
        else:
            # Generate fresh random initialization for each example in the batch
            h_prev = torch.randn(batch_size, self.hidden_size, device=x.device, dtype=x.dtype)

        hidden_states_over_time = []

        # Store probe evaluation results if requested
        probe_outputs = [] if probe_inputs is not None else None

        def _memory_step(single_input, single_hidden):
            return self.memory_module.forward(single_input, single_hidden)

        batched_memory_step = vmap(_memory_step)

        # --- Explicit "unrolling" of the RNN with probing ---
        for t in range(seq_len):
            input_t = x[:, t, :]

            # Use vmap to explicitly handle batch dimension
            h_next = batched_memory_step(input_t, h_prev)

            # Store the result
            hidden_states_over_time.append(h_next)

            # === PROBING: Evaluate additional inputs at this time step ===
            if probe_inputs is not None:
                probe_results_at_t = self._evaluate_probes_at_timestep(
                    probe_inputs, h_prev, t
                )
                probe_outputs.append(probe_results_at_t)

            # Update h_prev for the next time step (main sequence only)
            h_prev = h_next

        # Stack the collected hidden states
        rnn_out = torch.stack(hidden_states_over_time, dim=1)
        
        # Get the final output using the memory module's output method
        output = self.memory_module.output(rnn_out)

        # Return both main output and probe results if probing was done
        if probe_outputs is not None:
            return output, probe_outputs
        else:
            return output, []  # Return empty list if no probes were evaluated

    def _evaluate_probes_at_timestep(self, probe_inputs, h_prev, t):
        """Evaluate probe inputs at a specific timestep."""
        # This is a placeholder implementation
        # In practice, this would evaluate the probe inputs using the current hidden state

        def _output_step(single_input, single_hidden):
            return self.memory_module.output(single_input, single_hidden)        

        # Get dimensions
        batch_size = h_prev.size(0)
        num_probes = probe_inputs.size(0)

        # Expand probe_inputs to (batch_size, num_probes, input_dim)
        expanded_probes = probe_inputs.unsqueeze(0).expand(batch_size, -1, -1)
        
        # Expand h_prev to (batch_size, num_probes, hidden)
        expanded_h = h_prev.unsqueeze(1).expand(-1, num_probes, -1)
        
        # Use nested vmap to vectorize over batch and probe dimensions
        batched_output_step = vmap(vmap(_output_step))
        return batched_output_step(expanded_probes, expanded_h)



# Helper functions for parameter dict manipulation

def _flatten_named(params: Dict[str, torch.Tensor]):
    names = list(params.keys())
    #tensors = [params[n] for n in names]
    tensors = list(params.values())
    return names, tensors

def _zip_to_dict(names: Iterable[str], tensors: Iterable[torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {n: t for n, t in zip(names, tensors)}

def _stack_param_dict(params: Dict[str, torch.Tensor], batch_size: int) -> Dict[str, torch.Tensor]:
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

# better loop
from meta_optimizers import MetaOptimizer
from losses import windowed_p_loss, windowed_recall_cross_entropy

from synthetic_datasets import InContextRecallDataset, BatchedInContextRecallDataset


# should only work with batch ICR
# forward pass unrolling sequence
@torch.enable_grad()
def unroll_with_inner_param_dict(
    memory_module: nn.Module,   # e.g., InnerMLP; its weights are the outer-learned initialization
    dataset: BatchedInContextRecallDataset,
    inner_opt: MetaOptimizer,
    lr_head: nn.Module | torch.Tensor | float,          # SigmoidLRHead
    loss_weight_head: nn.Module | torch.Tensor | float, # LossWeightHead
    inner_param_dict: Dict[str, torch.Tensor], # inner optimizer hyperparameters
    eval_mode = False
):
    # Get device from the first parameter of the module
    first_param = next(memory_module.parameters())
    device = first_param.device

    inputs = dataset.inputs.to(device)
    targets = dataset.targets.to(device)

    # we expect inputs and targets to be 3D tensors: (B, seq_len, key/val-dim)
    if inputs.dim() != 3 or targets.dim() != 3:
        raise ValueError(f"Expected dataset inputs to have 2 or 3 dims, got {inputs.dim()}.")

    effective_batch = inputs.shape[0]
    seq_len = inputs.shape[1]

    base_params = {name: param for name, param in memory_module.named_parameters()}
    theta = _stack_param_dict(base_params, effective_batch)
    states = _broadcast_state_tree(inner_opt.init_states(base_params), effective_batch, device)

    # these are inner optimizer hyperparameters that do not change per step. lr can change
    static_hparams = {k: v for k, v in inner_param_dict.items() if k != 'lr'}
    static_hparams = {
        k: val.to(device) if torch.is_tensor(val) else val
        for k, val in static_hparams.items()
    }

    def single_inner_loss(params: Dict[str, torch.Tensor], key_window: torch.Tensor, value_window: torch.Tensor, weight_vec: torch.Tensor) -> torch.Tensor:
        predictions = functional_call(memory_module, params, key_window)
        return windowed_p_loss(predictions.T, value_window.T, weight_vec)

    grad_fn = vmap(grad(single_inner_loss))

    outer_loss = torch.zeros((), device=device)

    full_inputs = inputs
    full_targets = targets

    for t in range(seq_len):
        key_window, value_window = dataset[t] # of shapes (B, ctx_window, key_dim), (B, ctx_window, val_dim)
        
        window_length = key_window.shape[1]
        current_keys = key_window[:, -1]

        if isinstance(loss_weight_head, nn.Module):
            loss_weight = loss_weight_head(current_keys)
        else:
            loss_weight = loss_weight_head
            
        loss_weights = _ensure_batch_vector(loss_weight, effective_batch, window_length, device, "loss_weight")

        grads = grad_fn(theta, key_window, value_window, loss_weights)

        if isinstance(lr_head, nn.Module):
            lr_values = lr_head(current_keys)
        else:
            lr_values = lr_head if lr_head is not None else inner_param_dict.get('lr', 1.0)
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
                window_size=1,
                loss_fn=F.cross_entropy
            )

        per_sample_outer = vmap(outer_loss_fn)(theta, full_inputs, full_targets)
        outer_loss = outer_loss + per_sample_outer.mean()

    return outer_loss
