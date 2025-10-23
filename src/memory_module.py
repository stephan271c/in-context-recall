import torch
import torch.nn as nn
from torch.func import functional_call, vmap
from typing import Callable, Dict, Sequence, Iterable
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

# better loop
from meta_optimizers import MetaOptimizer
from losses import windowed_p_loss, windowed_recall_cross_entropy

from synthetic_datasets import InContextRecallDataset


@torch.enable_grad()
def unroll_with_inner_param_dict(
    memory_module: nn.Module,   # e.g., InnerMLP; its weights are the outer-learned initialization
    dataset: InContextRecallDataset, # not batched
    batch_size: int,
    inner_opt: MetaOptimizer,
    lr_head: nn.Module | torch.Tensor,          # SigmoidLRHead
    loss_weight_head: nn.Module | torch.Tensor, # LossWeightHead
    inner_param_dict: Dict[str, torch.Tensor]
):
    # Get device from the first parameter of the module
    first_param = next(memory_module.parameters())
    device = first_param.device
    if dataset.inputs.device != device:
        dataset.inputs = dataset.inputs.to(device)
        dataset.targets = dataset.targets.to(device)
    seq_len = dataset.seq_len

    # theta_0: take the module parameters as the initial hidden state
    theta = {name: p for name, p in memory_module.named_parameters()}
    states = inner_opt.init_states(theta)

    outer_loss = 0.0

    #not handling batch dimension
    for t in range(seq_len): # sequence length
        key, value = dataset[t]  # get single time step input and target


        # 1) Inner loss on the parameter dict (e.g., weighted L2 over all leaves)
        #    ou can swap this for any differentiable regularizer over theta.Y
        if isinstance(loss_weight_head, nn.Module):
            loss_weight_t = loss_weight_head(key[-1])
        else:
            loss_weight_t = loss_weight_head
        
        L_inner = windowed_p_loss(functional_call(memory_module, inner_param_dict, key).T,
                                  value.T,
                                  weights=loss_weight_t)

        # 2) Gradients w.r.t. ALL leaves in theta (keep graph so the outer learns the init)
        names, leaves = _flatten_named(theta)
        grads_list = torch.autograd.grad(L_inner, leaves, create_graph=True, retain_graph=False, allow_unused=False)
        grads = _zip_to_dict(names, grads_list)

        # 3) Learning rate for this step (global or context)
        if isinstance(lr_head, nn.Module):
            lr_t = lr_head(key[-1])
        else:
            lr_t = lr_head

        # 4) One inner optimizer step on the whole parameter dict
        hyperparams = inner_param_dict.copy()
        hyperparams['lr'] = lr_t
        theta, states = inner_opt.step(theta, grads, states, **hyperparams)

        # by the way, the context window for windowed cross entropy may be different
        # 5) windowed cross entropy calculates logits and everything internally
        outer_loss = outer_loss + windowed_recall_cross_entropy(
            memory_module,
            theta,
            dataset.inputs,
            dataset.targets,
            time_index=t,
            window_size=1,
            loss_fn=F.cross_entropy
        )   


    return outer_loss