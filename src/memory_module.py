import torch
import torch.nn as nn
from torch.func import functional_call, vmap
from typing import Callable, Dict, Sequence
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

        # --- Explicit "unrolling" of the RNN with probing ---
        for t in range(seq_len):
            input_t = x[:, t, :]

            # Use vmap to explicitly handle batch dimension
            h_next = vmap(...)

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
            return output

    def _evaluate_probes_at_timestep(self, probe_inputs, h_prev, t):
        """Evaluate probe inputs at a specific timestep."""
        # This is a placeholder implementation
        # In practice, this would evaluate the probe inputs using the current hidden state
        probe_results = []
        for probe_input in probe_inputs:
            # problematic: memory_module cannot handle batch dimension, need to use vmap
            # For now, just use the memory module's output method
            probe_output = self.memory_module.output(probe_input.unsqueeze(0))
            probe_results.append(probe_output)
        return torch.stack(probe_results)

