import torch
import torch.nn as nn
import copy
from typing import Tuple, List, Callable, Type
from abc import ABC, abstractmethod

# we will need to define functional forwards. we can pass in params.

class MemoryModule(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    @abstractmethod
    def functional_forward(self, x : torch.Tensor, params : Tuple[torch.Tensor,...]) -> torch.Tensor:
        pass
    
    @abstractmethod
    def forward(self, x : torch.Tensor) -> torch.Tensor:
        pass

    
class OneStep(nn.Module):
    """Wraps a model for online learning and evaluation on a per-sample basis.

    This module simulates an online learning process. The backward pass for this
    module is designed to optimize the `weight_model` parameters based on the
    outcome of the entire online learning simulation over a sequence.

    in the forward, key and val are the input and output of the model, both of shape (1, d)
    """
    def __init__(
        self, 
        loss_fn: Callable, # (loss_fn(predictions, targets, weights))
        memory_module: MemoryModule,
        loss_context_size: int = 1, # for the context window for loss. weird b/c loss_fn implicitly uses this. I used it in the forward for evals
        weight_model: nn.Module | None = None
        ):
        super().__init__()
        if loss_context_size < 1:
            raise ValueError("context_size must be at least 1.")

        # The base model is stored as a template and is not trained by the external optimizer.
        self.loss_fn = loss_fn
        self.loss_context_size = loss_context_size
        self.weight_model = weight_model
        self.inner_model = memory_module
        
    def forward(self, params: Tuple[torch.Tensor], keys: torch.Tensor, vals: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """should be only 1 step of the sequence. but depends on the context size."""
        preds = [self.inner_model.functional_forward(key, params) for key in keys]
        if not self.weight_model:
            loss = self.loss_fn(preds, vals)
        else:
            loss = self.loss_fn(preds, vals, self.weight_model(keys))
        grads = torch.autograd.grad(loss, params, create_graph=True)
        new_params = tuple(param - 0.1 * grad for param, grad in zip(params, grads))
        return new_params
