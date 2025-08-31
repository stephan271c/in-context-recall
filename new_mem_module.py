import torch
import torch.nn as nn
from torch.func import functional_call
from typing import Tuple, Callable, Dict
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

    
class metaRNN(nn.Module):
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
        
    def forward(self, param_dict: Dict[str, torch.Tensor], keys: torch.Tensor, vals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """should be only 1 step of the sequence. but depends on the context size."""

        param_tensors = tuple(param_dict.values())
        preds = [functional_call(self.inner_model,param_dict,key) for key in keys]  
        if not self.weight_model:
            loss = self.loss_fn(preds, vals)
        else:
            loss = self.loss_fn(preds, vals, self.weight_model(keys))
        grads = torch.autograd.grad(loss, param_tensors, create_graph=True)
        new_params = {name: param - 0.1 * grad for (name, param), grad in zip(param_dict.items(), grads)}
        return new_params
