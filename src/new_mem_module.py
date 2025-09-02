import torch
import torch.nn as nn
from torch.func import functional_call
from typing import List, Callable, Dict

    
class metaRNN(nn.Module):
    """Wraps a model for online learning and evaluation on a per-sample basis.

    This module simulates an online learning process. The backward pass for this
    module is designed to optimize the `weight_model` parameters based on the
    outcome of the entire online learning simulation over a sequence.

    in the forward, key and val are the input and output of the model, both of shape (context_size, d)
    """
    def __init__(
        self, 
        loss_fn: Callable, # (loss_fn(predictions, targets, weights))
        memory_module: nn.Module,
        context_size: int = 1, # for the context window for loss. weird b/c loss_fn implicitly uses this. I used it in the forward for evals
        weight_model: nn.Module | None = None
        ):
        super().__init__()
        if context_size < 1:
            raise ValueError("context_size must be at least 1.")

        # The base model is stored as a template and is not trained by the external optimizer.
        self.loss_fn = loss_fn
        self.loss_context_size = context_size
        self.weight_model = weight_model
        self.inner_model = memory_module
        
    def forward(self, param_dict: Dict[str, torch.Tensor], keys: torch.Tensor, vals: torch.Tensor) -> Dict[str, torch.Tensor]:
        """should be only 1 step of the sequence. but depends on the context size."""
        #return param_dict
        param_tensors = tuple(param_dict.values())
        preds = torch.stack([functional_call(self.inner_model, param_dict, key) for key in keys])
        if not self.weight_model:
            loss = self.loss_fn(preds, vals)
        else:
            loss = self.loss_fn(preds, vals, self.weight_model(keys))
        grads = torch.autograd.grad(loss, param_tensors, create_graph=True)
        new_params = {name: param - 0.1 * grad for (name, param), grad in zip(param_dict.items(), grads)}
        #new_params = param_dict
        return new_params


def lookback_accuracy_fn(
    t: int,
    y_pred_context: List[torch.Tensor],
    y_target_sequence: torch.Tensor,
    lookback_correct_counts: List[int],
    lookback_total_counts: List[int],
):
    """
    Calculates lookback accuracy for a given timestep.

    For each prediction z_i made by model M_t, compares it only against
    targets y_0 through y_i and updates the count lists.
    """
    # Skip accuracy calculation for t=0
    if t == 0:
        return

    with torch.no_grad():
        for i in range(t + 1):
            z_i = y_pred_context[i]
            relevant_targets = y_target_sequence[: i + 1]
            similarity_scores = torch.matmul(z_i, relevant_targets.T)
            best_match_idx = torch.argmax(similarity_scores)
            
            lookback_dist = t - i
            
            if best_match_idx.item() == i:
                lookback_correct_counts[lookback_dist] += 1
            
            lookback_total_counts[lookback_dist] += 1

def general_accuracy(
    t: int,
    y_pred_context: List[torch.Tensor],
    y_target_sequence: torch.Tensor,
    correct_counts: List[int],
    total_counts: List[int],
):
    """
    Calculates accuracy for a given timestep.

    For each prediction z_i made by model M_t, compares it against the entire
    set of targets. Updates the count lists.
    """
    # Skip accuracy calculation for t=0
    if t == 0:
        return

    with torch.no_grad():
        for i in range(t + 1):
            z_i = y_pred_context[i]
            relevant_targets = y_target_sequence
            similarity_scores = torch.matmul(z_i, relevant_targets.T)
            best_match_idx = torch.argmax(similarity_scores)
            
            if best_match_idx.item() == i:
                correct_counts[i] += 1
            total_counts[i] += 1

