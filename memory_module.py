import torch
import torch.nn as nn
import copy
from typing import Tuple, List, Callable, Type
    
class OnlineModule(nn.Module):
    """Wraps a model for online learning and evaluation on a per-sample basis.

    This module simulates an online learning process. The backward pass for this
    module is designed to optimize the `weight_model` parameters based on the
    outcome of the entire online learning simulation over a sequence.
    """
    def __init__(
        self, 
        model: nn.Module, 
        loss_fn: Callable, 
        optimizer_class: Type[torch.optim.Optimizer],
        optimizer_kwargs: dict, 
        eval_fn: Callable, 
        context_size: int = 1, 
        weight_model: nn.Module | None = None
        ):
        super().__init__()
        if context_size < 1:
            raise ValueError("context_size must be at least 1.")

        # The base model is stored as a template and is not trained by the external optimizer.
        self.model = model
        self.loss_fn = loss_fn
        self.eval_fn = eval_fn
        self.context_size = context_size
        self.weight_model = weight_model

        # <<< CHANGE: Store optimizer details instead of the instance
        self.optimizer_class = optimizer_class
        self.optimizer_kwargs = optimizer_kwargs
        
        # <<< CHANGE: Ensure the base model's parameters are not part of the external computation graph
        # This prevents the external optimizer from accidentally seeing and updating them.
        for param in self.model.parameters():
            param.requires_grad = False

    def forward(self, x_sequence: torch.Tensor, y_target_sequence: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, List[float], List[float]]:
        """
        Processes sequences by simulating online updates and returns a differentiable loss.

        Args:
            x_sequence (torch.Tensor): The input sequence of shape (n, d).
            y_target_sequence (torch.Tensor): The target sequence of shape (n, d).

        Returns:
            A tuple containing:
            - torch.Tensor: The final, differentiable loss for the entire sequence.
            - torch.Tensor: Predictions made at each step before the update.
            - List[float]: The scalar loss value calculated at each step.
            - List[float]: The lookback accuracy scores from the `eval_fn`.
        """
        # <<< CHANGE: Create a temporary copy of the model and its optimizer for the simulation
        online_model = copy.deepcopy(self.model)
        # Ensure the copied model's parameters require gradients for the internal simulation
        for param in online_model.parameters():
            param.requires_grad = True
        online_optimizer = self.optimizer_class(online_model.parameters(), **self.optimizer_kwargs)

        n_steps = x_sequence.shape[0]
        predictions_output = []
        losses_for_backward = [] # <<< CHANGE: Store tensors for backprop
        losses_for_logging = []  # <<< CHANGE: Store scalar values for logging
        
        lookback_correct_counts = [0] * n_steps
        lookback_total_counts = [0] * n_steps

        for t in range(n_steps):
            x_context_full = x_sequence[: t + 1]
            y_target_context_full = y_target_sequence[: t + 1]
            
            # Use the temporary online_model for predictions
            y_pred_context = online_model(x_context_full)
            
            weights = None
            if self.weight_model is not None:
                start_idx = min(t + 1, self.context_size)
                # The weight_model's computation remains part of the graph
                weights = self.weight_model(x_sequence[t])[-start_idx:]

            predictions_output.append(y_pred_context[-1].detach())
            
            loss_start_idx = max(0, t - self.context_size + 1)
            loss = self.loss_fn(
                y_pred_context[loss_start_idx : t + 1],
                y_target_context_full[loss_start_idx : t + 1],
                weights
            )
            
            losses_for_backward.append(loss)
            losses_for_logging.append(loss.item())
            
            # Update the temporary online_model
            online_optimizer.zero_grad()
            loss.backward() # This computes gradients for online_model
            online_optimizer.step() # This updates online_model's weights
            
            # --- Call the external evaluation function ---
            self.eval_fn(
                t=t,
                y_pred_context=y_pred_context,
                y_target_sequence=y_target_sequence,
                lookback_correct_counts=lookback_correct_counts,
                lookback_total_counts=lookback_total_counts
            )
        
        # --- Finalize Accuracy Scores ---
        lookback_acc = []
        for k in range(n_steps):
            if lookback_total_counts[k] > 0:
                accuracy = lookback_correct_counts[k] / lookback_total_counts[k]
                lookback_acc.append(accuracy)
            else:
                lookback_acc.append(0.0)
        
        # <<< CHANGE: Aggregate the losses into a single tensor for the external backward pass
        final_loss = torch.stack(losses_for_backward).sum()

        return final_loss, torch.stack(predictions_output), losses_for_logging, lookback_acc

    def get_model_state(self) -> dict:
        """Returns the state dictionary of the base model template."""
        return self.model.state_dict()

def lookback_accuracy_fn(
    t: int,
    y_pred_context: torch.Tensor,
    y_target_sequence: torch.Tensor,
    lookback_correct_counts: List[int],
    lookback_total_counts: List[int]
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