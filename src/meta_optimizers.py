import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Any

class MetaOptimizer(ABC):
    """
    Abstract base class for manual, differentiable inner-loop optimizers.

    """
    def __init__(self, params: Iterable[nn.Parameter]):
        """
        Initializes the optimizer with the parameters of the model to be trained.

        Args:
            params: An iterable of torch.nn.Parameter objects.
        """
        self.params = list(params)
        if not self.params:
            raise ValueError("Optimizer received an empty list of parameters.")

    @abstractmethod
    def step(self, grads: Tuple[torch.Tensor, ...], **hyperparams: Any) -> None:
        """
        Performs a single, differentiable optimization step.

        This method must be implemented by all subclasses. The update operations
        on the parameters must remain in the PyTorch computation graph.

        Args:
            grads: A tuple of gradient tensors corresponding to self.params.
            **hyperparams: A dictionary of dynamic, tensor-valued hyperparameters
                           (e.g., lr, beta1, beta2).
        """
        raise NotImplementedError

    @abstractmethod
    def reset_states(self) -> None:
        """
        Resets all internal states of the optimizer.

        This is crucial for meta-learning, as states (like momentum) must be
        cleared at the beginning of each new task or episode.
        """
        raise NotImplementedError


class ManualAdam(MetaOptimizer):
    """A manual, differentiable implementation of the Adam optimizer."""
    def __init__(self, params: Iterable[nn.Parameter]):
        super().__init__(params)
        self.reset_states()

    def reset_states(self) -> None:
        self.m_states = {p: torch.zeros_like(p.data) for p in self.params}
        self.v_states = {p: torch.zeros_like(p.data) for p in self.params}
        self.t = 0

    def step(self, grads: Tuple[torch.Tensor, ...], **hyperparams: Any) -> None:
        self.t += 1
        lr = hyperparams['lr']
        beta1 = hyperparams['beta1']
        beta2 = hyperparams['beta2']
        epsilon = hyperparams.get('epsilon', 1e-8)

        for i, p in enumerate(self.params):
            g = grads[i]
            if g is None:
                continue
            
            self.m_states[p] = beta1 * self.m_states[p] + (1 - beta1) * g
            self.v_states[p] = beta2 * self.v_states[p] + (1 - beta2) * g.pow(2)

            m_hat = self.m_states[p] / (1 - beta1**self.t)
            v_hat = self.v_states[p] / (1 - beta2**self.t)

            update_step = m_hat / (v_hat.sqrt() + epsilon)
            p.sub_(lr * update_step)


class ManualSGD(MetaOptimizer):
    """A manual, differentiable implementation of SGD with momentum."""
    def __init__(self, params: Iterable[nn.Parameter]):
        super().__init__(params)
        self.reset_states()

    def reset_states(self) -> None:
        self.v_states = {p: torch.zeros_like(p.data) for p in self.params}

    def step(self, grads: Tuple[torch.Tensor, ...], **hyperparams: Any) -> None:
        lr = hyperparams['lr']
        beta = hyperparams.get('beta', 0.9) # Momentum coefficient

        for i, p in enumerate(self.params):
            g = grads[i]
            if g is None:
                continue

            # Update velocity
            self.v_states[p] = beta * self.v_states[p] + g
            
            # Update parameter
            p.sub_(lr * self.v_states[p])