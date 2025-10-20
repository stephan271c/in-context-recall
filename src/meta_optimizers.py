import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import Iterable, Tuple, Any, Dict


def _match_batch_shape(value: Any, reference: torch.Tensor) -> Any:
    """Broadcast ``value`` so it is compatible with ``reference`` shapes."""
    if not isinstance(value, torch.Tensor):
        return value
    if value.device != reference.device:
        value = value.to(reference.device)
    extra_dims = reference.dim() - value.dim()
    if extra_dims < 0:
        raise ValueError(
            "Hyperparameter tensor has more dimensions than reference parameter."
        )
    if extra_dims > 0:
        value = value.view(value.shape + (1,) * extra_dims)
    return value

class MetaOptimizer(ABC):
    """
    Abstract base class for manual, differentiable inner-loop optimizers.

    """
    @abstractmethod
    def init_states(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        """
        Initializes the states for the optimizer given initial parameters.

        Args:
            params: Dict of {param_name: tensor} initial parameter tensors.

        Returns:
            Dict containing initial states (e.g., momentum, variance).
        """
        pass

    @abstractmethod
    def step(self, params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], states: Dict[str, Any], **hyperparams: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        """
        Performs a single, differentiable optimization step functionally.

        This method computes new parameters and updated states without in-place
        modifications, keeping everything in the computation graph.

        Args:
            params: Current dict of {param_name: tensor}.
            grads: Dict of {param_name: gradient_tensor}.
            states: Current optimizer states.
            **hyperparams: Dictionary of dynamic hyperparameters (e.g., lr).

        Returns:
            Tuple of (new_params: Dict[str, torch.Tensor], new_states: Dict[str, Any])
        """
        pass


class ManualAdam(MetaOptimizer):
    """A manual, differentiable implementation of the Adam optimizer."""
    def __init__(self):
        pass

    def init_states(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            'm': {name: torch.zeros_like(p) for name, p in params.items()},
            'v': {name: torch.zeros_like(p) for name, p in params.items()},
            't': 0
        }

    def step(self, params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], states: Dict[str, Any], **hyperparams: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        m = states['m']
        v = states['v']
        t = states['t'] + 1
        lr = hyperparams['lr']
        beta1 = hyperparams['beta1']
        beta2 = hyperparams['beta2']
        epsilon = hyperparams.get('epsilon', 1e-8)

        new_params = {}
        new_m = {}
        new_v = {}
        beta1_pow_t = beta1 ** t
        beta2_pow_t = beta2 ** t

        for name, p in params.items():
            g = grads.get(name, None)
            if g is None:
                new_params[name] = p
                new_m[name] = m[name]
                new_v[name] = v[name]
                continue

            m_i = beta1 * m[name] + (1 - beta1) * g
            v_i = beta2 * v[name] + (1 - beta2) * (g ** 2)

            m_hat = m_i / (1 - beta1_pow_t)
            v_hat = v_i / (1 - beta2_pow_t)

            update_step = m_hat / (v_hat.sqrt() + epsilon)
            lr_param = _match_batch_shape(lr, p)
            new_p = p - lr_param * update_step

            new_params[name] = new_p
            new_m[name] = m_i
            new_v[name] = v_i

        new_states = {
            'm': new_m,
            'v': new_v,
            't': t
        }
        return new_params, new_states


class ManualSGD(MetaOptimizer):
    """A manual, differentiable implementation of SGD with momentum."""
    def __init__(self):
        pass
    def init_states(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            'v': {name: torch.zeros_like(p) for name, p in params.items()}
        }

    def step(self, params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], states: Dict[str, Any], **hyperparams: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        v = states['v']
        lr = hyperparams['lr']
        beta = hyperparams.get('beta', 0.9)

        new_v = {}
        new_params = {}

        for name, p in params.items():
            g = grads.get(name, None)
            if g is None:
                new_params[name] = p
                new_v[name] = v[name]
                continue

            v_i = beta * v[name] + g
            lr_param = _match_batch_shape(lr, p)
            new_p = p - lr_param * v_i

            new_params[name] = new_p
            new_v[name] = v_i

        new_states = {'v': new_v}
        return new_params, new_states

class ManualAdamW(MetaOptimizer):
    """A manual, differentiable implementation of the AdamW optimizer with decoupled weight decay."""
    def __init__(self):
        pass

    def init_states(self, params: Dict[str, torch.Tensor]) -> Dict[str, Any]:
        return {
            'm': {name: torch.zeros_like(p) for name, p in params.items()},
            'v': {name: torch.zeros_like(p) for name, p in params.items()},
            't': 0
        }

    def step(self, params: Dict[str, torch.Tensor], grads: Dict[str, torch.Tensor], states: Dict[str, Any], **hyperparams: Any) -> Tuple[Dict[str, torch.Tensor], Dict[str, Any]]:
        m = states['m']
        v = states['v']
        t = states['t'] + 1
        lr = hyperparams['lr']
        beta1 = hyperparams['beta1']
        beta2 = hyperparams['beta2']
        epsilon = hyperparams.get('epsilon', 1e-8)
        weight_decay = hyperparams.get('weight_decay', 0.0)

        new_params = {}
        new_m = {}
        new_v = {}
        beta1_pow_t = beta1 ** t
        beta2_pow_t = beta2 ** t

        for name, p in params.items():
            g = grads.get(name, None)
            if g is None:
                # Apply only weight decay if no gradient
                lr_param = _match_batch_shape(lr, p)
                weight_decay_param = _match_batch_shape(weight_decay, p)
                new_p = p - lr_param * weight_decay_param * p
                new_params[name] = new_p
                new_m[name] = m[name]
                new_v[name] = v[name]
                continue

            # Update moments with raw gradient only
            m_i = beta1 * m[name] + (1 - beta1) * g
            v_i = beta2 * v[name] + (1 - beta2) * (g ** 2)

            # Bias correction
            m_hat = m_i / (1 - beta1_pow_t)
            v_hat = v_i / (1 - beta2_pow_t)

            # Adam update term
            adam_step = m_hat / (v_hat.sqrt() + epsilon)

            # Decoupled weight decay term
            weight_decay_param = _match_batch_shape(weight_decay, p)
            wd_step = weight_decay_param * p

            # Combined update
            update_step = adam_step + wd_step
            lr_param = _match_batch_shape(lr, p)
            new_p = p - lr_param * update_step

            new_params[name] = new_p
            new_m[name] = m_i
            new_v[name] = v_i

        new_states = {
            'm': new_m,
            'v': new_v,
            't': t
        }
        return new_params, new_states
