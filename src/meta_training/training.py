"""Outer-loop training routines for the meta-learning system."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.optim
from torch import nn
from torch.func import functional_call, grad_and_value, vmap

from losses import windowed_p_loss, windowed_recall_cross_entropy
from meta_optimizers import MetaOptimizer

from .artifacts import MetaTrainingArtifacts
from .batch import sample_meta_batch
from .config import MetaTrainingConfig, resolve_device
from .models import build_meta_models

__all__ = ["run_meta_training"]


def _stack_state(template: Dict[str, Any], batch_size: int, device: torch.device) -> Dict[str, Any]:
    """Replicate an optimizer state ``template`` along a new batch dimension."""

    def _clone_value(value: Any) -> Any:
        if isinstance(value, torch.Tensor):
            clones = [value.detach().clone().to(device) for _ in range(batch_size)]
            return torch.stack(clones, dim=0)
        if isinstance(value, dict):
            return {k: _clone_value(v) for k, v in value.items()}
        # Integers (e.g. step counters) are shared across the batch.
        return value

    return {key: _clone_value(val) for key, val in template.items()}


def _ensure_batched(tensor: torch.Tensor, batch_size: int) -> torch.Tensor:
    """Ensure ``tensor`` has a leading batch dimension of size ``batch_size``."""

    if tensor.dim() == 0:
        return tensor.expand(batch_size)
    if tensor.size(0) == batch_size:
        return tensor
    return tensor.unsqueeze(0).expand(batch_size, *tensor.shape)


def _initialise_inner_state(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    inner_optimizer: MetaOptimizer,
) -> Tuple[nn.Module, List[str], Tuple[torch.Tensor, ...], Dict[str, Any]]:
    """Create a functional copy of ``model`` and batched optimizer states."""

    fast_model = copy.deepcopy(model).to(device)
    param_items = list(fast_model.named_parameters())
    param_names = [name for name, _ in param_items]

    def _clone_param(param: torch.Tensor) -> torch.Tensor:
        return param.detach().clone().to(device).requires_grad_(True)

    single_params: Tuple[torch.Tensor, ...] = tuple(
        _clone_param(param) for _, param in param_items
    )
    batched_params: Tuple[torch.Tensor, ...] = tuple(
        torch.stack([
            _clone_param(param) for _ in range(batch_size)
        ], dim=0)
        for _, param in param_items
    )

    base_param_dict = {name: param for name, param in zip(param_names, single_params)}
    base_state = inner_optimizer.init_states(base_param_dict)
    batched_state = _stack_state(base_state, batch_size, device)

    return fast_model, param_names, batched_params, batched_state


def _tuple_to_param_dict(param_names: List[str], params: Tuple[torch.Tensor, ...]) -> Dict[str, torch.Tensor]:
    return {name: tensor for name, tensor in zip(param_names, params)}


def _param_dict_to_tuple(param_names: List[str], params: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, ...]:
    return tuple(params[name] for name in param_names)


def _gather_context_window(
    sequences: torch.Tensor,
    time_index: int,
    window_size: int,
) -> torch.Tensor:
    """Extract the context window for each task at ``time_index``."""

    start_index = time_index - window_size + 1
    if start_index < 0:
        pad_len = -start_index
        pad_shape = (sequences.size(0), pad_len, sequences.size(2))
        padding = torch.zeros(pad_shape, device=sequences.device, dtype=sequences.dtype)
        window = sequences[:, : time_index + 1]
        return torch.cat([padding, window], dim=1)
    return sequences[:, start_index : time_index + 1]


def run_meta_training(
    config: MetaTrainingConfig,
    *,
    memory_module_factory: Optional[Callable[[MetaTrainingConfig], nn.Module]] = None,
    weight_model_override: Optional[Union[nn.Module, torch.Tensor]] = None,
    lr_model_override: Optional[Union[nn.Module, torch.Tensor, float]] = None,
    inner_loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = windowed_p_loss,
    outer_loss_fn: Optional[nn.Module] = None,
    log_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> MetaTrainingArtifacts:
    """Execute the meta-learning outer loop and return the trained modules."""
    device = resolve_device(config.device_preference)
    weight_model, memory_module, lr_model = build_meta_models(
        config,
        device,
        memory_module_factory=memory_module_factory,
        weight_model=weight_model_override,
        lr_model=lr_model_override,
    )

    outer_loss_fn = outer_loss_fn or nn.CrossEntropyLoss()

    memory_params = list(memory_module.parameters())
    weight_params = list(weight_model.parameters())
    lr_params = list(lr_model.parameters())

    for param in memory_params:
        param.requires_grad_(config.train_memory_module)
    for param in weight_params:
        param.requires_grad_(config.train_weight_model)
    for param in lr_params:
        param.requires_grad_(config.train_lr_model)

    trainable_parameters: List[nn.Parameter] = []
    if config.train_memory_module:
        trainable_parameters.extend(memory_params)
    if config.train_weight_model:
        trainable_parameters.extend(weight_params)
    if config.train_lr_model:
        trainable_parameters.extend(lr_params)

    if trainable_parameters:
        optimizer_config = config.outer_optimizer
        optimizer_kwargs: Dict[str, Any] = dict(config.outer_optimizer_kwargs)
        if isinstance(optimizer_config, torch.optim.Optimizer):
            outer_optimizer = optimizer_config
        else:
            optimizer_factory = optimizer_config or torch.optim.AdamW
            if optimizer_config is None and "lr" not in optimizer_kwargs:
                optimizer_kwargs["lr"] = 0.01
            try:
                outer_optimizer = optimizer_factory(
                    trainable_parameters, **optimizer_kwargs
                )
            except TypeError as exc:
                if optimizer_kwargs:
                    raise TypeError(
                        "outer_optimizer could not be called with the provided "
                        "outer_optimizer_kwargs."
                    ) from exc
                outer_optimizer = optimizer_factory(trainable_parameters)
    else:
        outer_optimizer = None

    inner_optimizer = copy.deepcopy(config.inner_optimizer)
    base_inner_hparams = dict(config.inner_optimizer_kwargs)
    history: List[float] = []

    # One meta-step is one batch of sequences processed through the inner loop
    for meta_step in range(config.total_meta_updates):
        batch = sample_meta_batch(config, device)
        (
            fast_model,
            param_names,
            param_batches,
            state,
        ) = _initialise_inner_state(
            memory_module, config.batch_size, device, inner_optimizer
        )

        keys_batch = torch.stack([item.keys for item in batch], dim=0)
        values_batch = torch.stack([item.values for item in batch], dim=0)

        if outer_optimizer is not None:
            outer_optimizer.zero_grad()
        total_outer_loss = torch.zeros((), device=device)

        param_in_dims = tuple(0 for _ in param_batches)

        def inner_objective(
            params: Tuple[torch.Tensor, ...],
            key_window: torch.Tensor,
            value_window: torch.Tensor,
            weights: torch.Tensor,
        ) -> torch.Tensor:
            param_dict = _tuple_to_param_dict(param_names, params)
            predictions = functional_call(fast_model, param_dict, key_window)
            return inner_loss_fn(predictions.T, value_window.T, weights)

        inner_grad_fn = grad_and_value(inner_objective)

        for time_index in range(config.seq_len):
            context_keys = _gather_context_window(
                keys_batch, time_index, config.context_dim
            )
            context_vals = _gather_context_window(
                values_batch, time_index, config.context_dim
            )

            query_keys = context_keys[:, -1, :]
            loss_weights = weight_model(query_keys)
            lr_values = lr_model(query_keys)

            loss_weights = _ensure_batched(loss_weights, config.batch_size)
            lr_values = _ensure_batched(lr_values, config.batch_size)

            hyperparams = dict(base_inner_hparams)
            hyperparams["lr"] = lr_values

            inner_grads_tuple, _ = vmap(
                inner_grad_fn, (param_in_dims, 0, 0, 0)
            )(param_batches, context_keys, context_vals, loss_weights)

            params_dict = _tuple_to_param_dict(param_names, param_batches)
            grads_dict = _tuple_to_param_dict(param_names, inner_grads_tuple)

            updated_params_dict, state = inner_optimizer.step(
                params_dict, grads_dict, state, **hyperparams
            )
            param_batches = _param_dict_to_tuple(param_names, updated_params_dict)

            def compute_outer_loss(
                params: Tuple[torch.Tensor, ...],
                keys: torch.Tensor,
                values: torch.Tensor,
            ) -> torch.Tensor:
                param_dict = _tuple_to_param_dict(param_names, params)
                return windowed_recall_cross_entropy(
                    fast_model,
                    param_dict,
                    keys,
                    values,
                    time_index=time_index,
                    window_size=config.recall_window,
                    loss_fn=outer_loss_fn,
                )

            outer_losses = vmap(
                compute_outer_loss, (param_in_dims, 0, 0)
            )(param_batches, keys_batch, values_batch)

            total_outer_loss = total_outer_loss + outer_losses.sum()

        avg_outer_loss = total_outer_loss / (config.seq_len * config.batch_size)
        history.append(float(avg_outer_loss.detach()))
        if outer_optimizer is not None:
            avg_outer_loss.backward()
            outer_optimizer.step()

        processed_sequences = (meta_step + 1) * config.batch_size
        should_log = meta_step == 0
        if config.log_every_sequences:
            should_log = should_log or (
                processed_sequences % config.log_every_sequences == 0
            )
        if should_log:
            sample_key = batch[0].keys[0].unsqueeze(0)
            sample_lr = float(lr_model(sample_key).item())
            if log_callback is not None:
                log_callback(
                    meta_step,
                    processed_sequences,
                    float(avg_outer_loss.item()),
                    sample_lr,
                )
            else:
                print(
                    f"Epoch {processed_sequences} | Avg Outer Loss: {avg_outer_loss.item():.4f}"
                )
                print(f"  Sample Hyperparams -> LR: {sample_lr:.4f}")

    return MetaTrainingArtifacts(
        memory_module=memory_module,
        weight_model=weight_model,
        lr_model=lr_model,
        outer_losses=history,
    )
