"""Outer-loop training routines for the meta-learning system."""

from __future__ import annotations

import copy
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import torch
import torch.optim
from torch import nn
from torch.func import functional_call

from losses import windowed_p_loss, windowed_recall_cross_entropy
from meta_optimizers import MetaOptimizer

from .artifacts import MetaTrainingArtifacts
from .batch import sample_meta_batch
from .config import MetaTrainingConfig, resolve_device
from .models import build_meta_models

__all__ = ["run_meta_training"]


def _initialise_inner_state(
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    inner_optimizer: MetaOptimizer,
) -> Tuple[nn.Module, List[Dict[str, torch.Tensor]], List[Dict[str, Any]]]:
    """Create a functional copy of ``model`` and optimizer states per task."""
    fast_model = copy.deepcopy(model).to(device)
    param_sets: List[Dict[str, torch.Tensor]] = []
    state_list: List[Dict[str, Any]] = []
    for _ in range(batch_size):
        params: Dict[str, torch.Tensor] = {}
        for name, param in fast_model.named_parameters():
            cloned = param.detach().clone().requires_grad_(True)
            params[name] = cloned
        param_sets.append(params)
        state_list.append(inner_optimizer.init_states(params))
    return fast_model, param_sets, state_list


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

    for meta_step in range(config.total_meta_updates):
        batch = sample_meta_batch(config, device)
        fast_model, param_sets, state_list = _initialise_inner_state(
            memory_module, config.batch_size, device, inner_optimizer
        )

        if outer_optimizer is not None:
            outer_optimizer.zero_grad()
        total_outer_loss = torch.zeros((), device=device)

        for time_index in range(config.seq_len):
            for task_idx, item in enumerate(batch):
                current_key, current_val = item.dataset[time_index]
                current_key = current_key.to(device)
                current_val = current_val.to(device)

                loss_weights = weight_model(current_key[-1])
                hyperparams = dict(base_inner_hparams)
                hyperparams["lr"] = lr_model(current_key[-1])

                params = param_sets[task_idx]
                state = state_list[task_idx]

                predictions = functional_call(fast_model, params, current_key)
                inner_loss = inner_loss_fn(predictions.T, current_val.T, loss_weights)
                grad_tuple = torch.autograd.grad(
                    inner_loss, tuple(params.values()), create_graph=True
                )
                grads = dict(zip(params.keys(), grad_tuple))

                updated_params, updated_state = inner_optimizer.step(
                    params, grads, state, **hyperparams
                )
                param_sets[task_idx] = updated_params
                state_list[task_idx] = updated_state

                outer_loss_step = windowed_recall_cross_entropy(
                    fast_model,
                    updated_params,
                    item.keys,
                    item.values,
                    time_index=time_index,
                    window_size=config.recall_window,
                    loss_fn=outer_loss_fn,
                )
                total_outer_loss = total_outer_loss + outer_loss_step

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
