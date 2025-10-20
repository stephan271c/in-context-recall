"""Model construction helpers for the meta-training workflow."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Tuple, Union

import torch
from torch import nn

from func_memory_module import HyperparamModel, TTT, WeightModel

from .config import MetaTrainingConfig, MemoryModuleFactory

__all__ = ["build_meta_models"]


def _default_memory_module_factory(config: MetaTrainingConfig) -> nn.Module:
    """Instantiate the default memory module when no custom factory is provided."""
    return TTT(config.key_dim, config.val_dim)


class ConstantOutputModule(nn.Module):
    """Wrap a tensor so it can be reused as an nn.Module output."""

    def __init__(self, value: Union[torch.Tensor, float]):
        super().__init__()
        tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        self.register_buffer("value", tensor_value.clone().detach())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.value


def build_meta_models(
    config: MetaTrainingConfig,
    device: torch.device,
    *,
    memory_module_factory: Optional[MemoryModuleFactory] = None,
    weight_model: Optional[Union[nn.Module, torch.Tensor]] = None,
    lr_model: Optional[Union[nn.Module, torch.Tensor, float]] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Instantiate the meta-learner components on the requested device."""
    if weight_model is None:
        weight_model_module: nn.Module = WeightModel(config.key_dim, config.context_dim)
    elif isinstance(weight_model, torch.Tensor):
        weight_model_module = ConstantOutputModule(weight_model)
    elif isinstance(weight_model, nn.Module):
        weight_model_module = weight_model
    else:
        raise TypeError(
            "weight_model must be an nn.Module or tensor when provided; "
            f"received type {type(weight_model).__name__}"
        )
    weight_model_module = weight_model_module.to(device)

    factory = memory_module_factory or _default_memory_module_factory
    memory_module = factory(config).to(device)
    if lr_model is None:
        lr_model_module: nn.Module = HyperparamModel(
            config.key_dim, initial_bias=config.hyper_lr_initial_bias
        )
    elif isinstance(lr_model, (torch.Tensor, float)):
        lr_model_module = ConstantOutputModule(lr_model)
    elif isinstance(lr_model, nn.Module):
        lr_model_module = lr_model
    else:
        raise TypeError(
            "lr_model must be an nn.Module, tensor, or float when provided; "
            f"received type {type(lr_model).__name__}"
        )
    lr_model_module = lr_model_module.to(device)

    return weight_model_module, memory_module, lr_model_module
