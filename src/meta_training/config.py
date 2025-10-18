"""Configuration dataclasses and device resolution helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional

import torch
from torch import nn

__all__ = [
    "MetaTrainingConfig",
    "MemoryModuleFactory",
    "EvaluationConfig",
    "resolve_device",
]


@dataclass
class MetaTrainingConfig:
    """Configuration options that control the outer-loop optimisation."""

    key_dim: int = 16
    val_dim: int = 16
    context_dim: int = 5
    seq_len: int = 50
    num_sequences: int = 500
    batch_size: int = 10
    recall_window: int = 1
    output_corr: float = 0.5
    device_preference: str = "cuda"
    hyper_lr_initial_bias: float = -2.0
    log_every_sequences: int = 50
    train_memory_module: bool = True
    train_weight_model: bool = True
    train_lr_model: bool = True
    inner_optimizer_name: str = "manual_adam"
    inner_optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"beta1": 0.95, "beta2": 0.99, "epsilon": 1e-8}
    )
    outer_optimizer_factory: Optional[Callable[..., torch.optim.Optimizer]] = None
    outer_optimizer_kwargs: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.num_sequences % self.batch_size:
            raise ValueError("batch_size must divide num_sequences for full epochs")

    @property
    def total_meta_updates(self) -> int:
        return self.num_sequences // self.batch_size


MemoryModuleFactory = Callable[[MetaTrainingConfig], nn.Module]


@dataclass
class EvaluationConfig:
    """Settings used when evaluating trained memory modules."""

    seq_len: int
    num_sequences: int = 20
    key_dim: Optional[int] = None
    val_dim: Optional[int] = None
    context_dim: Optional[int] = None
    output_corr: float = 0.5


def resolve_device(device_preference: str) -> torch.device:
    """Pick an appropriate torch.device for the current machine."""
    if device_preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_preference)
