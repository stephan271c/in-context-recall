"""Helpers for constructing inner and outer optimizers."""

from __future__ import annotations

from typing import Any, Callable, Dict, Optional, Sequence, Type

import torch
from torch import nn

from meta_optimizers import ManualAdam, ManualAdamW, ManualSGD, MetaOptimizer

INNER_OPTIMIZER_REGISTRY: Dict[str, Type[MetaOptimizer]] = {
    "manual_adam": ManualAdam,
    "manual_adamw": ManualAdamW,
    "manual_sgd": ManualSGD,
}


def _build_inner_optimizer(name: str) -> MetaOptimizer:
    """Return an instantiated meta-optimizer from the configured registry."""
    key = name.lower()
    try:
        optimizer_cls = INNER_OPTIMIZER_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(INNER_OPTIMIZER_REGISTRY))
        raise ValueError(
            f"Unknown inner optimizer '{name}'. Available options: {available}"
        ) from exc
    return optimizer_cls()


def _build_outer_optimizer(
    params: Sequence[nn.Parameter],
    factory: Optional[Callable[..., torch.optim.Optimizer]],
    kwargs: Dict[str, Any],
) -> Optional[torch.optim.Optimizer]:
    """Construct the outer-loop optimizer, defaulting to AdamW when omitted."""
    if not params:
        return None

    optimizer_factory: Callable[..., torch.optim.Optimizer]
    optimizer_kwargs: Dict[str, Any]
    if factory is None:
        optimizer_factory = torch.optim.AdamW
        optimizer_kwargs = {"lr": 0.01}
    else:
        optimizer_factory = factory
        optimizer_kwargs = {}

    optimizer_kwargs.update(kwargs)

    try:
        return optimizer_factory(params, **optimizer_kwargs)
    except TypeError as exc:
        if optimizer_kwargs:
            raise TypeError(
                "outer_optimizer_factory could not be called with the provided "
                "outer_optimizer_kwargs."
            ) from exc
        # Retry without kwargs for callables that already capture configuration.
        return optimizer_factory(params)
