"""Container types returned by the meta-training routines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

from torch import nn

__all__ = ["MetaTrainingArtifacts"]


@dataclass
class MetaTrainingArtifacts:
    """Trained models alongside simple training history."""

    memory_module: nn.Module
    weight_model: nn.Module
    lr_model: nn.Module
    outer_losses: List[float]
