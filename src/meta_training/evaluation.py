"""Evaluation utilities for trained memory modules."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence

import torch
from torch import nn

from evaluate_functions import average_accuracy_across_sequences, compute_recall_accuracies
from synthetic_datasets import InContextRecallDataset

from .config import EvaluationConfig

__all__ = ["EvaluationResult", "evaluate_memory_module"]


@dataclass
class EvaluationResult:
    """Aggregated recall accuracy statistics across evaluation sequences."""

    offsets: torch.Tensor
    mean_accuracy: torch.Tensor
    counts: torch.Tensor

    def cpu(self) -> "EvaluationResult":
        """Return a copy of the result with tensors moved to the CPU."""
        return EvaluationResult(
            offsets=self.offsets.cpu(),
            mean_accuracy=self.mean_accuracy.cpu(),
            counts=self.counts.cpu(),
        )


def evaluate_memory_module(
    memory_module: nn.Module,
    config: EvaluationConfig,
    *,
    device: Optional[torch.device] = None,
) -> EvaluationResult:
    """Evaluate recall accuracy across newly sampled sequences."""
    if device is None:
        try:
            first_param = next(memory_module.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cpu")

    key_dim = (
        config.key_dim
        if config.key_dim is not None
        else getattr(memory_module, "input_dim", None)
    )
    if key_dim is None:
        raise ValueError(
            "EvaluationConfig.key_dim must be provided when the memory_module "
            "does not define an 'input_dim' attribute."
        )
    val_dim = (
        config.val_dim
        if config.val_dim is not None
        else getattr(memory_module, "output_dim", None)
    )
    if val_dim is None:
        raise ValueError(
            "EvaluationConfig.val_dim must be provided when the memory_module "
            "does not define an 'output_dim' attribute."
        )
    key_dim = int(key_dim)
    val_dim = int(val_dim)
    if config.context_dim is None:
        raise ValueError("EvaluationConfig.context_dim must be specified for evaluation")
    context_dim = config.context_dim

    histories: List[Sequence[torch.Tensor]] = []
    was_training = memory_module.training
    memory_module.eval()
    with torch.no_grad():
        for _ in range(config.num_sequences):
            dataset = InContextRecallDataset(
                seq_len=config.seq_len,
                key_dim=key_dim,
                val_dim=val_dim,
                context_size=context_dim,
                output_corr=config.output_corr,
            )
            keys = dataset.inputs.to(device)
            values = dataset.targets.to(device)
            history = compute_recall_accuracies(memory_module, keys, values)
            histories.append(history)
    if was_training:
        memory_module.train()

    mean_accuracy, counts = average_accuracy_across_sequences(histories)
    offsets = torch.arange(mean_accuracy.shape[0], device=mean_accuracy.device)
    return EvaluationResult(offsets=offsets, mean_accuracy=mean_accuracy, counts=counts)
