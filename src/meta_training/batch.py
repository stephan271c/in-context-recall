"""Data structures and helpers for sampling meta-training batches."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List

import torch

from synthetic_datasets import InContextRecallDataset

from .config import MetaTrainingConfig

__all__ = ["MetaBatchItem", "sample_meta_batch"]


@dataclass
class MetaBatchItem:
    """In-memory representation of a sampled sequence for the inner loop."""

    dataset: InContextRecallDataset
    keys: torch.Tensor
    values: torch.Tensor


def sample_meta_batch(config: MetaTrainingConfig, device: torch.device) -> List[MetaBatchItem]:
    """Generate a fresh batch of synthetic recall tasks."""
    batch: List[MetaBatchItem] = []
    for _ in range(config.batch_size):
        dataset = InContextRecallDataset(
            seq_len=config.seq_len,
            key_dim=config.key_dim,
            val_dim=config.val_dim,
            context_size=config.context_dim,
            output_corr=config.output_corr,
        )
        batch.append(
            MetaBatchItem(
                dataset=dataset,
                keys=dataset.inputs.to(device),
                values=dataset.targets.to(device),
            )
        )
    return batch
