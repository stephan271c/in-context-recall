"""Utilities to measure recall accuracy for differentiable memory modules."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.func import functional_call


__all__ = [
    "compute_recall_accuracies",
    "average_accuracy_by_offset",
    "average_accuracy_across_sequences",
]


def compute_recall_accuracies(
    model: nn.Module,
    keys: torch.Tensor,
    values: torch.Tensor,
    params: Optional[Dict[str, torch.Tensor]] = None,
) -> List[torch.Tensor]:
    """Compute hard-max recall accuracies for each timestep of a sequence.

    For every timestep ``t`` the memory module is evaluated on every key it has
    observed so far (``k_0`` through ``k_t``). The resulting accuracy tensor has
    length ``t + 1`` with the convention that index ``0`` corresponds to the
    current pair ``(k_t, v_t)``, index ``1`` to ``(k_{t-1}, v_{t-1})``, and so on.

    Args:
        model: Memory module that maps keys to value predictions.
        keys: Tensor of shape ``(sequence_length, key_dim)`` describing the key
            sequence encountered by the model.
        values: Tensor of shape ``(sequence_length, value_dim)`` containing the
            corresponding value vectors.
        params: Optional parameter dictionary for ``functional_call``. Supply
            this when evaluating a functionalized copy of ``model`` (e.g., during
            meta-learning inner loops).

    Returns:
        A list ``accuracies`` where ``accuracies[t]`` is a 1D tensor of length
        ``t + 1`` containing the hard-max accuracy for offsets ``0..t`` at
        timestep ``t``.

    Raises:
        ValueError: If the input tensors have incompatible shapes.
    """
    if keys.dim() != 2:
        raise ValueError("keys tensor must be 2-dimensional (sequence_length, key_dim)")
    if values.dim() != 2:
        raise ValueError("values tensor must be 2-dimensional (sequence_length, value_dim)")
    if keys.shape[0] != values.shape[0]:
        raise ValueError(
            "keys and values must describe the same number of timesteps; "
            f"got {keys.shape[0]} and {values.shape[0]}"
        )

    sequence_length = keys.shape[0]
    eval_device = _infer_evaluation_device(model, params)
    keys = keys.to(eval_device)
    values = values.to(eval_device)

    evaluations: List[torch.Tensor] = []

    with torch.no_grad():
        for t in range(sequence_length):
            window_keys = keys[: t + 1]
            window_values = values[: t + 1]

            predictions = _call_model(model, params, window_keys)
            if predictions.ndim != 2 or predictions.shape[0] != t + 1:
                raise ValueError(
                    "model must return a 2D tensor with one prediction per key; "
                    f"received shape {tuple(predictions.shape)} at timestep {t}"
                )
            if predictions.shape[1] != values.shape[1]:
                raise ValueError(
                    "prediction vector dimension must match value dimension; "
                    f"got {predictions.shape[1]} and {values.shape[1]}"
                )

            logits = predictions @ window_values.T
            predicted_indices = logits.argmax(dim=-1)
            target_indices = torch.arange(t + 1, device=predictions.device)
            per_key_accuracy = (predicted_indices == target_indices).to(torch.float32)
            offset_view = per_key_accuracy.flip(0)
            evaluations.append(offset_view)

    return evaluations


def average_accuracy_by_offset(
    accuracy_history: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average accuracies for a single sequence across timesteps by offset.

    Args:
        accuracy_history: Output of :func:`compute_recall_accuracies` for a
            single sequence. Each tensor must be one-dimensional with element ``0``
            representing the accuracy of the most recent ``(k_t, v_t)`` pair.

    Returns:
        A tuple ``(mean_accuracy, counts)`` where:

        * ``mean_accuracy`` – Tensor whose ``i``-th element is the average
          accuracy ``i`` steps prior across all applicable timesteps. Offsets
          that never occur are set to ``NaN``.
        * ``counts`` – Integer tensor containing the number of observations that
          contributed to each offset average.

    Raises:
        ValueError: If ``accuracy_history`` is empty or contains invalid tensors.
    """
    if not accuracy_history:
        raise ValueError("accuracy_history must contain at least one timestep tensor")

    max_len = max(tensor.shape[0] for tensor in accuracy_history)
    if max_len == 0:
        raise ValueError("accuracy tensors must be one-dimensional and non-empty")

    device = accuracy_history[0].device
    totals = torch.zeros(max_len, dtype=torch.float32, device=device)
    counts = torch.zeros(max_len, dtype=torch.int64, device=device)

    for offset_tensor in accuracy_history:
        if offset_tensor.ndim != 1:
            raise ValueError("each accuracy tensor must be one-dimensional")
        length = offset_tensor.shape[0]
        totals[:length] += offset_tensor.to(device=totals.device, dtype=totals.dtype)
        counts[:length] += 1

    mean = totals / counts.clamp(min=1).to(totals.dtype)
    mean = mean.masked_fill(counts == 0, float("nan"))
    return mean, counts


def average_accuracy_across_sequences(
    batch_histories: Sequence[Sequence[torch.Tensor]],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Aggregate recall accuracies across multiple sequences.

    Args:
        batch_histories: Iterable where each element is the output of
            :func:`compute_recall_accuracies` for a different sequence.

    Returns:
        A tuple ``(mean_accuracy, counts)`` summarizing all sequences. Interpretation
        matches :func:`average_accuracy_by_offset` but now aggregated over the
        entire batch.

    Raises:
        ValueError: If no accuracy tensors are provided.
    """
    first_tensor = _find_first_tensor(batch_histories)
    if first_tensor is None:
        raise ValueError("batch_histories must contain at least one accuracy tensor")

    max_len = 0
    for history in batch_histories:
        for tensor in history:
            if tensor.ndim != 1:
                raise ValueError("each accuracy tensor must be one-dimensional")
            if tensor.shape[0] == 0:
                raise ValueError("accuracy tensors must not be empty")
            max_len = max(max_len, tensor.shape[0])

    totals = torch.zeros(max_len, dtype=torch.float32, device=first_tensor.device)
    counts = torch.zeros(max_len, dtype=torch.int64, device=first_tensor.device)

    for history in batch_histories:
        for offset_tensor in history:
            length = offset_tensor.shape[0]
            totals[:length] += offset_tensor.to(device=totals.device, dtype=totals.dtype)
            counts[:length] += 1

    mean = totals / counts.clamp(min=1).to(totals.dtype)
    mean = mean.masked_fill(counts == 0, float("nan"))
    return mean, counts


def _call_model(
    model: nn.Module,
    params: Optional[Dict[str, torch.Tensor]],
    inputs: torch.Tensor,
) -> torch.Tensor:
    """Evaluate ``model`` using either ``params`` or its registered parameters."""
    if params is None:
        predictions = model(inputs)
    else:
        predictions = functional_call(model, params, inputs)

    if predictions.ndim == 1:
        predictions = predictions.unsqueeze(0)
    return predictions


def _infer_evaluation_device(
    model: nn.Module,
    params: Optional[Dict[str, torch.Tensor]],
) -> torch.device:
    """Choose the device to run evaluation on, preferring ``params`` when set."""
    if params:
        return next(iter(params.values())).device

    try:
        first_param = next(model.parameters())
        return first_param.device
    except StopIteration:
        return torch.device("cpu")


def _find_first_tensor(
    batch_histories: Sequence[Sequence[torch.Tensor]],
) -> Optional[torch.Tensor]:
    for history in batch_histories:
        for tensor in history:
            return tensor
    return None

