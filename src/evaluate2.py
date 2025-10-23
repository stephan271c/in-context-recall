"""Refactored utilities to measure recall accuracy for differentiable memory modules."""

from typing import List, Sequence, Tuple

import torch


def compute_recall_accuracies(
    predictions: List[torch.Tensor],
    values: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute hard-max recall accuracies for each timestep of a sequence with batch support.

    Refactored version where predictions are pre-computed and provided as a list.

    For every timestep ``t`` the memory module is evaluated on every key it has
    observed so far (``k_0`` through ``k_t``). The resulting accuracy tensor has
    length ``t + 1`` with the convention that index ``0`` corresponds to the
    current pair ``(k_t, v_t)``, index ``1`` to ``(k_{t-1}, v_{t-1})``, and so on.

    Args:
        predictions: List of tensors where ``predictions[t]`` is of shape
            ``(B, t+1, value_dim)`` containing pre-computed predictions for the first
            ``t+1`` keys at timestep ``t`` for batch size ``B``.
        values: Tensor of shape ``(B, sequence_length, value_dim)`` containing the
            corresponding value vectors for batch size ``B``.

    Returns:
        A list ``accuracies`` where ``accuracies[t]`` is a 2D tensor of shape
            ``(B, t + 1)`` containing the hard-max accuracy for offsets ``0..t`` at
            timestep ``t`` for each batch.

    Raises:
        ValueError: If the input tensors have incompatible shapes.
    """
    if not predictions:
        raise ValueError("predictions list must not be empty")

    sequence_length = len(predictions)

    if values.dim() != 3:
        raise ValueError("values tensor must be 3-dimensional (B, sequence_length, value_dim)")

    if values.shape[1] != sequence_length:
        raise ValueError(
            "Number of predictions must match sequence_length of values; "
            f"got {len(predictions)} and {values.shape[1]}"
        )

    value_dim = values.shape[2]
    B = values.shape[0]

    # Infer device from first prediction
    eval_device = predictions[0].device
    values = values.to(eval_device)

    evaluations: List[torch.Tensor] = []

    with torch.no_grad():
        for t in range(sequence_length):
            pred = predictions[t]
            if pred.ndim != 3 or pred.shape[0] != B or pred.shape[1] != t + 1 or pred.shape[2] != value_dim:
                raise ValueError(
                    f"predictions[{t}] must have shape ({B}, {t+1}, {value_dim}); "
                    f"received shape {tuple(pred.shape)}"
                )

            window_values = values[:, : t + 1]

            logits = torch.einsum('b t v, b s v -> b t s', pred, window_values)
            predicted_indices = logits.argmax(dim=-1)
            target_indices = torch.arange(t + 1, device=pred.device).expand(B, -1)
            per_key_accuracy = (predicted_indices == target_indices).to(torch.float32)
            offset_view = per_key_accuracy.flip(-1)  # reverses the tensor on dimension -1, to get offsets
            evaluations.append(offset_view)

    return evaluations


def average_accuracy_by_offset(
    accuracy_history: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average accuracies for a batch of sequences across timesteps by offset.

    Args:
        accuracy_history: Output of :func:`compute_recall_accuracies` for a
            single sequence. Each tensor must be one-dimensional with element ``0``
            representing the accuracy of the most recent ``(k_t, v_t)`` pair.

    Returns:
        A tuple ``(mean_accuracy, counts)`` where:

        * ``mean_accuracy`` - Tensor whose ``i``-th element is the average
          accuracy ``i`` steps prior across all applicable timesteps. Offsets
          that never occur are set to ``NaN``.
        * ``counts`` - Integer tensor containing the number of observations that
          contributed to each offset average. Note that we average/aggregate across the batch dimension as well.

    Raises:
        ValueError: If ``accuracy_history`` is empty or contains invalid tensors.
    """
    if not accuracy_history:
        raise ValueError("accuracy_history is empty")

    # Find the maximum length among the tensors
    max_len = max(len(t) for t in accuracy_history)

    # Create a padded tensor with NaN for missing values
    padded = torch.full((len(accuracy_history), max_len), float('nan'))

    for i, t in enumerate(accuracy_history):
        if t.dim() != 1:
            raise ValueError(f"Each tensor in accuracy_history must be one-dimensional, got shape {t.shape}")
        padded[i, :len(t)] = t

    # Compute mean and counts for each offset
    mean_accuracy = torch.full((max_len,), float('nan'))
    counts = torch.zeros((max_len,), dtype=torch.long)

    for i in range(max_len):
        values = padded[:, i]
        mean_accuracy[i] = torch.nanmean(values)
        counts[i] = torch.sum(~torch.isnan(values))

    return mean_accuracy, counts