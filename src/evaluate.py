from typing import List, Sequence, Tuple

import torch


def compute_recall_accuracies(
    predictions: Sequence[torch.Tensor],
    values: torch.Tensor,
) -> List[torch.Tensor]:
    """Compute hard-max recall accuracies for each timestep of a sequence with batch support.

    The predictions are pre-computed and provided as a list.

    For every timestep ``t`` the memory module is evaluated on every key it has
    observed so far (``k_0`` through ``k_t``). The resulting accuracy tensor has
    length ``t + 1`` with the convention that index ``0`` corresponds to the
    first key ``(k_0, v_0)``, index ``t`` to the current pair ``(k_t, v_t)``, and so on.

    Args:
        predictions: List of tensors (whose length is equal to sequence length of the
            input) where ``predictions[t]`` is of shape ``(B, t+1, value_dim)`` containing
            pre-computed predictions for the first ``t+1`` keys at timestep ``t`` for
            batch size ``B``.
        values: Tensor of shape ``(B, sequence_length, value_dim)`` containing the
            corresponding value vectors for batch size ``B``.

    Returns:
        A list ``accuracies`` where ``accuracies[t]`` is a 2D tensor of shape
            ``(B, t + 1)`` containing the hard-max accuracy for each key position at
            timestep ``t`` for each batch. Index ``0`` corresponds to the first key,
            index ``t`` to the current key.

    Raises:
        ValueError: If the input tensors have incompatible shapes.
    """
    if not predictions:
        raise ValueError("predictions list must not be empty")

    sequence_length = len(predictions)

    if values.dim() != 3:
        raise ValueError(
            "values tensor must be 3-dimensional (B, sequence_length, value_dim)"
        )

    if values.shape[1] != sequence_length:
        raise ValueError(
            "Number of predictions must match sequence_length of values; "
            f"got {len(predictions)} and {values.shape[1]}"
        )

    value_dim = values.shape[2]
    B = values.shape[0]

    eval_device = predictions[0].device
    values = values.to(eval_device)

    evaluations: List[torch.Tensor] = []

    with torch.no_grad():
        for t in range(sequence_length):
            pred = predictions[t]
            if (
                pred.ndim != 3
                or pred.shape[0] != B
                or pred.shape[1] != t + 1
                or pred.shape[2] != value_dim
            ):
                raise ValueError(
                    f"predictions[{t}] must have shape ({B}, {t+1}, {value_dim}); "
                    f"received shape {tuple(pred.shape)}"
                )

            window_values = values[:, : t + 1]

            logits = torch.einsum("b t v, b s v -> b t s", pred, window_values)
            predicted_indices = logits.argmax(dim=-1)
            target_indices = torch.arange(t + 1, device=pred.device).expand(B, -1)
            per_key_accuracy = (predicted_indices == target_indices).to(torch.float32)
            evaluations.append(per_key_accuracy)

    return evaluations


def average_accuracy_by_offset(
    accuracy_history: Sequence[torch.Tensor],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Average accuracies for a batch of sequences across timesteps by offset.

    Args:
        accuracy_history: Output of :func:`compute_recall_accuracies` for a
            batch of sequences. Each tensor must be 2D with shape ``(B, t+1)`` where
            ``t`` is the timestep index. The function will handle offset calculation
            internally.

    Returns:
        A tuple ``(mean_accuracy, counts)`` where:

        * ``mean_accuracy`` - 1D tensor of shape ``(seq_len+1,)`` whose ``i``-th element is the average
          accuracy ``i`` steps prior across all applicable timesteps. Offset ``0``
          corresponds to the most recent ``(k_t, v_t)`` pair, offset ``1`` to
          ``(k_{t-1}, v_{t-1})``, and so on. Offsets that never occur are set to ``NaN``.
        * ``counts`` - 1D integer tensor of shape ``(seq_len+1,)`` containing the number of observations that
          contributed to each offset average. Note that we average/aggregate across the batch dimension as well.

    Raises:
        ValueError: If ``accuracy_history`` is empty or contains invalid tensors.
    """
    if not accuracy_history:
        raise ValueError("accuracy_history is empty")

    all_offset_accuracies: List[torch.Tensor] = []

    reference_tensor = accuracy_history[0]
    B = reference_tensor.shape[0]
    device = reference_tensor.device
    dtype = reference_tensor.dtype

    for t, accuracy_tensor in enumerate(accuracy_history):
        if accuracy_tensor.dim() != 2:
            raise ValueError(
                f"Each tensor in accuracy_history must be 2D with shape (B, t+1), got shape {accuracy_tensor.shape}"
            )

        # Average along batch dimension
        mean_accuracy_tensor = accuracy_tensor.mean(dim=0)  # shape (t+1,)

        # Flip along the last dimension to get offset view (offset 0 = most recent)
        offset_accuracies = mean_accuracy_tensor.flip(-1)

        all_offset_accuracies.append(offset_accuracies)

    max_len = max(len(t) for t in all_offset_accuracies) if all_offset_accuracies else 0

    if max_len == 0:
        return torch.empty(0), torch.empty(0, dtype=torch.long)

    padded = torch.full(
        (len(all_offset_accuracies), max_len), float("nan"), device=device, dtype=dtype
    )

    for i, t in enumerate(all_offset_accuracies):
        if t.dim() != 1:
            raise ValueError(
                f"Each flattened tensor must be one-dimensional, got shape {t.shape}"
            )
        padded[i, : len(t)] = t

    mean_accuracy = torch.full((max_len,), float("nan"), device=device, dtype=dtype)
    counts = torch.zeros((max_len,), dtype=torch.long, device=device)

    for i in range(max_len):
        values = padded[:, i]
        mean_accuracy[i] = torch.nanmean(values)
        counts[i] = torch.sum(~torch.isnan(values)) * B

    return mean_accuracy, counts


def correct_retrieval_counts_by_timestep(
    accuracy_history: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Count the number of correct retrievals for each timestep in a batch of sequences.

    Args:
        accuracy_history: Output of :func:`compute_recall_accuracies` for a
            batch of sequences. Each tensor must be 2D with shape ``(B, t+1)`` where
            ``t`` is the timestep index.

    Returns:
        A 1D tensor of shape ``(seq_len,)`` containing the number of correct
        retrievals for each timestep.
    """
    if not accuracy_history:
        raise ValueError("accuracy_history is empty")

    device = accuracy_history[0].device
    counts = torch.zeros(len(accuracy_history), device=device)
    for t, accuracy_tensor in enumerate(accuracy_history):
        if accuracy_tensor.dim() != 2:
            raise ValueError(
                f"Each tensor in accuracy_history must be 2D with shape (B, t+1), got shape {accuracy_tensor.shape}"
            )
        counts[t] = torch.sum(
            accuracy_tensor.mean(dim=0)
        )  # take average along batch dimension
    return counts
