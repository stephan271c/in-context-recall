from typing import Callable, Dict, List, Tuple, Union

import torch
from torch.utils.data import Dataset


class InContextRecallDataset(Dataset):
    def __init__(
        self,
        seq_len: int,
        key_dim: int,
        val_dim: int,
        context_size: int,
        input_corr: float = 0.0,
        output_corr: float = 0.0,
    ) -> None:
        self.seq_len = seq_len
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.context_size = context_size
        self.input_corr = input_corr
        self.output_corr = output_corr
        self.inputs = generate_vectors(seq_len, key_dim, input_corr)
        self.targets = generate_vectors(seq_len, val_dim, output_corr)

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        # Case 1: The index is a single integer
        """
        returns a tuple (input_window, target_window) where:
        - input_window is a tensor of shape (context_size, key_dim)
        - target_window is a tensor of shape (context_size, val_dim)
        """
        if isinstance(idx, int):
            start_idx = idx - self.context_size + 1
            if start_idx < 0:
                padding_len = abs(start_idx)
                device = self.inputs.device
                input_padding = torch.zeros(padding_len, self.key_dim, device=device)
                target_padding = torch.zeros(padding_len, self.val_dim, device=device)
                input_window = torch.cat((input_padding, self.inputs[: idx + 1]), dim=0)
                target_window = torch.cat(
                    (target_padding, self.targets[: idx + 1]), dim=0
                )
            else:
                input_window = self.inputs[start_idx : idx + 1]
                target_window = self.targets[start_idx : idx + 1]
            return input_window, target_window

        # Case 2: The index is a slice
        elif isinstance(idx, slice):

            input_batch = self.inputs[idx]
            target_batch = self.targets[idx]

            return input_batch, target_batch

        else:
            raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        return self.seq_len


class BatchedInContextRecallDataset(Dataset):
    """
    Generate a batch of independent in-context recall sequences.

    Each batch element mirrors :class:`InContextRecallDataset`, but the inputs
    and targets are stacked along a leading batch dimension so code consuming the
    dataset can operate on all sequences in parallel.
    """

    def __init__(
        self,
        seq_len: int,
        key_dim: int,
        val_dim: int,
        context_size: int,
        input_corr: float = 0.0,
        output_corr: float = 0.0,
        batch_size: int = 1,
    ) -> None:
        if batch_size < 1:
            raise ValueError("batch_size must be a positive integer.")

        self.seq_len = seq_len
        self.key_dim = key_dim
        self.val_dim = val_dim
        self.context_size = context_size
        self.input_corr = input_corr
        self.output_corr = output_corr
        self.batch_size = batch_size

        inputs = []
        targets = []
        for _ in range(batch_size):
            inputs.append(generate_vectors(seq_len, key_dim, input_corr))
            targets.append(generate_vectors(seq_len, val_dim, output_corr))

        self.inputs = torch.stack(inputs, dim=0)
        self.targets = torch.stack(targets, dim=0)

    def __getitem__(self, idx: Union[int, slice]) -> Tuple[torch.Tensor, torch.Tensor]:
        if isinstance(idx, int):
            start_idx = idx - self.context_size + 1
            device = self.inputs.device

            if start_idx < 0:
                pad_len = -start_idx
                input_padding = torch.zeros(
                    self.batch_size, pad_len, self.key_dim, device=device
                )
                target_padding = torch.zeros(
                    self.batch_size, pad_len, self.val_dim, device=device
                )
                input_window = torch.cat(
                    (input_padding, self.inputs[:, : idx + 1]),
                    dim=1,
                )
                target_window = torch.cat(
                    (target_padding, self.targets[:, : idx + 1]),
                    dim=1,
                )
            else:
                input_window = self.inputs[:, start_idx : idx + 1]
                target_window = self.targets[:, start_idx : idx + 1]

            if input_window.shape[1] < self.context_size:
                pad_len = self.context_size - input_window.shape[1]
                pad_inputs = torch.zeros(
                    self.batch_size, pad_len, self.key_dim, device=device
                )
                pad_targets = torch.zeros(
                    self.batch_size, pad_len, self.val_dim, device=device
                )
                input_window = torch.cat((pad_inputs, input_window), dim=1)
                target_window = torch.cat((pad_targets, target_window), dim=1)

            return input_window, target_window

        if isinstance(idx, slice):
            return self.inputs[:, idx], self.targets[:, idx]

        raise TypeError("Invalid argument type.")

    def __len__(self) -> int:
        return self.seq_len

    def to(self, device: Union[str, torch.device]) -> "BatchedInContextRecallDataset":
        """
        Move all tensors in the dataset to the specified device.

        Args:
            device: The device to move the tensors to (e.g., 'cpu', 'cuda', 'cuda:0')

        Returns:
            Self for method chaining
        """
        self.inputs = self.inputs.to(device)
        self.targets = self.targets.to(device)
        return self


def generate_vectors(
    num_examples: int,
    dim: int,
    correlation: float = 0.0,
) -> torch.Tensor:
    """
    Generates (num_examples) unit norm vectors of length (dim).

    Args:
        num_examples: The number of vector pairs to generate (number of rows).
        dim: The dimension of each individual vector x_i
        correlation: The expected correlation between vector x_i and x_i+1.
    Returns:
        A tensor of shape (num_examples, dim).
    """
    if not 0.0 <= correlation <= 1.0:
        raise ValueError("Correlation must be between 0.0 and 1.0")

    if correlation == 0.0:
        random_vectors = torch.randn(num_examples, dim)
    else:
        random_vectors = torch.empty(num_examples, dim)
        random_vectors[0] = torch.randn(dim)
        for i in range(num_examples - 1):
            prev_vector = random_vectors[i]
            noise = torch.randn(dim)
            random_vectors[i + 1] = (
                correlation * prev_vector + (1 - correlation**2) ** 0.5 * noise
            )
    norms = torch.linalg.norm(random_vectors, ord=2, dim=-1, keepdim=True)

    # Normalize the vectors by dividing by their norm.
    # A small epsilon (1e-8) is added for numerical stability
    unit_vectors = random_vectors / (norms + 1e-8)

    return unit_vectors
