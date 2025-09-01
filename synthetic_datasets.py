import torch
from torch.utils.data import Dataset
from typing import Tuple, Callable


class InContextRecallDataset(Dataset):
    def __init__(self,
        seq_len: int,
        dim: int,
        context_size: int,
        input_corr: float=0.0,
        output_corr: float=0.0
    )-> None:
        self.seq_len = seq_len
        self.dim = dim
        self.context_size = context_size
        self.input_corr = input_corr
        self.output_corr = output_corr        
        self.inputs = generate_vectors(seq_len, dim, input_corr)
        self.targets = generate_vectors(seq_len, dim, output_corr)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        # Case 1: The index is a single integer
        if isinstance(idx, int):
            start_idx = idx - self.context_size + 1
            if start_idx < 0:
                padding_len = abs(start_idx)
                padding = torch.zeros(padding_len, self.dim)
                input_window = torch.cat((padding, self.inputs[:idx+1]), dim=0)
                target_window = torch.cat((padding, self.targets[:idx+1]), dim=0)
            else:
                input_window = self.inputs[start_idx:idx+1]
                target_window = self.targets[start_idx:idx+1]
            return input_window, target_window
        
        # Case 2: The index is a slice
        elif isinstance(idx, slice):
            indices = range(*idx.indices(len(self)))
            
            input_batch = torch.stack([self[i][0] for i in indices])
            target_batch = torch.stack([self[i][1] for i in indices])
            
            return input_batch, target_batch
        
        else:
            raise TypeError("Invalid argument type.")
    
    def __len__(self) -> int:
        return self.seq_len


class SequenceLearningDataset(Dataset):
    def __init__(self,
        num_examples: int,
        dim: int,
        seq_func: Callable[[torch.Tensor], torch.Tensor],
        input_corr: float=0.0,
    )-> None:
        self.num_examples = num_examples
        self.dim = dim
        self.input_corr = input_corr      
        self.inputs = generate_vectors(num_examples, dim, input_corr)
        self.targets = seq_func(self.inputs)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, torch.Tensor]:
        assert index < self.num_examples, "Index out of range"
        return self.inputs[index], self.targets[index]
    
    def __len__(self) -> int:
        return self.num_examples
    

def generate_vectors(
        num_examples: int,
        dim: int,
        correlation: float = 0.0,
)-> torch.Tensor:
    """
    Generates (num_examples) unit norm vectors of length (dim).

    Args:
        num_examples: The number of vector pairs to generate (number of rows).
        dim: The dimension of each individual vector x_i
        correlation: The expected correlation between vector x_i and x_i+1.
    Returns:
        A tensor of shape (num_examples, dim).
    """
    assert 0.0 <= correlation <= 1.0, "Correlation must be between 0.0 and 1.0"

    if correlation == 0.0:
        random_vectors = torch.randn(num_examples, dim)
    else:
        random_vectors = torch.empty(num_examples, dim)
        random_vectors[0] = torch.randn(dim)
        for i in range(num_examples - 1):
            prev_vector = random_vectors[i]
            noise = torch.randn(dim)
            random_vectors[i+1] = (
                correlation * prev_vector + 
                (1 - correlation**2)**0.5 * noise
            )
    norms = torch.linalg.norm(random_vectors, ord=2, dim=-1, keepdim=True)

    # Normalize the vectors by dividing by their norm.
    # A small epsilon (1e-8) is added for numerical stability
    unit_vectors = random_vectors / (norms + 1e-8)

    return unit_vectors
