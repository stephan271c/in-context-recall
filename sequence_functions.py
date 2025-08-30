import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import activations

class CausalSelfAttention(nn.Module):
    """
    An efficient single-head causal self-attention layer with a sliding window.
    A token at a given position can only attend to itself and a fixed number
    of previous tokens defined by the `window_size`. This implementation avoids
    the O(n^2) complexity of full self-attention by only computing scores
    within the sliding window, resulting in O(n * window_size) complexity.

    If `window_size` is set to 0 or a negative number, it reverts to standard
    (full) causal self-attention.
    """
    def __init__(self, d_model: int, window_size: int):
        """
        Initializes the self-attention layer.

        Args:
            d_model (int): The dimension of the model (and input/output features).
                           This is 'd' in the (n, d) input shape.
            window_size (int): The size of the sliding attention window.
                               Each token can attend to itself and the previous
                               `window_size - 1` tokens. If set to 0 or less,
                               vanilla causal attention is used.
        """
        super().__init__()
        self.d_model = d_model
        self.window_size = window_size

        # Define the linear projections for Query, Key, and Value
        self.query_proj = nn.Linear(d_model, d_model)
        self.key_proj = nn.Linear(d_model, d_model)
        self.value_proj = nn.Linear(d_model, d_model)

        # Softmax layer to compute attention weights
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Performs the forward pass of the causal self-attention layer.

        Args:
            x (torch.Tensor): The input tensor of shape (n, d_model), where 'n' is
                              the sequence length (number of examples/tokens).

        Returns:
            torch.Tensor: The output tensor of the same shape (n, d_model).
        """
        n, d = x.shape
        if d != self.d_model:
            raise ValueError(f"Input dimension {d} does not match model dimension {self.d_model}")

        # 1. Project the input into Query, Key, and Value tensors
        Q = self.query_proj(x)
        K = self.key_proj(x)
        V = self.value_proj(x)

        # If window_size is non-positive, perform full (vanilla) causal attention
        if self.window_size <= 0:
            # Calculate full attention scores (n, n)
            scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_model)
            
            # Apply the standard causal mask
            mask = torch.tril(torch.ones(n, n, device=x.device))
            scores = scores.masked_fill(mask == 0, float('-inf'))
            
            # Compute attention weights and output
            attention_weights = self.softmax(scores)
            output = torch.matmul(attention_weights, V)
            return output

        # --- Efficient Sliding Window Attention Logic ---

        # 2. Create sliding window views of K and V for efficient computation.
        pad_size = self.window_size - 1
        padded_K = F.pad(K, (0, 0, pad_size, 0)).contiguous()
        padded_V = F.pad(V, (0, 0, pad_size, 0)).contiguous()

        # `as_strided` creates a view into the tensor without copying data.
        k_stride_0, k_stride_1 = padded_K.stride()
        K_windows = padded_K.as_strided(size=(n, self.window_size, d), stride=(k_stride_0, k_stride_0, k_stride_1))
        
        v_stride_0, v_stride_1 = padded_V.stride()
        V_windows = padded_V.as_strided(size=(n, self.window_size, d), stride=(v_stride_0, v_stride_0, v_stride_1))

        # 3. Compute attention scores using batched matrix multiplication.
        scores = torch.bmm(Q.unsqueeze(1), K_windows.transpose(1, 2))
        scores = scores.squeeze(1) # -> (n, window_size)

        # 4. Scale the scores
        scores = scores / math.sqrt(self.d_model)
        
        # 5. Apply causal mask to the windowed scores.
        num_valid_keys = torch.minimum(torch.arange(n, device=x.device) + 1, torch.tensor(self.window_size, device=x.device))
        first_valid_idx = self.window_size - num_valid_keys
        
        mask_indices = torch.arange(self.window_size, device=x.device).unsqueeze(0)
        causal_mask = mask_indices < first_valid_idx.unsqueeze(1)
        
        scores.masked_fill_(causal_mask, float('-inf'))

        # 6. Compute attention weights using softmax
        attention_weights = self.softmax(scores) # Shape: (n, window_size)

        # 7. Compute the final output by weighting the Value windows.
        output = torch.bmm(attention_weights.unsqueeze(1), V_windows)
        output = output.squeeze(1) # -> (n, d)

        return output

class MLP(nn.Module):
    def __init__(self, d_model: int, num_layers: int, activ_str: str='relu'):
        super().__init__()
        self.d_model = d_model
        self.num_layers = num_layers
        self.layers = nn.ModuleList([nn.Linear(d_model, d_model) for _ in range(num_layers)])
        self.activation = activations.get_activation(activ_str)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for layer in self.layers:
            x = self.activation(layer(x))
        return x
