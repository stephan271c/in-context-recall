import torch
import torch.nn as nn
from collections import OrderedDict

# 1. Define the model architecture
class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer1 = nn.Linear(in_features=10, out_features=20)
        self.relu = nn.ReLU()
        self.layer2 = nn.Linear(in_features=20, out_features=5)

    def forward(self, x):
        return self.layer2(self.relu(self.layer1(x)))

# Instantiate the model
model = SimpleMLP()

# Let's see the names and shapes of the default parameters
print("Original model state_dict keys and parameter shapes:")
for name, param in model.named_parameters():
    print(f"{name}: {param.shape}")

# 2. Suppose you have a list of tensors with matching shapes
# The order MUST correspond to the model's parameter order.
# In this case: layer1.weight, layer1.bias, layer2.weight, layer2.bias
weights_list = [
    torch.randn(20, 10), # Shape for layer1.weight
    torch.randn(20),     # Shape for layer1.bias
    torch.randn(5, 20),  # Shape for layer2.weight
    torch.randn(5)       # Shape for layer2.bias
]

# 3. Construct the new state_dict
new_state_dict = OrderedDict()

# Get the keys from the model's state_dict
# We use named_parameters() to ensure we only get learnable parameters
# and maintain the correct order.
param_keys = [name for name, _ in model.named_parameters()]

# Check if the number of tensors matches the number of parameters
if len(param_keys) != len(weights_list):
    raise ValueError(
        f"Mismatch: model has {len(param_keys)} parameters, "
        f"but {len(weights_list)} tensors were provided."
    )

for key, tensor in zip(param_keys, weights_list):
    new_state_dict[key] = tensor

# 4. Load the new state_dict into the model
model.load_state_dict(new_state_dict)

print("\nSuccessfully loaded weights from the list!")

# --- Verification ---
# Check if the first layer's weight tensor was updated correctly
print("\nVerification:")
# torch.equal checks for both shape and value equality
are_equal = torch.equal(model.layer1.weight, weights_list[0])
print(f"Is model.layer1.weight equal to the first tensor in the list? {are_equal}")