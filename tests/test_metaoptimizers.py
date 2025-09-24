import torch
from typing import Any, Dict, Tuple
from src.meta_optimizers import MetaOptimizer, ManualAdamW


def test_manual_adamw():
    """Simple unit test to verify ManualAdamW matches torch.optim.AdamW."""
    import torch.optim as optim
    
    # Setup
    params = {'w': torch.tensor([1.0, 2.0], requires_grad=True)}
    grads = {'w': torch.tensor([0.1, 0.2])}
    hyperparams = {
        'lr': 0.01,
        'beta1': 0.9,
        'beta2': 0.999,
        'epsilon': 1e-8,
        'weight_decay': 0.1
    }
    
    # Manual
    manual_opt = ManualAdamW()
    states = manual_opt.init_states(params)
    new_params_m, new_states_m = manual_opt.step(params, grads, states, **hyperparams)
    
    # Torch
    torch_params = list(params.values())
    torch_opt = optim.AdamW(torch_params, lr=hyperparams['lr'], betas=(hyperparams['beta1'], hyperparams['beta2']),
                            eps=hyperparams['epsilon'], weight_decay=hyperparams['weight_decay'])
    torch_opt.zero_grad()
    torch_params[0].grad = grads['w']
    torch_opt.step()
    
    # Compare (with tolerance for floating point)
    torch.manual_seed(0)  # For reproducibility
    assert torch.allclose(new_params_m['w'], torch_params[0], atol=1e-6)
    print("ManualAdamW matches torch.optim.AdamW!")

if __name__ == "__main__":
    test_manual_adamw()


def test_differentiability():
    """Verify that gradients flow through multiple ManualAdamW steps."""
    params = {'w': torch.tensor([1.0, 2.0], requires_grad=True)}
    hyperparams = {'lr': 0.01, 'beta1': 0.9, 'beta2': 0.999, 'epsilon': 1e-8, 'weight_decay': 0.0}
    
    manual_opt = ManualAdamW()
    states = manual_opt.init_states(params)
    
    # Simulate inner loop: multiple steps with synthetic grads
    for _ in range(3):
        # Synthetic grad (e.g., from a loss)
        grads = {'w': torch.tensor([0.1, 0.2], requires_grad=True)}
        params, states = manual_opt.step(params, grads, states, **hyperparams)
    
    # Compute a loss on final params to check if grad_fn exists
    loss = params['w'].sum()
    loss.backward()
    
    # Check if gradients are populated and grad_fn exists
    assert params['w'].grad is not None
    assert loss.grad_fn is not None
    print("Differentiability verified: Gradients flow through ManualAdamW steps!")

if __name__ == "__main__":
    test_manual_adamw()
    test_differentiability()
