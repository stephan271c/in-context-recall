import torch
import torch.nn as nn

from src.losses import windowed_recall_cross_entropy
from src.meta_optimizers import MetaAdamW


def test_manual_adamw():
    """Simple unit test to verify ManualAdamW matches torch.optim.AdamW."""
    import torch.optim as optim

    # Setup
    params = {
        "w": torch.tensor([1.0, 2.0], requires_grad=True),
        "b": torch.tensor([3.0, 4.0], requires_grad=True),
    }
    grads = {"w": torch.tensor([0.1, 0.2])}  # no grad for b
    hyperparams = {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.1,
    }

    # Manual
    manual_opt = MetaAdamW()
    states = manual_opt.init_states(params)
    new_params_m, new_states_m = manual_opt.step(params, grads, states, **hyperparams)

    # Torch
    torch_params = list(params.values())
    torch_opt = optim.AdamW(
        torch_params,
        lr=hyperparams["lr"],
        betas=(hyperparams["beta1"], hyperparams["beta2"]),
        eps=hyperparams["epsilon"],
        weight_decay=hyperparams["weight_decay"],
    )
    torch_opt.zero_grad()
    torch_params[0].grad = grads["w"]
    torch_opt.step()

    assert torch.allclose(new_params_m["w"], torch_params[0], atol=1e-6)
    assert torch.allclose(new_params_m["b"], torch_params[1], atol=1e-6)


def test_differentiability():
    """Verify that gradients flow through multiple ManualAdamW steps."""
    params = {"w": torch.tensor([1.0, 2.0], requires_grad=True)}
    hyperparams = {
        "lr": 0.01,
        "beta1": 0.9,
        "beta2": 0.999,
        "epsilon": 1e-8,
        "weight_decay": 0.0,
    }

    manual_opt = MetaAdamW()
    states = manual_opt.init_states(params)
    grads_history = []

    # Simulate inner loop: multiple steps with synthetic grads
    for _ in range(3):
        grads = {"w": torch.tensor([0.1, 0.2], requires_grad=True)}
        grads_history.append(grads["w"])
        params, states = manual_opt.step(params, grads, states, **hyperparams)

    params["w"].retain_grad()
    loss = params["w"].sum()
    loss.backward()

    # Check if gradients are populated and grad_fn exists
    assert params["w"].grad is not None
    assert loss.grad_fn is not None
    assert all(g.grad is not None for g in grads_history)


def test_windowed_recall_cross_entropy_matches_manual():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    params = {
        name: p.clone().detach().requires_grad_(True)
        for name, p in model.named_parameters()
    }
    keys = torch.eye(2)
    values = keys.clone()
    time_index = 1
    window_size = 2
    offset = 0

    loss = windowed_recall_cross_entropy(
        model,
        params,
        keys,
        values,
        time_index=time_index,
        window_size=window_size,
        offset=offset,
    )

    window_end = time_index - offset
    start_index = max(0, window_end - window_size)
    manual_logits = torch.matmul(keys[start_index:window_end], values.T)
    targets = torch.arange(start_index, window_end)
    expected_loss = nn.functional.cross_entropy(manual_logits, targets)

    assert torch.allclose(loss, expected_loss)

    loss.backward()
    assert all(param.grad is not None for param in params.values())


def test_windowed_recall_cross_entropy_respects_offset():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    params = {
        name: p.clone().detach().requires_grad_(True)
        for name, p in model.named_parameters()
    }
    keys = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, -1.0]]
    )
    values = keys.clone()

    baseline_loss = windowed_recall_cross_entropy(
        model,
        params,
        keys,
        values,
        time_index=3,
        window_size=2,
        offset=0,
    )

    offset_loss = windowed_recall_cross_entropy(
        model,
        params,
        keys,
        values,
        time_index=3,
        window_size=2,
        offset=1,
    )

    window_end = 3 - 1
    start_index = max(0, window_end - 2)
    manual_logits = torch.matmul(keys[start_index:window_end], values.T)
    targets = torch.arange(start_index, window_end)
    expected_offset_loss = nn.functional.cross_entropy(manual_logits, targets)

    assert torch.allclose(offset_loss, expected_offset_loss)
    assert not torch.allclose(baseline_loss, offset_loss)


def test_windowed_recall_cross_entropy_returns_zero_if_offset_exceeds_time():
    model = nn.Linear(2, 2, bias=False)
    with torch.no_grad():
        model.weight.copy_(torch.eye(2))

    params = {
        name: p.clone().detach().requires_grad_(True)
        for name, p in model.named_parameters()
    }
    keys = torch.eye(2)
    values = keys.clone()

    loss = windowed_recall_cross_entropy(
        model,
        params,
        keys,
        values,
        time_index=0,
        window_size=2,
        offset=5,
    )

    assert loss.item() == 0.0
