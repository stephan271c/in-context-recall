import torch
import torch.nn as nn

from src.forward_loop import inner_optimization_forward
from src.meta_optimizers import MetaSGD
from src.model_components import HyperparamHeadWrapper, TTT
from src.synthetic_datasets import InContextRecallDataset


def test_inner_optimization_forward_runs_and_backprops():
    torch.manual_seed(0)

    model = TTT(input_dim=2, output_dim=2, num_layers=1)
    dataset = InContextRecallDataset(
        seq_len=3, key_dim=2, val_dim=2, context_size=2, batch_size=2
    )
    optimizer = MetaSGD()

    inner_lr_head = 0.1  # float path through HyperparamHeadWrapper
    inner_loss_weight_head = torch.ones(dataset.context_size)  # tensor path

    outer_loss, preds = inner_optimization_forward(
        model,
        dataset,
        optimizer,
        inner_lr_head,
        inner_loss_weight_head,
        inner_optimizer_kwargs={"beta": 0.0},
        eval_mode=True,
    )

    assert outer_loss.shape == torch.Size([])
    assert len(preds) == dataset.seq_len
    assert preds[0].shape == (dataset.batch_size, 1, model.output_dim)
    assert preds[-1].shape == (dataset.batch_size, dataset.seq_len, model.output_dim)

    outer_loss.backward()
    grads = [p.grad for _, p in model.named_parameters()]
    assert all(g is not None for g in grads)
    assert any(torch.linalg.norm(g) > 0 for g in grads)


def test_hyperparam_head_wrapper_shapes():
    device = torch.device("cpu")
    batch_size = 4
    keys = torch.randn(batch_size, 3, device=device)

    # float should expand to (B,)
    lr = HyperparamHeadWrapper(0.5, device)(keys, batch_size)
    assert lr.shape == (batch_size,)

    # 1D tensor expands to (B, feature_dim)
    weights = HyperparamHeadWrapper(torch.tensor([1.0, 2.0]), device)(
        keys, batch_size
    )
    assert weights.shape == (batch_size, 2)

    # module returning scalar broadcasts to (B,)
    class ScalarHead(nn.Module):
        def forward(self):
            return torch.tensor(0.3)

    scalar = HyperparamHeadWrapper(ScalarHead(), device)(keys, batch_size)
    assert scalar.shape == (batch_size,)

    # module consuming keys keeps batch dim
    class KeyHead(nn.Module):
        def forward(self, k):
            return k.sum(dim=-1)

    keyed = HyperparamHeadWrapper(KeyHead(), device)(keys, batch_size)
    assert keyed.shape == (batch_size,)
