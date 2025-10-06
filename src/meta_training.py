"""Utilities for meta-training and evaluating differentiable recall models."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import torch
from torch import nn
from torch.func import functional_call

from evaluate_functions import average_accuracy_across_sequences, compute_recall_accuracies
from func_memory_module import HyperparamModel, TTTMLP, WeightModel
from losses import windowed_p_loss, windowed_recall_cross_entropy
from meta_optimizers import ManualAdam
from synthetic_datasets import InContextRecallDataset


__all__ = [
    "MetaTrainingConfig",
    "EvaluationConfig",
    "MetaBatchItem",
    "MetaTrainingArtifacts",
    "EvaluationResult",
    "resolve_device",
    "build_meta_models",
    "sample_meta_batch",
    "run_meta_training",
    "evaluate_memory_module",
]


@dataclass
class MetaTrainingConfig:
    """Configuration options that control the outer-loop optimisation."""

    key_dim: int = 16
    val_dim: int = 16
    context_dim: int = 5
    seq_len: int = 50
    num_sequences: int = 500
    batch_size: int = 10
    recall_window: int = 1
    output_corr: float = 0.5
    device_preference: str = "cuda"
    outer_lr: float = 0.01
    hyper_lr_initial_bias: float = -2.0
    beta1: float = 0.95
    beta2: float = 0.99
    log_every_sequences: int = 50
    train_memory_module: bool = True
    train_weight_model: bool = True
    train_lr_model: bool = True

    def __post_init__(self) -> None:
        if self.num_sequences % self.batch_size:
            raise ValueError("batch_size must divide num_sequences for full epochs")

    @property
    def total_meta_updates(self) -> int:
        return self.num_sequences // self.batch_size


@dataclass
class EvaluationConfig:
    """Settings used when evaluating trained memory modules."""

    seq_len: int
    num_sequences: int = 20
    key_dim: Optional[int] = None
    val_dim: Optional[int] = None
    context_dim: Optional[int] = None
    output_corr: float = 0.5


@dataclass
class MetaBatchItem:
    """In-memory representation of a sampled sequence for the inner loop."""

    dataset: InContextRecallDataset
    keys: torch.Tensor
    values: torch.Tensor


@dataclass
class MetaTrainingArtifacts:
    """Trained models alongside simple training history."""

    memory_module: TTTMLP
    weight_model: WeightModel
    lr_model: HyperparamModel
    outer_losses: List[float]


@dataclass
class EvaluationResult:
    """Aggregated recall accuracy statistics across evaluation sequences."""

    offsets: torch.Tensor
    mean_accuracy: torch.Tensor
    counts: torch.Tensor

    def cpu(self) -> "EvaluationResult":
        """Return a copy of the result with tensors moved to the CPU."""
        return EvaluationResult(
            offsets=self.offsets.cpu(),
            mean_accuracy=self.mean_accuracy.cpu(),
            counts=self.counts.cpu(),
        )


def resolve_device(device_preference: str) -> torch.device:
    """Pick an appropriate torch.device for the current machine."""
    if device_preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_preference)


def build_meta_models(
    config: MetaTrainingConfig, device: torch.device
) -> Tuple[WeightModel, TTTMLP, HyperparamModel]:
    """Instantiate the meta-learner components on the requested device."""
    weight_model = WeightModel(config.key_dim, config.context_dim).to(device)
    memory_module = TTTMLP(config.key_dim, config.val_dim).to(device)
    lr_model = HyperparamModel(config.key_dim, initial_bias=config.hyper_lr_initial_bias).to(device)
    return weight_model, memory_module, lr_model


def sample_meta_batch(config: MetaTrainingConfig, device: torch.device) -> List[MetaBatchItem]:
    """Generate a fresh batch of synthetic recall tasks."""
    batch: List[MetaBatchItem] = []
    for _ in range(config.batch_size):
        dataset = InContextRecallDataset(
            seq_len=config.seq_len,
            key_dim=config.key_dim,
            val_dim=config.val_dim,
            context_size=config.context_dim,
            output_corr=config.output_corr,
        )
        batch.append(
            MetaBatchItem(
                dataset=dataset,
                keys=dataset.inputs.to(device),
                values=dataset.targets.to(device),
            )
        )
    return batch


def _initialise_inner_state(
    model: TTTMLP,
    batch_size: int,
    device: torch.device,
    inner_optimizer: ManualAdam,
) -> Tuple[TTTMLP, List[Dict[str, torch.Tensor]], List[Dict[str, Any]]]:
    """Create a functional copy of ``model`` and optimizer states per task."""
    fast_model = copy.deepcopy(model).to(device)
    param_sets: List[Dict[str, torch.Tensor]] = []
    state_list: List[Dict[str, Any]] = []
    for _ in range(batch_size):
        params: Dict[str, torch.Tensor] = {}
        for name, param in fast_model.named_parameters():
            cloned = param.detach().clone().requires_grad_(True)
            params[name] = cloned
        param_sets.append(params)
        state_list.append(inner_optimizer.init_states(params))
    return fast_model, param_sets, state_list


def run_meta_training(
    config: MetaTrainingConfig,
    *,
    inner_loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = windowed_p_loss,
    outer_loss_fn: Optional[nn.Module] = None,
    log_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> MetaTrainingArtifacts:
    """Execute the meta-learning outer loop and return the trained modules.

    Args:
        config: Hyperparameters controlling data generation and optimisation.
        inner_loss_fn: Differentiable loss used for the inner updates.
        outer_loss_fn: Loss used when scoring recall performance. A new
            ``nn.CrossEntropyLoss`` instance is created when this is ``None``.
        log_callback: Optional callable taking ``(meta_step, processed_sequences,
            avg_outer_loss, sample_lr)`` to report progress.
    """
    device = resolve_device(config.device_preference)
    weight_model, memory_module, lr_model = build_meta_models(config, device)

    outer_loss_fn = outer_loss_fn or nn.CrossEntropyLoss()

    memory_params = list(memory_module.parameters())
    weight_params = list(weight_model.parameters())
    lr_params = list(lr_model.parameters())

    for param in memory_params:
        param.requires_grad_(config.train_memory_module)
    for param in weight_params:
        param.requires_grad_(config.train_weight_model)
    for param in lr_params:
        param.requires_grad_(config.train_lr_model)

    trainable_parameters: List[nn.Parameter] = []
    if config.train_memory_module:
        trainable_parameters.extend(memory_params)
    if config.train_weight_model:
        trainable_parameters.extend(weight_params)
    if config.train_lr_model:
        trainable_parameters.extend(lr_params)

    outer_optimizer: Optional[torch.optim.Optimizer]
    if trainable_parameters:
        outer_optimizer = torch.optim.AdamW(trainable_parameters, lr=config.outer_lr)
    else:
        outer_optimizer = None

    inner_optimizer = ManualAdam()
    history: List[float] = []

    for meta_step in range(config.total_meta_updates):
        batch = sample_meta_batch(config, device)
        fast_model, param_sets, state_list = _initialise_inner_state(
            memory_module, config.batch_size, device, inner_optimizer
        )

        if outer_optimizer is not None:
            outer_optimizer.zero_grad()
        total_outer_loss = torch.zeros((), device=device)

        for time_index in range(config.seq_len):
            for task_idx, item in enumerate(batch):
                current_key, current_val = item.dataset[time_index]
                current_key = current_key.to(device)
                current_val = current_val.to(device)

                loss_weights = weight_model(current_key[-1])
                hyperparams = {
                    "lr": lr_model(current_key[-1]),
                    "beta1": config.beta1,
                    "beta2": config.beta2,
                }

                params = param_sets[task_idx]
                state = state_list[task_idx]

                predictions = functional_call(fast_model, params, current_key)
                inner_loss = inner_loss_fn(predictions.T, current_val.T, loss_weights)
                grad_tuple = torch.autograd.grad(
                    inner_loss, tuple(params.values()), create_graph=True
                )
                grads = dict(zip(params.keys(), grad_tuple))

                updated_params, updated_state = inner_optimizer.step(
                    params, grads, state, **hyperparams
                )
                param_sets[task_idx] = updated_params
                state_list[task_idx] = updated_state

                outer_loss_step = windowed_recall_cross_entropy(
                    fast_model,
                    updated_params,
                    item.keys,
                    item.values,
                    time_index=time_index,
                    window_size=config.recall_window,
                    loss_fn=outer_loss_fn,
                )
                total_outer_loss = total_outer_loss + outer_loss_step


        avg_outer_loss = total_outer_loss / (config.seq_len * config.batch_size)
        history.append(float(avg_outer_loss.detach()))
        if outer_optimizer is not None:
            avg_outer_loss.backward()
            outer_optimizer.step()

        processed_sequences = (meta_step + 1) * config.batch_size
        should_log = meta_step == 0
        if config.log_every_sequences:
            should_log = should_log or (
                processed_sequences % config.log_every_sequences == 0
            )
        if should_log:
            sample_key = batch[0].keys[0].unsqueeze(0)
            sample_lr = float(lr_model(sample_key).item())
            if log_callback is not None:
                log_callback(meta_step, processed_sequences, float(avg_outer_loss.item()), sample_lr)
            else:
                print(
                    f"Epoch {processed_sequences} | Avg Outer Loss: {avg_outer_loss.item():.4f}"
                )
                print(f"  Sample Hyperparams -> LR: {sample_lr:.4f}")

    return MetaTrainingArtifacts(
        memory_module=memory_module,
        weight_model=weight_model,
        lr_model=lr_model,
        outer_losses=history,
    )


def evaluate_memory_module(
    memory_module: TTTMLP,
    config: EvaluationConfig,
    *,
    device: Optional[torch.device] = None,
) -> EvaluationResult:
    """Evaluate recall accuracy across newly sampled sequences."""
    if device is None:
        try:
            first_param = next(memory_module.parameters())
            device = first_param.device
        except StopIteration:
            device = torch.device("cpu")

    key_dim = config.key_dim or memory_module.input_dim
    val_dim = config.val_dim or memory_module.output_dim
    if config.context_dim is None:
        raise ValueError("EvaluationConfig.context_dim must be specified for evaluation")
    context_dim = config.context_dim

    histories: List[Sequence[torch.Tensor]] = []
    was_training = memory_module.training
    memory_module.eval()
    with torch.no_grad():
        for _ in range(config.num_sequences):
            dataset = InContextRecallDataset(
                seq_len=config.seq_len,
                key_dim=key_dim,
                val_dim=val_dim,
                context_size=context_dim,
                output_corr=config.output_corr,
            )
            keys = dataset.inputs.to(device)
            values = dataset.targets.to(device)
            history = compute_recall_accuracies(memory_module, keys, values)
            histories.append(history)
    if was_training:
        memory_module.train()

    mean_accuracy, counts = average_accuracy_across_sequences(histories)
    offsets = torch.arange(mean_accuracy.shape[0], device=mean_accuracy.device)
    return EvaluationResult(offsets=offsets, mean_accuracy=mean_accuracy, counts=counts)
