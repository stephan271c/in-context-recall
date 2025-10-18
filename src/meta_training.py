"""Utilities for meta-training and evaluating differentiable recall models."""

from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Type, Union

import torch
from torch import nn
from torch.func import functional_call

from evaluate_functions import average_accuracy_across_sequences, compute_recall_accuracies
from func_memory_module import HyperparamModel, TTTMLP, WeightModel
from losses import windowed_p_loss, windowed_recall_cross_entropy
from meta_optimizers import ManualAdam, ManualAdamW, ManualSGD, MetaOptimizer
from synthetic_datasets import InContextRecallDataset


__all__ = [
    "MetaTrainingConfig",
    "MemoryModuleFactory",
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
    hyper_lr_initial_bias: float = -2.0
    log_every_sequences: int = 50
    train_memory_module: bool = True
    train_weight_model: bool = True
    train_lr_model: bool = True
    inner_optimizer_name: str = "manual_adam"
    inner_optimizer_kwargs: Dict[str, Any] = field(
        default_factory=lambda: {"beta1": 0.95, "beta2": 0.99, "epsilon": 1e-8}
    )
    outer_optimizer_name: str = "adamw"
    outer_optimizer_kwargs: Dict[str, Any] = field(default_factory=lambda: {"lr": 0.01})

    def __post_init__(self) -> None:
        if self.num_sequences % self.batch_size:
            raise ValueError("batch_size must divide num_sequences for full epochs")

    @property
    def total_meta_updates(self) -> int:
        return self.num_sequences // self.batch_size


MemoryModuleFactory = Callable[[MetaTrainingConfig], nn.Module]


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

    memory_module: nn.Module
    weight_model: nn.Module
    lr_model: nn.Module
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


def _default_memory_module_factory(config: MetaTrainingConfig) -> nn.Module:
    """Instantiate the default memory module when no custom factory is provided."""
    return TTTMLP(config.key_dim, config.val_dim)


def resolve_device(device_preference: str) -> torch.device:
    """Pick an appropriate torch.device for the current machine."""
    if device_preference == "cuda" and not torch.cuda.is_available():
        return torch.device("cpu")
    return torch.device(device_preference)


INNER_OPTIMIZER_REGISTRY: Dict[str, Type[MetaOptimizer]] = {
    "manual_adam": ManualAdam,
    "manual_adamw": ManualAdamW,
    "manual_sgd": ManualSGD,
}

OUTER_OPTIMIZER_REGISTRY: Dict[str, Type[torch.optim.Optimizer]] = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}


def _build_inner_optimizer(name: str) -> MetaOptimizer:
    """Return an instantiated meta-optimizer from the configured registry."""
    key = name.lower()
    try:
        optimizer_cls = INNER_OPTIMIZER_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(INNER_OPTIMIZER_REGISTRY))
        raise ValueError(
            f"Unknown inner optimizer '{name}'. Available options: {available}"
        ) from exc
    return optimizer_cls()


def _get_outer_optimizer_class(name: str) -> Type[torch.optim.Optimizer]:
    """Resolve the torch optimizer class to use for the outer loop."""
    key = name.lower()
    try:
        return OUTER_OPTIMIZER_REGISTRY[key]
    except KeyError as exc:
        available = ", ".join(sorted(OUTER_OPTIMIZER_REGISTRY))
        raise ValueError(
            f"Unknown outer optimizer '{name}'. Available options: {available}"
        ) from exc


class ConstantOutputModule(nn.Module):
    """Wrap a tensor so it can be reused as an nn.Module output."""

    def __init__(self, value: Union[torch.Tensor, float]):
        super().__init__()
        tensor_value = value if isinstance(value, torch.Tensor) else torch.tensor(value)
        self.register_buffer("value", tensor_value.clone().detach())

    def forward(self, *args: Any, **kwargs: Any) -> torch.Tensor:
        return self.value


def build_meta_models(
    config: MetaTrainingConfig,
    device: torch.device,
    *,
    memory_module_factory: Optional[MemoryModuleFactory] = None,
    weight_model: Optional[Union[nn.Module, torch.Tensor]] = None,
    lr_model: Optional[Union[nn.Module, torch.Tensor, float]] = None,
) -> Tuple[nn.Module, nn.Module, nn.Module]:
    """Instantiate the meta-learner components on the requested device.

    Args:
        config: Meta-training hyperparameters controlling model dimensions.
        device: Device all returned modules should live on.
        memory_module_factory: Optional callable that constructs the memory module.
            Defaults to :class:`TTTMLP` when omitted.
        weight_model: Optional override for the weight model. When a tensor is
            provided it is wrapped so the same weights are returned for every
            forward pass, keeping them fixed throughout training.
        lr_model: Optional override for the learning-rate model. Tensors or floats
            are treated as constant outputs.

    Returns:
        Tuple containing the weight-producing module, the memory module, and the
        hyper-parameter module.
    """
    if weight_model is None:
        weight_model_module: nn.Module = WeightModel(config.key_dim, config.context_dim)
    elif isinstance(weight_model, torch.Tensor):
        weight_model_module = ConstantOutputModule(weight_model)
    elif isinstance(weight_model, nn.Module):
        weight_model_module = weight_model
    else:
        raise TypeError(
            "weight_model must be an nn.Module or tensor when provided; "
            f"received type {type(weight_model).__name__}"
        )
    weight_model_module = weight_model_module.to(device)

    factory = memory_module_factory or _default_memory_module_factory
    memory_module = factory(config).to(device)
    if lr_model is None:
        lr_model_module: nn.Module = HyperparamModel(
            config.key_dim, initial_bias=config.hyper_lr_initial_bias
        )
    elif isinstance(lr_model, (torch.Tensor, float)):
        lr_model_module = ConstantOutputModule(lr_model)
    elif isinstance(lr_model, nn.Module):
        lr_model_module = lr_model
    else:
        raise TypeError(
            "lr_model must be an nn.Module, tensor, or float when provided; "
            f"received type {type(lr_model).__name__}"
        )
    lr_model_module = lr_model_module.to(device)

    return weight_model_module, memory_module, lr_model_module


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
    model: nn.Module,
    batch_size: int,
    device: torch.device,
    inner_optimizer: MetaOptimizer,
) -> Tuple[nn.Module, List[Dict[str, torch.Tensor]], List[Dict[str, Any]]]:
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
    memory_module_factory: Optional[MemoryModuleFactory] = None,
    weight_model_override: Optional[Union[nn.Module, torch.Tensor]] = None,
    lr_model_override: Optional[Union[nn.Module, torch.Tensor, float]] = None,
    inner_loss_fn: Callable[[torch.Tensor, torch.Tensor, Optional[torch.Tensor]], torch.Tensor] = windowed_p_loss,
    outer_loss_fn: Optional[nn.Module] = None,
    log_callback: Optional[Callable[[int, int, float, float], None]] = None,
) -> MetaTrainingArtifacts:
    """Execute the meta-learning outer loop and return the trained modules.

    Args:
        config: Hyperparameters controlling data generation, optimisation, and
            the choice/configuration of inner and outer optimizers.
        memory_module_factory: Optional callable that returns a freshly
            initialised memory module when provided. Defaults to :class:`TTTMLP`.
        weight_model_override: Optional module or weight tensor to use instead of
            instantiating a new :class:`WeightModel`. Passing a tensor keeps the
            loss weights fixed during training.
        lr_model_override: Optional module, tensor, or scalar that replaces the
            default :class:`HyperparamModel`. Tensors and scalars act as constant
            outputs.
        inner_loss_fn: Differentiable loss used for the inner updates.
        outer_loss_fn: Loss used when scoring recall performance. A new
            ``nn.CrossEntropyLoss`` instance is created when this is ``None``.
        log_callback: Optional callable taking ``(meta_step, processed_sequences,
            avg_outer_loss, sample_lr)`` to report progress.
    """
    device = resolve_device(config.device_preference)
    weight_model, memory_module, lr_model = build_meta_models(
        config,
        device,
        memory_module_factory=memory_module_factory,
        weight_model=weight_model_override,
        lr_model=lr_model_override,
    )

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
        outer_optimizer_cls = _get_outer_optimizer_class(config.outer_optimizer_name)
        outer_optimizer_kwargs = dict(config.outer_optimizer_kwargs)
        outer_optimizer = outer_optimizer_cls(trainable_parameters, **outer_optimizer_kwargs)
    else:
        outer_optimizer = None

    inner_optimizer = _build_inner_optimizer(config.inner_optimizer_name)
    base_inner_hparams = dict(config.inner_optimizer_kwargs)
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
                hyperparams = dict(base_inner_hparams)
                hyperparams["lr"] = lr_model(current_key[-1])

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
    memory_module: nn.Module,
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

    key_dim = config.key_dim if config.key_dim is not None else getattr(memory_module, "input_dim", None)
    if key_dim is None:
        raise ValueError(
            "EvaluationConfig.key_dim must be provided when the memory_module "
            "does not define an 'input_dim' attribute."
        )
    val_dim = config.val_dim if config.val_dim is not None else getattr(memory_module, "output_dim", None)
    if val_dim is None:
        raise ValueError(
            "EvaluationConfig.val_dim must be provided when the memory_module "
            "does not define an 'output_dim' attribute."
        )
    key_dim = int(key_dim)
    val_dim = int(val_dim)
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
