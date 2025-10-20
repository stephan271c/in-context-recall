from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch import Tensor
from torch.func import functional_call

from configuration import ExperimentConfig, ModelConfig
from evaluate_functions import (
    average_accuracy_across_sequences,
    average_correct_retrievals_across_sequences,
)
from func_memory_module import TTTMLP
from losses import (
    windowed_inner_product_loss,
    windowed_p_loss,
    windowed_recall_cross_entropy,
)
from meta_optimizers import ManualAdam, ManualAdamW, ManualSGD
from synthetic_datasets import InContextRecallDataset


INNER_LOSS_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "windowed_p_loss": windowed_p_loss,
    "windowed_inner_product_loss": windowed_inner_product_loss,
}

OUTER_LOSS_REGISTRY: Dict[str, Callable[..., Tensor]] = {
    "windowed_recall_cross_entropy": windowed_recall_cross_entropy,
}

OPTIMIZER_REGISTRY: Dict[str, Callable[[], Any]] = {
    "manual_sgd": ManualSGD,
    "manual_adam": ManualAdam,
    "manual_adamw": ManualAdamW,
}


@dataclass
class ModelRunResult:
    model_config: ModelConfig
    mean_accuracy: Tensor
    accuracy_counts: Tensor
    mean_retrievals: Tensor
    retrieval_counts: Tensor
    mean_outer_loss: Optional[Tensor] = None
    outer_loss_counts: Optional[Tensor] = None
    per_sequence_outer_losses: List[Tensor] = field(default_factory=list)

    def offset_zero_accuracy(self) -> Optional[float]:
        if self.mean_accuracy.numel() == 0:
            return None
        return float(self.mean_accuracy[0])


def build_datasets(config: ExperimentConfig) -> List[InContextRecallDataset]:
    """Instantiate the synthetic datasets used across all model runs."""
    dataset_kwargs = dict(config.dataset or {})
    input_corr = float(dataset_kwargs.get("input_corr", 0.0))
    output_corr = float(dataset_kwargs.get("output_corr", 0.0))

    return [
        InContextRecallDataset(
            seq_len=config.sequence_length,
            key_dim=config.key_dim,
            val_dim=config.val_dim,
            context_size=config.context_size,
            input_corr=input_corr,
            output_corr=output_corr,
        )
        for _ in range(config.num_sequences)
    ]


def resolve_device(preferred: Optional[str]) -> torch.device:
    if preferred:
        return torch.device(preferred)
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def run_model(
    experiment: ExperimentConfig,
    model_config: ModelConfig,
    datasets: Sequence[InContextRecallDataset],
    device: torch.device,
) -> ModelRunResult:
    model_type = model_config.model.type
    if model_type == "outer_product":
        return _run_outer_product(experiment, model_config, datasets, device)

    if model_type == "tttmlp" or model_type == "ttt_mlp":
        if model_config.trainer is None:
            raise ValueError(f"Model '{model_config.name}' requires a trainer configuration")
        return _run_ttt_mlp(
            experiment=experiment,
            model_config=model_config,
            datasets=datasets,
            device=device,
        )

    raise ValueError(f"Unsupported model type '{model_type}' for model '{model_config.name}'")


def _run_ttt_mlp(
    experiment: ExperimentConfig,
    model_config: ModelConfig,
    datasets: Sequence[InContextRecallDataset],
    device: torch.device,
) -> ModelRunResult:
    trainer_cfg = model_config.trainer
    if trainer_cfg is None:
        raise ValueError("TTT-MLP models must specify a trainer")

    optimizer_factory = OPTIMIZER_REGISTRY.get(trainer_cfg.type)
    if optimizer_factory is None:
        raise ValueError(
            f"Trainer type '{trainer_cfg.type}' is not supported "
            f"for model '{model_config.name}'"
        )
    optimizer = optimizer_factory()

    inner_loss_fn = _resolve_loss(model_config.inner_loss, INNER_LOSS_REGISTRY)
    outer_loss_fn = (
        _resolve_loss(model_config.outer_loss, OUTER_LOSS_REGISTRY)
        if model_config.outer_loss
        else None
    )

    trainer_params = dict(trainer_cfg.params)
    if trainer_cfg.type == "manual_sgd":
        if "lr" not in trainer_params:
            raise ValueError("ManualSGD trainer requires an 'lr' parameter")
        if "momentum" in trainer_params and "beta" not in trainer_params:
            trainer_params["beta"] = trainer_params.pop("momentum")
        trainer_params.setdefault("beta", 0.0)

    accuracy_histories: List[List[Tensor]] = []
    retrieval_histories: List[Tensor] = []
    outer_loss_histories: List[Tensor] = []

    for dataset in datasets:
        module = _initialize_ttt_module(model_config, experiment, device)
        params = _make_functional_params(module)
        state = _tree_detach(optimizer.init_states(params))

        full_keys = dataset.inputs.to(device)
        full_values = dataset.targets.to(device)

        sequence_accuracy: List[Tensor] = []
        sequence_retrievals: List[Tensor] = []
        sequence_outer_losses: List[Tensor] = []

        for timestep in range(len(dataset)):
            key_window, value_window = dataset[timestep]
            key_window = key_window.to(device)
            value_window = value_window.to(device)

            predictions = functional_call(module, params, key_window)
            loss_kwargs = dict(model_config.inner_loss.params)
            weights = loss_kwargs.pop("weights", None)
            weights_tensor = _maybe_tensor(weights, predictions.device, predictions.dtype)
            loss = inner_loss_fn(
                predictions.T,
                value_window.T,
                weights_tensor,
                **loss_kwargs,
            )

            param_items = list(params.items())
            grads_tuple = torch.autograd.grad(
                loss,
                tuple(p for _, p in param_items),
                allow_unused=True,
            )
            grads = {
                name: torch.zeros_like(param) if grad is None else grad
                for (name, param), grad in zip(param_items, grads_tuple)
            }

            updated_params, updated_state = optimizer.step(
                params,
                grads,
                state,
                **trainer_params,
            )

            params = _detach_params_for_next_step(updated_params)
            state = _tree_detach(updated_state)

            with torch.no_grad():
                window_keys = full_keys[: timestep + 1]
                window_values = full_values[: timestep + 1]
                eval_predictions = functional_call(module, params, window_keys)
                logits = eval_predictions @ window_values.T
                predicted_indices = logits.argmax(dim=-1)
                target_indices = torch.arange(timestep + 1, device=predicted_indices.device)
                per_key_accuracy = (predicted_indices == target_indices).to(torch.float32)
                sequence_accuracy.append(per_key_accuracy.flip(0))
                sequence_retrievals.append(per_key_accuracy.sum())

                if outer_loss_fn is not None:
                    outer_kwargs = dict(model_config.outer_loss.params)
                    loss_value = outer_loss_fn(
                        module,
                        params,
                        full_keys,
                        full_values,
                        time_index=timestep,
                        **outer_kwargs,
                    )
                    sequence_outer_losses.append(loss_value.detach().clone())

        accuracy_histories.append(sequence_accuracy)
        retrieval_histories.append(torch.stack(sequence_retrievals))
        if sequence_outer_losses:
            outer_loss_histories.append(torch.stack(sequence_outer_losses))

    mean_accuracy, accuracy_counts = average_accuracy_across_sequences(accuracy_histories)
    wrapped_retrieval_histories = [[counts] for counts in retrieval_histories]
    mean_retrievals, retrieval_counts = average_correct_retrievals_across_sequences(
        wrapped_retrieval_histories
    )

    mean_outer_loss: Optional[Tensor]
    outer_loss_counts: Optional[Tensor]
    if outer_loss_histories:
        mean_outer_loss, outer_loss_counts = _aggregate_outer_losses(outer_loss_histories)
    else:
        mean_outer_loss = None
        outer_loss_counts = None

    return ModelRunResult(
        model_config=model_config,
        mean_accuracy=mean_accuracy.cpu(),
        accuracy_counts=accuracy_counts.cpu(),
        mean_retrievals=mean_retrievals.cpu(),
        retrieval_counts=retrieval_counts.cpu(),
        mean_outer_loss=mean_outer_loss.cpu() if mean_outer_loss is not None else None,
        outer_loss_counts=outer_loss_counts.cpu() if outer_loss_counts is not None else None,
        per_sequence_outer_losses=[tensor.cpu() for tensor in outer_loss_histories],
    )


def _run_outer_product(
    experiment: ExperimentConfig,
    model_config: ModelConfig,
    datasets: Sequence[InContextRecallDataset],
    device: torch.device,
) -> ModelRunResult:
    accuracy_histories: List[List[Tensor]] = []
    retrieval_histories: List[Tensor] = []

    for dataset in datasets:
        keys = dataset.inputs.to(device)
        values = dataset.targets.to(device)
        cumulative = torch.zeros(experiment.val_dim, experiment.key_dim, device=device)
        sequence_accuracy: List[Tensor] = []
        sequence_retrievals: List[Tensor] = []

        with torch.no_grad():
            for timestep in range(keys.shape[0]):
                key_t = keys[timestep]
                value_t = values[timestep]
                cumulative = cumulative + torch.outer(value_t, key_t)

                window_keys = keys[: timestep + 1]
                window_values = values[: timestep + 1]
                predictions = (cumulative @ window_keys.T).T
                logits = predictions @ window_values.T
                predicted_indices = logits.argmax(dim=-1)
                target_indices = torch.arange(timestep + 1, device=device)
                per_key_accuracy = (predicted_indices == target_indices).to(torch.float32)
                sequence_accuracy.append(per_key_accuracy.flip(0))
                sequence_retrievals.append(per_key_accuracy.sum())

        accuracy_histories.append(sequence_accuracy)
        retrieval_histories.append(torch.stack(sequence_retrievals))

    mean_accuracy, accuracy_counts = average_accuracy_across_sequences(accuracy_histories)
    wrapped_retrieval_histories = [[counts] for counts in retrieval_histories]
    mean_retrievals, retrieval_counts = average_correct_retrievals_across_sequences(
        wrapped_retrieval_histories
    )

    return ModelRunResult(
        model_config=model_config,
        mean_accuracy=mean_accuracy.cpu(),
        accuracy_counts=accuracy_counts.cpu(),
        mean_retrievals=mean_retrievals.cpu(),
        retrieval_counts=retrieval_counts.cpu(),
    )


def _initialize_ttt_module(
    model_config: ModelConfig,
    experiment: ExperimentConfig,
    device: torch.device,
) -> TTTMLP:
    params = dict(model_config.model.params)
    num_layers = int(params.get("num_layers", 1))
    module = TTTMLP(experiment.key_dim, experiment.val_dim, num_layers=num_layers).to(device)

    init_strategy = str(params.get("init", "ones")).lower()
    if init_strategy == "ones":
        fill_value = float(params.get("init_value", 1.0))
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.fill_(fill_value)
    elif init_strategy == "normal":
        mean = float(params.get("mean", 0.0))
        std = float(params.get("std", 0.02))
        with torch.no_grad():
            for parameter in module.parameters():
                parameter.normal_(mean=mean, std=std)
    else:
        raise ValueError(
            f"Unsupported initialization strategy '{init_strategy}' "
            f"for model '{model_config.name}'"
        )

    module.train()
    return module


def _make_functional_params(module: TTTMLP) -> Dict[str, Tensor]:
    return {
        name: parameter.detach().clone().requires_grad_(True)
        for name, parameter in module.named_parameters()
    }


def _detach_params_for_next_step(params: Dict[str, Tensor]) -> Dict[str, Tensor]:
    return {
        name: tensor.detach().clone().requires_grad_(True)
        for name, tensor in params.items()
    }


def _tree_detach(obj: Any) -> Any:
    if isinstance(obj, Tensor):
        return obj.detach().clone()
    if isinstance(obj, dict):
        return {name: _tree_detach(value) for name, value in obj.items()}
    if isinstance(obj, list):
        return [_tree_detach(value) for value in obj]
    if isinstance(obj, tuple):
        return tuple(_tree_detach(value) for value in obj)
    return obj


def _maybe_tensor(
    value: Any,
    device: torch.device,
    dtype: torch.dtype,
) -> Optional[Tensor]:
    if value is None:
        return None
    tensor = torch.as_tensor(value, dtype=dtype, device=device)
    if tensor.ndim == 0:
        tensor = tensor.unsqueeze(0)
    return tensor


def _resolve_loss(
    loss_config: Optional[Any],
    registry: Dict[str, Callable[..., Tensor]],
) -> Callable[..., Tensor]:
    if loss_config is None:
        raise ValueError("Loss configuration must not be None")
    loss_fn = registry.get(loss_config.name)
    if loss_fn is None:
        raise ValueError(f"Loss '{loss_config.name}' is not registered")
    return loss_fn


def _aggregate_outer_losses(
    outer_loss_histories: Sequence[Tensor],
) -> Tuple[Tensor, Tensor]:
    max_len = max(tensor.shape[0] for tensor in outer_loss_histories)
    device = outer_loss_histories[0].device
    totals = torch.zeros(max_len, dtype=torch.float32, device=device)
    counts = torch.zeros(max_len, dtype=torch.int64, device=device)

    for tensor in outer_loss_histories:
        length = tensor.shape[0]
        totals[:length] += tensor.to(device=totals.device, dtype=totals.dtype)
        counts[:length] += 1

    mean = totals / counts.clamp(min=1).to(totals.dtype)
    mean = mean.masked_fill(counts == 0, float("nan"))
    return mean, counts
