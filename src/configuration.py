from __future__ import annotations

from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import json
import yaml


__all__ = [
    "ExperimentConfig",
    "LossConfig",
    "ModelConfig",
    "ModelSpec",
    "TrainerConfig",
    "load_experiment_config",
    "serialize_resolved_config",
]


@dataclass
class ExperimentConfig:
    key_dim: int
    val_dim: int
    context_size: int
    sequence_length: int
    num_sequences: int
    seed: Optional[int] = None
    device: Optional[str] = None
    dataset: Dict[str, Any] = field(default_factory=dict)
    output_dir: Optional[str] = None

    def metadata(self) -> Dict[str, Any]:
        data = {
            "experiment_key_dim": self.key_dim,
            "experiment_val_dim": self.val_dim,
            "experiment_context_size": self.context_size,
            "experiment_sequence_length": self.sequence_length,
            "experiment_num_sequences": self.num_sequences,
        }
        if self.seed is not None:
            data["experiment_seed"] = self.seed
        if self.device is not None:
            data["experiment_device"] = self.device
        if self.output_dir is not None:
            data["experiment_output_dir"] = self.output_dir

        dataset_params = self.dataset or {}
        data["experiment_dataset_params"] = json.dumps(dataset_params, sort_keys=True)
        return data

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSpec:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TrainerConfig:
    type: str
    params: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LossConfig:
    name: str
    params: Dict[str, Any] = field(default_factory=dict)

    def metadata(self, prefix: str) -> Dict[str, Any]:
        return {
            f"{prefix}_name": self.name,
            f"{prefix}_params": json.dumps(self.params, sort_keys=True),
        }


@dataclass
class ModelConfig:
    name: str
    model: ModelSpec
    inner_loss: LossConfig
    trainer: Optional[TrainerConfig] = None
    outer_loss: Optional[LossConfig] = None
    notes: Optional[str] = None

    def metadata(self) -> Dict[str, Any]:
        data = {
            "model_name": self.name,
            "model_type": self.model.type,
            "model_params": json.dumps(self.model.params, sort_keys=True),
        }
        if self.notes:
            data["model_notes"] = self.notes

        if self.trainer is not None:
            data["trainer_type"] = self.trainer.type
            data["trainer_params"] = json.dumps(self.trainer.params, sort_keys=True)
        else:
            data["trainer_type"] = "none"
            data["trainer_params"] = "{}"

        data.update(self.inner_loss.metadata("inner_loss"))
        if self.outer_loss is not None:
            data.update(self.outer_loss.metadata("outer_loss"))
        else:
            data["outer_loss_name"] = "none"
            data["outer_loss_params"] = "{}"
        return data

    def to_dict(self) -> Dict[str, Any]:
        payload: Dict[str, Any] = {
            "name": self.name,
            "model": asdict(self.model),
            "inner_loss": asdict(self.inner_loss),
        }
        if self.trainer is not None:
            payload["trainer"] = asdict(self.trainer)
        if self.outer_loss is not None:
            payload["outer_loss"] = asdict(self.outer_loss)
        if self.notes:
            payload["notes"] = self.notes
        return payload


def load_experiment_config(path: Path | str) -> Tuple[ExperimentConfig, List[ModelConfig]]:
    """Load experiment and model configurations from a multi-document YAML file."""
    config_path = Path(path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file {config_path} does not exist")

    with config_path.open("r", encoding="utf-8") as handle:
        documents = list(yaml.safe_load_all(handle))

    if not documents:
        raise ValueError(f"Config file {config_path} did not contain any YAML documents")

    experiment_doc = documents[0] or {}
    if "experiment" in experiment_doc and isinstance(experiment_doc["experiment"], dict):
        experiment_payload = experiment_doc["experiment"]
    else:
        experiment_payload = experiment_doc

    experiment = ExperimentConfig(
        key_dim=_require_field(experiment_payload, "key_dim", int, "experiment"),
        val_dim=_require_field(experiment_payload, "val_dim", int, "experiment"),
        context_size=_require_field(experiment_payload, "context_size", int, "experiment"),
        sequence_length=_require_field(experiment_payload, "sequence_length", int, "experiment"),
        num_sequences=_require_field(experiment_payload, "num_sequences", int, "experiment"),
        seed=experiment_payload.get("seed"),
        device=experiment_payload.get("device"),
        dataset=dict(experiment_payload.get("dataset", {}) or {}),
        output_dir=experiment_payload.get("output_dir"),
    )

    model_docs = documents[1:]
    if not model_docs:
        raise ValueError(
            "Config file must contain at least one model document following the experiment block"
        )

    models: List[ModelConfig] = []
    for index, raw_doc in enumerate(model_docs, start=2):
        if raw_doc is None:
            continue
        models.append(_parse_model_doc(raw_doc, index))

    if not models:
        raise ValueError("No model configurations were parsed from the config file")

    return experiment, models


def serialize_resolved_config(
    experiment: ExperimentConfig, models: Iterable[ModelConfig]
) -> Dict[str, Any]:
    return {
        "experiment": experiment.to_dict(),
        "models": [model.to_dict() for model in models],
    }


def _parse_model_doc(doc: Dict[str, Any], doc_index: int) -> ModelConfig:
    if "name" not in doc:
        raise ValueError(f"Model document #{doc_index} is missing the required 'name' field")

    name = str(doc["name"])
    model_section = doc.get("model", {})
    if not isinstance(model_section, dict):
        raise ValueError(f"Model document #{doc_index} must contain a 'model' mapping")

    model = ModelSpec(
        type=_require_field(model_section, "type", str, f"model '{name}'").lower(),
        params=dict(model_section.get("params", {}) or {}),
    )

    trainer_section = doc.get("trainer")
    trainer = None
    if trainer_section is not None:
        if not isinstance(trainer_section, dict):
            raise ValueError(f"Model document '{name}' trainer section must be a mapping")
        trainer = TrainerConfig(
            type=_require_field(trainer_section, "type", str, f"trainer for '{name}'").lower(),
            params=dict(trainer_section.get("params", {}) or {}),
        )

    inner_loss_section = doc.get("inner_loss")
    if inner_loss_section is None:
        raise ValueError(f"Model document '{name}' must specify an 'inner_loss' section")
    inner_loss = _parse_loss(inner_loss_section, f"inner_loss for '{name}'")

    outer_loss_section = doc.get("outer_loss")
    outer_loss = _parse_loss(outer_loss_section, f"outer_loss for '{name}'", allow_none=True)

    notes = doc.get("notes")
    return ModelConfig(
        name=name,
        model=model,
        inner_loss=inner_loss,
        trainer=trainer,
        outer_loss=outer_loss,
        notes=notes,
    )


def _parse_loss(
    payload: Any,
    context: str,
    allow_none: bool = False,
) -> Optional[LossConfig]:
    if payload is None:
        if allow_none:
            return None
        raise ValueError(f"{context} must not be null")

    if isinstance(payload, str):
        name = payload
        params: Dict[str, Any] = {}
    elif isinstance(payload, dict):
        name = payload.get("name")
        params = dict(payload.get("params", {}) or {})
    else:
        raise ValueError(f"{context} must be a string or mapping, got {type(payload).__name__}")

    if name is None:
        raise ValueError(f"{context} is missing the required 'name' field")

    normalized = str(name).lower()
    if allow_none and normalized in {"none", "null"}:
        return None

    return LossConfig(name=normalized, params=params)


def _require_field(
    payload: Dict[str, Any],
    key: str,
    expected_type: type,
    context: str,
) -> Any:
    if key not in payload:
        raise ValueError(f"Missing required field '{key}' in {context} configuration")
    value = payload[key]
    if not isinstance(value, expected_type):
        raise ValueError(
            f"Field '{key}' in {context} must be of type {expected_type.__name__}, "
            f"got {type(value).__name__}"
        )
    return value
