from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Optional

import pandas as pd
import torch
import yaml

from configuration import (
    ExperimentConfig,
    ModelConfig,
    load_experiment_config,
    serialize_resolved_config,
)
from experiments.note6_runner import (
    ModelRunResult,
    build_datasets,
    resolve_device,
    run_model,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate differentiable memory models using the note6 pipeline."
    )
    parser.add_argument(
        "--config",
        required=True,
        type=Path,
        help="Path to the multi-document YAML configuration file.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Optional output directory. Defaults to outputs/<timestamp>.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Allow writing into a non-empty output directory.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        help="Override the random seed defined in the experiment configuration.",
    )
    args = parser.parse_args()

    experiment_cfg, model_cfgs = load_experiment_config(args.config)
    if args.seed is not None:
        experiment_cfg.seed = args.seed

    output_dir = _resolve_output_dir(experiment_cfg, args.output_dir)
    _prepare_output_dir(output_dir, args.overwrite)

    if experiment_cfg.seed is not None:
        torch.manual_seed(experiment_cfg.seed)
    torch.set_grad_enabled(True)

    device = resolve_device(experiment_cfg.device)
    print(f"[note6] Using device: {device}")

    datasets = build_datasets(experiment_cfg)
    print(f"[note6] Generated {len(datasets)} synthetic sequences.")

    results: List[ModelRunResult] = []
    for model_cfg in model_cfgs:
        print(f"[note6] Running model '{model_cfg.name}'...")
        result = run_model(experiment_cfg, model_cfg, datasets, device)
        results.append(result)
        offset_acc = result.offset_zero_accuracy()
        if offset_acc is not None:
            print(f"[note6]   Offset 0 accuracy: {offset_acc:.3f}")

    accuracy_frame = _combine_frames(
        (build_accuracy_frame(result, experiment_cfg) for result in results)
    )
    retrieval_frame = _combine_frames(
        (build_retrieval_frame(result, experiment_cfg) for result in results)
    )
    outer_loss_frame = _combine_frames(
        (
            build_outer_loss_frame(result, experiment_cfg)
            for result in results
            if result.mean_outer_loss is not None
        )
    )

    accuracy_path = output_dir / "offset_accuracy.csv"
    retrieval_path = output_dir / "timestep_retrievals.csv"
    accuracy_frame.to_csv(accuracy_path, index=False)
    retrieval_frame.to_csv(retrieval_path, index=False)
    print(f"[note6] Wrote offset accuracy data to {accuracy_path}")
    print(f"[note6] Wrote timestep retrieval data to {retrieval_path}")

    if outer_loss_frame is not None and not outer_loss_frame.empty:
        outer_loss_path = output_dir / "outer_loss.csv"
        outer_loss_frame.to_csv(outer_loss_path, index=False)
        print(f"[note6] Wrote outer loss data to {outer_loss_path}")

    resolved_config = serialize_resolved_config(experiment_cfg, model_cfgs)
    config_copy_path = output_dir / "resolved_config.yaml"
    with config_copy_path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(resolved_config, handle, sort_keys=False)
    print(f"[note6] Saved resolved configuration to {config_copy_path}")

    summary_payload = {
        "config_path": str(args.config),
        "output_dir": str(output_dir),
        "device": str(device),
        "models": [
            {
                **result.model_config.metadata(),
                "offset_zero_accuracy": result.offset_zero_accuracy(),
            }
            for result in results
        ],
    }
    summary_path = output_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary_payload, handle, indent=2)
    print(f"[note6] Summary written to {summary_path}")


def build_accuracy_frame(result: ModelRunResult, experiment: ExperimentConfig) -> pd.DataFrame:
    offsets = torch.arange(result.mean_accuracy.shape[0], dtype=torch.int64)
    data = {
        "offset": offsets.numpy(),
        "mean_accuracy": result.mean_accuracy.numpy(),
        "observations": result.accuracy_counts.numpy(),
    }
    frame = pd.DataFrame(data)
    metadata = {**experiment.metadata(), **result.model_config.metadata()}
    for key, value in metadata.items():
        frame[key] = value
    return frame


def build_retrieval_frame(result: ModelRunResult, experiment: ExperimentConfig) -> pd.DataFrame:
    timesteps = torch.arange(result.mean_retrievals.shape[0], dtype=torch.int64)
    data = {
        "timestep": timesteps.numpy(),
        "mean_correct_retrievals": result.mean_retrievals.numpy(),
        "observations": result.retrieval_counts.numpy(),
    }
    frame = pd.DataFrame(data)
    metadata = {**experiment.metadata(), **result.model_config.metadata()}
    for key, value in metadata.items():
        frame[key] = value
    return frame


def build_outer_loss_frame(
    result: ModelRunResult,
    experiment: ExperimentConfig,
) -> Optional[pd.DataFrame]:
    if result.mean_outer_loss is None or result.outer_loss_counts is None:
        return None

    timesteps = torch.arange(result.mean_outer_loss.shape[0], dtype=torch.int64)
    data = {
        "timestep": timesteps.numpy(),
        "mean_outer_loss": result.mean_outer_loss.numpy(),
        "observations": result.outer_loss_counts.numpy(),
    }
    frame = pd.DataFrame(data)
    metadata = {**experiment.metadata(), **result.model_config.metadata()}
    for key, value in metadata.items():
        frame[key] = value
    return frame


def _combine_frames(frames: Iterable[Optional[pd.DataFrame]]) -> pd.DataFrame:
    materialized = [frame for frame in frames if frame is not None]
    if not materialized:
        return pd.DataFrame()
    return pd.concat(materialized, ignore_index=True)


def _resolve_output_dir(
    experiment: ExperimentConfig,
    override: Optional[Path],
) -> Path:
    if override is not None:
        return override
    if experiment.output_dir:
        return Path(experiment.output_dir)
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    return Path("outputs") / f"note6_{timestamp}"


def _prepare_output_dir(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite and any(path.iterdir()):
            raise FileExistsError(
                f"Output directory {path} already exists and is not empty. "
                "Pass --overwrite to proceed."
            )
    path.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()
