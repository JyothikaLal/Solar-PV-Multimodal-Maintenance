"""Reusable MLflow experiment-tracking utilities for the Solar PV project."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any, Mapping

import mlflow


DEFAULT_TRACKING_URI = "sqlite:///mlflow.db"

EXPERIMENT_TECNALIA_REGRESSION = "SolarPV_TECNALIA_Regression"
EXPERIMENT_RAPTORMAPS_CLASSIFICATION = "SolarPV_RaptorMaps_Classification"
EXPERIMENT_RAPTORMAPS_TRANSFER = "SolarPV_RaptorMaps_TransferLearning"
EXPERIMENT_FUSION = "SolarPV_Fusion"


def configure_tracking(tracking_uri: str | None = None) -> str:
    """Configure and return the MLflow tracking URI.

    Priority:
    1. Explicit function argument.
    2. MLFLOW_TRACKING_URI environment variable.
    3. Local SQLite backend.
    """
    uri = (
        tracking_uri
        or os.getenv("MLFLOW_TRACKING_URI")
        or DEFAULT_TRACKING_URI
    )

    mlflow.set_tracking_uri(uri)
    return mlflow.get_tracking_uri()


def get_or_create_experiment(experiment_name: str) -> str:
    """Return an experiment ID, creating the experiment when necessary."""
    experiment = mlflow.get_experiment_by_name(experiment_name)

    if experiment is not None:
        return experiment.experiment_id

    return mlflow.create_experiment(experiment_name)


def start_run(
    *,
    experiment_name: str,
    run_name: str,
    tags: Mapping[str, Any] | None = None,
):
    """Start an MLflow run after configuring the tracking backend."""
    configure_tracking()

    experiment_id = get_or_create_experiment(experiment_name)

    normalized_tags = {
        str(key): _serialize_value(value)
        for key, value in (tags or {}).items()
    }

    return mlflow.start_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=normalized_tags,
    )


def log_params(params: Mapping[str, Any]) -> None:
    """Log a parameter mapping after normalizing complex values."""
    mlflow.log_params(
        {
            str(key): _serialize_value(value)
            for key, value in params.items()
        }
    )


def log_metrics(metrics: Mapping[str, float | int]) -> None:
    """Log numeric metrics."""
    normalized: dict[str, float] = {}

    for key, value in metrics.items():
        if isinstance(value, bool):
            normalized[str(key)] = float(value)
        elif isinstance(value, (int, float)):
            normalized[str(key)] = float(value)
        else:
            raise TypeError(
                f"Metric '{key}' must be numeric, got {type(value).__name__}."
            )

    mlflow.log_metrics(normalized)


def log_tags(tags: Mapping[str, Any]) -> None:
    """Log additional run tags."""
    mlflow.set_tags(
        {
            str(key): _serialize_value(value)
            for key, value in tags.items()
        }
    )


def log_artifact(path: str | Path, artifact_path: str | None = None) -> None:
    """Log one local file as an MLflow artifact."""
    artifact = Path(path)

    if not artifact.is_file():
        raise FileNotFoundError(f"Artifact does not exist: {artifact}")

    mlflow.log_artifact(
        str(artifact),
        artifact_path=artifact_path,
    )


def log_artifacts(
    directory: str | Path,
    artifact_path: str | None = None,
) -> None:
    """Log all files from a local directory."""
    artifact_dir = Path(directory)

    if not artifact_dir.is_dir():
        raise FileNotFoundError(
            f"Artifact directory does not exist: {artifact_dir}"
        )

    mlflow.log_artifacts(
        str(artifact_dir),
        artifact_path=artifact_path,
    )


def log_dataset_metadata(
    *,
    dataset_name: str,
    dataset_version: str | None = None,
    manifest_path: str | Path | None = None,
    split_strategy: str | None = None,
    sample_counts: Mapping[str, int] | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Log dataset identity and split metadata without uploading the dataset."""
    tags: dict[str, Any] = {
        "dataset_name": dataset_name,
    }

    if dataset_version is not None:
        tags["dataset_version"] = dataset_version

    if manifest_path is not None:
        tags["dataset_manifest"] = str(manifest_path)

    if split_strategy is not None:
        tags["split_strategy"] = split_strategy

    if sample_counts is not None:
        tags["sample_counts"] = dict(sample_counts)

    if metadata:
        tags.update(metadata)

    log_tags(tags)


def build_standard_tags(
    *,
    dataset: str,
    modality: str,
    task: str,
    model_family: str,
    stage: str,
    run_type: str,
    project: str = "SolarPV_Multimodal_Maintenance",
    split_strategy: str | None = None,
) -> dict[str, str]:
    """Build the standard metadata vocabulary used across experiments."""
    tags = {
        "project": project,
        "dataset": dataset,
        "modality": modality,
        "task": task,
        "model_family": model_family,
        "stage": stage,
        "run_type": run_type,
    }

    if split_strategy is not None:
        tags["split_strategy"] = split_strategy

    return tags


def _serialize_value(value: Any) -> str:
    """Convert MLflow metadata values to stable string representations."""
    if isinstance(value, str):
        return value

    if value is None:
        return "None"

    if isinstance(value, (bool, int, float)):
        return str(value)

    try:
        return json.dumps(
            value,
            sort_keys=True,
            default=str,
            separators=(",", ":"),
        )
    except (TypeError, ValueError):
        return str(value)

    
