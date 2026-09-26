from __future__ import annotations

import json
from typing import Any

import mlflow
from mlflow import MlflowClient


RAPTORMAPS_DATASET = "RaptorMaps"
RAPTORMAPS_DATASET_VERSION = "project_locked"
RAPTORMAPS_MODALITY = "thermal"
RAPTORMAPS_TASK = "classification"
RAPTORMAPS_SPLIT_STRATEGY = "frozen_raptormaps_manifest"

RAPTORMAPS_CLASS_NAMES = [
    "Cell",
    "Cell-Multi",
    "Cracking",
    "Diode",
    "Diode-Multi",
    "Hot-Spot",
    "Hot-Spot-Multi",
    "No-Anomaly",
    "Offline-Module",
    "Shadowing",
    "Soiling",
    "Vegetation",
]

RAPTORMAPS_SPATIAL_SIZE = [160, 96]

REGISTERED_MODEL_RESNET18 = "SolarPV_RaptorMaps_ResNet18"
REGISTERED_MODEL_EFFICIENTNET_B0 = "SolarPV_RaptorMaps_EfficientNetB0"

LIFECYCLE_CANDIDATE = "candidate"
LIFECYCLE_PRODUCTION = "production"
LIFECYCLE_ARCHIVED = "archived"

VALID_LIFECYCLE_STAGES = {
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_PRODUCTION,
    LIFECYCLE_ARCHIVED,
}


def configure_registry(tracking_uri: str = "sqlite:///mlflow.db") -> MlflowClient:
    """Configure MLflow and return a registry client."""
    mlflow.set_tracking_uri(tracking_uri)
    return MlflowClient()


def build_model_metadata(
    *,
    model_family: str,
    source_run_id: str,
    source_checkpoint: str,
    training_config: dict[str, Any],
    test_metrics: dict[str, float],
    lifecycle_stage: str = LIFECYCLE_CANDIDATE,
) -> dict[str, str]:
    """Build validated registry metadata for a RaptorMaps model."""

    if lifecycle_stage not in VALID_LIFECYCLE_STAGES:
        raise ValueError(
            f"Unsupported lifecycle stage: {lifecycle_stage}"
        )

    class_names = training_config.get("class_names")
    spatial_size = training_config.get("spatial_size")

    if class_names != RAPTORMAPS_CLASS_NAMES:
        raise ValueError("RaptorMaps class contract does not match.")

    if spatial_size != RAPTORMAPS_SPATIAL_SIZE:
        raise ValueError("RaptorMaps spatial-size contract does not match.")

    required_metrics = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    }

    missing_metrics = required_metrics - set(test_metrics)
    if missing_metrics:
        raise ValueError(
            f"Missing required test metrics: {sorted(missing_metrics)}"
        )

    return {
        "project": "Solar_PV_Multimodal_Maintenance",
        "dataset": RAPTORMAPS_DATASET,
        "dataset_version": RAPTORMAPS_DATASET_VERSION,
        "modality": RAPTORMAPS_MODALITY,
        "task": RAPTORMAPS_TASK,
        "split_strategy": RAPTORMAPS_SPLIT_STRATEGY,
        "model_family": model_family,
        "source_run_id": source_run_id,
        "source_checkpoint": source_checkpoint,
        "pretrained": str(training_config["pretrained"]).lower(),
        "fine_tuning": str(training_config["fine_tuning"]).lower(),
        "trainable_backbone": str(
            training_config["trainable_backbone"]
        ),
        "seed": str(training_config["seed"]),
        "best_epoch": str(training_config["best_epoch"]),
        "best_validation_macro_f1": str(
            training_config["best_validation_macro_f1"]
        ),
        "class_count": str(len(class_names)),
        "class_names": json.dumps(class_names),
        "spatial_size": json.dumps(spatial_size),
        "lifecycle_stage": lifecycle_stage,
        "test_accuracy": str(test_metrics["accuracy"]),
        "test_balanced_accuracy": str(test_metrics["balanced_accuracy"]),
        "test_macro_precision": str(test_metrics["macro_precision"]),
        "test_macro_recall": str(test_metrics["macro_recall"]),
        "test_macro_f1": str(test_metrics["macro_f1"]),
        "test_weighted_f1": str(test_metrics["weighted_f1"]),
        "test_roc_auc_ovr_macro": str(
            test_metrics["roc_auc_ovr_macro"]
        ),
        "test_pr_auc_macro": str(test_metrics["pr_auc_macro"]),
    }


def validate_model_compatibility(
    metadata: dict[str, str],
    *,
    expected_dataset: str = RAPTORMAPS_DATASET,
    expected_modality: str = RAPTORMAPS_MODALITY,
    expected_task: str = RAPTORMAPS_TASK,
    expected_class_count: int = 12,
    expected_spatial_size: list[int] | None = None,
) -> None:
    """Validate registry metadata against the inference contract."""

    expected_spatial_size = (
        expected_spatial_size or RAPTORMAPS_SPATIAL_SIZE
    )

    if metadata["dataset"] != expected_dataset:
        raise ValueError("Dataset compatibility check failed.")

    if metadata["modality"] != expected_modality:
        raise ValueError("Modality compatibility check failed.")

    if metadata["task"] != expected_task:
        raise ValueError("Task compatibility check failed.")

    if int(metadata["class_count"]) != expected_class_count:
        raise ValueError("Class-count compatibility check failed.")

    if json.loads(metadata["spatial_size"]) != expected_spatial_size:
        raise ValueError("Spatial-size compatibility check failed.")

    if json.loads(metadata["class_names"]) != RAPTORMAPS_CLASS_NAMES:
        raise ValueError("Class-name compatibility check failed.")


def register_candidate_model(
    *,
    registered_model_name: str,
    model_uri: str,
    metadata: dict[str, str],
    description: str,
    tracking_uri: str = "sqlite:///mlflow.db",
):
    """Register a model artifact as a candidate model version."""

    validate_model_compatibility(metadata)

    client = configure_registry(tracking_uri)

    try:
        client.get_registered_model(registered_model_name)
    except Exception:
        client.create_registered_model(
            registered_model_name,
            description=description,
        )

    version = mlflow.register_model(
        model_uri=model_uri,
        name=registered_model_name,
    )

    client.set_model_version_tag(
        registered_model_name,
        version.version,
        "lifecycle_stage",
        metadata["lifecycle_stage"],
    )

    for key, value in metadata.items():
        if key != "lifecycle_stage":
            client.set_model_version_tag(
                registered_model_name,
                version.version,
                key,
                value,
            )

    client.update_model_version(
        name=registered_model_name,
        version=version.version,
        description=description,
    )

    return version


def transition_model_version(
    *,
    registered_model_name: str,
    version: str,
    lifecycle_stage: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> None:
    """Explicitly transition a model version through the project lifecycle.

    Lifecycle rules:
    - candidate -> production is allowed.
    - candidate -> archived is allowed.
    - production -> archived is allowed.
    - production -> candidate is allowed only as an explicit tag update.
    - archived -> candidate/production is allowed only when explicitly requested.
    - When promoting a version to production, any existing production
      version of the same registered model is automatically archived.
    """

    if lifecycle_stage not in VALID_LIFECYCLE_STAGES:
        raise ValueError(
            f"Unsupported lifecycle stage: {lifecycle_stage}"
        )

    client = configure_registry(tracking_uri)

    # Verify that the requested model version actually exists.
    target_version = client.get_model_version(
        registered_model_name,
        version,
    )

    if lifecycle_stage == LIFECYCLE_PRODUCTION:
        versions = client.search_model_versions(
            f"name='{registered_model_name}'"
        )

        for existing_version in versions:
            if (
                str(existing_version.version) != str(target_version.version)
                and existing_version.tags.get("lifecycle_stage")
                == LIFECYCLE_PRODUCTION
            ):
                client.set_model_version_tag(
                    registered_model_name,
                    existing_version.version,
                    "lifecycle_stage",
                    LIFECYCLE_ARCHIVED,
                )

    client.set_model_version_tag(
        registered_model_name,
        target_version.version,
        "lifecycle_stage",
        lifecycle_stage,
    )


def get_registered_model_metadata(
    *,
    registered_model_name: str,
    version: str,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> dict[str, str]:
    """Return project metadata attached to a registered model version."""

    client = configure_registry(tracking_uri)

    model_version = client.get_model_version(
        registered_model_name,
        version,
    )

    return dict(model_version.tags)