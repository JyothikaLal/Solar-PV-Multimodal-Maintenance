from __future__ import annotations

import json

import pytest

from src.mlops.model_registry import (
    LIFECYCLE_ARCHIVED,
    LIFECYCLE_CANDIDATE,
    LIFECYCLE_PRODUCTION,
    RAPTORMAPS_CLASS_NAMES,
    RAPTORMAPS_SPATIAL_SIZE,
    build_model_metadata,
    validate_model_compatibility,
)


def valid_training_config() -> dict:
    return {
        "class_names": RAPTORMAPS_CLASS_NAMES.copy(),
        "spatial_size": RAPTORMAPS_SPATIAL_SIZE.copy(),
        "pretrained": True,
        "fine_tuning": True,
        "trainable_backbone": "layer4",
        "seed": 42,
        "best_epoch": 14,
        "best_validation_macro_f1": 0.6507379837195436,
    }


def valid_test_metrics() -> dict[str, float]:
    return {
        "accuracy": 0.784594864954985,
        "balanced_accuracy": 0.6160580192563342,
        "macro_precision": 0.6699010128316284,
        "macro_recall": 0.6160580192563342,
        "macro_f1": 0.6385971332082477,
        "weighted_f1": 0.7790920175787409,
        "roc_auc_ovr_macro": 0.9424316268041923,
        "pr_auc_macro": 0.6583795510819005,
    }


def build_valid_metadata(
    lifecycle_stage: str = LIFECYCLE_CANDIDATE,
) -> dict[str, str]:
    return build_model_metadata(
        model_family="resnet18",
        source_run_id="test-run-id",
        source_checkpoint="models/raptormaps/resnet18_finetune/best_model.pt",
        training_config=valid_training_config(),
        test_metrics=valid_test_metrics(),
        lifecycle_stage=lifecycle_stage,
    )


def test_valid_metadata_matches_raptormaps_contract():
    metadata = build_valid_metadata()

    validate_model_compatibility(metadata)


@pytest.mark.parametrize(
    "lifecycle_stage",
    [
        LIFECYCLE_CANDIDATE,
        LIFECYCLE_PRODUCTION,
        LIFECYCLE_ARCHIVED,
    ],
)
def test_supported_lifecycle_stages_are_accepted(
    lifecycle_stage: str,
):
    metadata = build_valid_metadata(lifecycle_stage)

    assert metadata["lifecycle_stage"] == lifecycle_stage


def test_invalid_lifecycle_stage_is_rejected():
    with pytest.raises(ValueError, match="Unsupported lifecycle stage"):
        build_valid_metadata("invalid_stage")


def test_wrong_dataset_is_rejected():
    metadata = build_valid_metadata()
    metadata["dataset"] = "DifferentDataset"

    with pytest.raises(
        ValueError,
        match="Dataset compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_wrong_modality_is_rejected():
    metadata = build_valid_metadata()
    metadata["modality"] = "telemetry"

    with pytest.raises(
        ValueError,
        match="Modality compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_wrong_task_is_rejected():
    metadata = build_valid_metadata()
    metadata["task"] = "regression"

    with pytest.raises(
        ValueError,
        match="Task compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_wrong_class_count_is_rejected():
    metadata = build_valid_metadata()
    metadata["class_count"] = "11"

    with pytest.raises(
        ValueError,
        match="Class-count compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_wrong_spatial_size_is_rejected():
    metadata = build_valid_metadata()
    metadata["spatial_size"] = json.dumps([224, 224])

    with pytest.raises(
        ValueError,
        match="Spatial-size compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_wrong_class_names_are_rejected():
    metadata = build_valid_metadata()
    metadata["class_names"] = json.dumps(
        RAPTORMAPS_CLASS_NAMES[:-1]
    )

    with pytest.raises(
        ValueError,
        match="Class-name compatibility check failed",
    ):
        validate_model_compatibility(metadata)


def test_metadata_contains_required_model_provenance():
    metadata = build_valid_metadata()

    required = {
        "project",
        "dataset",
        "dataset_version",
        "modality",
        "task",
        "split_strategy",
        "model_family",
        "source_run_id",
        "source_checkpoint",
        "pretrained",
        "fine_tuning",
        "trainable_backbone",
        "seed",
        "best_epoch",
        "best_validation_macro_f1",
        "class_count",
        "class_names",
        "spatial_size",
        "lifecycle_stage",
        "test_accuracy",
        "test_balanced_accuracy",
        "test_macro_precision",
        "test_macro_recall",
        "test_macro_f1",
        "test_weighted_f1",
        "test_roc_auc_ovr_macro",
        "test_pr_auc_macro",
    }

    assert required.issubset(metadata)

def test_promoting_new_version_archives_existing_production(
    tmp_path,
):
    import mlflow
    from mlflow import MlflowClient

    from src.mlops.model_registry import (
        transition_model_version,
    )

    tracking_uri = (
        f"sqlite:///{tmp_path / 'registry.db'}"
    )

    mlflow.set_tracking_uri(tracking_uri)
    client = MlflowClient()

    model_name = "Test_RaptorMaps_Model"

    client.create_registered_model(
        model_name,
        description="Test registry model",
    )

    version_1 = client.create_model_version(
        name=model_name,
        source="runs:/test-run-1/model",
        run_id="test-run-1",
    )

    version_2 = client.create_model_version(
        name=model_name,
        source="runs:/test-run-2/model",
        run_id="test-run-2",
    )

    client.set_model_version_tag(
        model_name,
        version_1.version,
        "lifecycle_stage",
        LIFECYCLE_PRODUCTION,
    )

    client.set_model_version_tag(
        model_name,
        version_2.version,
        "lifecycle_stage",
        LIFECYCLE_CANDIDATE,
    )

    transition_model_version(
        registered_model_name=model_name,
        version=str(version_2.version),
        lifecycle_stage=LIFECYCLE_PRODUCTION,
        tracking_uri=tracking_uri,
    )

    v1 = client.get_model_version(
        model_name,
        version_1.version,
    )
    v2 = client.get_model_version(
        model_name,
        version_2.version,
    )

    assert (
        v1.tags["lifecycle_stage"]
        == LIFECYCLE_ARCHIVED
    )

    assert (
        v2.tags["lifecycle_stage"]
        == LIFECYCLE_PRODUCTION
    )