from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import mlflow
import torch

from src.mlops.model_registry import (
    LIFECYCLE_CANDIDATE,
    REGISTERED_MODEL_EFFICIENTNET_B0,
    REGISTERED_MODEL_RESNET18,
    build_model_metadata,
    register_candidate_model,
)
from src.mlops.retraining.acceptance_contract import AcceptanceDecision
from src.mlops.retraining.evaluation import CandidateEvaluation
from src.mlops.retraining.training import ClassificationTrainingResult
from scripts.register_raptormaps_models import RaptorMapsMLflowModel


@dataclass(frozen=True)
class CandidateRegistrationResult:
    registered_model_name: str
    model_family: str
    run_id: str
    model_uri: str
    version: str


def _registered_model_name(model_family: str) -> str:
    if model_family == "resnet18_finetuned":
        return REGISTERED_MODEL_RESNET18

    if model_family == "efficientnet_b0_finetuned":
        return REGISTERED_MODEL_EFFICIENTNET_B0

    raise ValueError(
        f"Unsupported RaptorMaps model family: {model_family}"
    )


def _registry_model_family(model_family: str) -> str:
    if model_family == "resnet18_finetuned":
        return "resnet18"

    if model_family == "efficientnet_b0_finetuned":
        return "efficientnet_b0"

    raise ValueError(
        f"Unsupported RaptorMaps model family: {model_family}"
    )


def _load_checkpoint_metadata(
    checkpoint_path: Path,
) -> dict:
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Candidate checkpoint not found: {checkpoint_path}"
        )

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "epoch",
        "best_validation_macro_f1",
        "model_state_dict",
    }

    missing = required - set(checkpoint)
    if missing:
        raise ValueError(
            f"Candidate checkpoint is missing required fields: "
            f"{sorted(missing)}"
        )

    return checkpoint


def register_raptormaps_candidate(
    *,
    result: ClassificationTrainingResult,
    evaluation: CandidateEvaluation,
    tracking_uri: str = "sqlite:///mlflow.db",
) -> CandidateRegistrationResult:
    """
    Package and register an accepted RaptorMaps candidate.

    This function only creates a candidate registry version.
    It never promotes a model to production.
    """

    if evaluation.acceptance.decision != AcceptanceDecision.ACCEPT:
        raise ValueError(
            "RaptorMaps candidate cannot be registered unless "
            "acceptance decision is ACCEPT."
        )

    checkpoint = _load_checkpoint_metadata(result.checkpoint_path)

    registered_model_name = _registered_model_name(
        result.model_family
    )
    registry_model_family = _registry_model_family(
        result.model_family
    )

    training_config = {
        "class_names": list(result.class_names),
        "spatial_size": [160, 96],
        "pretrained": True,
        "fine_tuning": True,
        "trainable_backbone": (
            "layer4"
            if result.model_family == "resnet18_finetuned"
            else "features[8]"
        ),
        "seed": 42,
        "best_epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(
            checkpoint["best_validation_macro_f1"]
        ),
    }

    mlflow.set_tracking_uri(tracking_uri)

    with mlflow.start_run(
        run_name=f"{registry_model_family}_retraining_candidate",
        tags={
            "project": "Solar_PV_Multimodal_Maintenance",
            "dataset": "RaptorMaps",
            "modality": "thermal",
            "task": "classification",
            "stage": "retraining",
            "run_type": "candidate_registration",
            "lifecycle_stage": LIFECYCLE_CANDIDATE,
        },
    ) as run:

        model_metadata = build_model_metadata(
            model_family=registry_model_family,
            source_run_id=run.info.run_id,
            source_checkpoint=str(result.checkpoint_path),
            training_config=training_config,
            test_metrics=dict(result.test_metrics),
            lifecycle_stage=LIFECYCLE_CANDIDATE,
        )

        mlflow.log_params(
            {
                "model_family": registry_model_family,
                "registered_model_name": registered_model_name,
                "checkpoint_epoch": int(checkpoint["epoch"]),
                "spatial_height": 160,
                "spatial_width": 96,
                "num_classes": len(result.class_names),
            }
        )

        mlflow.log_metric(
            "best_validation_macro_f1",
            float(checkpoint["best_validation_macro_f1"]),
        )

        for metric, value in result.test_metrics.items():
            mlflow.log_metric(
                f"test_{metric}",
                float(value),
            )

        artifact_path = "raptormaps_model"

        model = RaptorMapsMLflowModel(
            model_family=registry_model_family,
            checkpoint_path=str(result.checkpoint_path),
        )

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            artifacts={
                "checkpoint": str(result.checkpoint_path),
            },
            pip_requirements=[
                "mlflow==3.16.0",
                "torch",
                "torchvision",
                "numpy",
                "pandas",
            ],
        )

        model_uri = (
            f"runs:/{run.info.run_id}/{artifact_path}"
        )

        version = register_candidate_model(
            registered_model_name=registered_model_name,
            model_uri=model_uri,
            metadata=model_metadata,
            description=(
                f"Accepted RaptorMaps retraining candidate: "
                f"{registry_model_family}"
            ),
            tracking_uri=tracking_uri,
        )

    return CandidateRegistrationResult(
        registered_model_name=registered_model_name,
        model_family=registry_model_family,
        run_id=run.info.run_id,
        model_uri=model_uri,
        version=str(version.version),
    )
