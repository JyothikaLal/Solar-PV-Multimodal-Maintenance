from __future__ import annotations

import json
import sys
from pathlib import Path

if str(Path(__file__).resolve().parents[1]) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import mlflow
import pandas as pd
import torch
from mlflow import MlflowClient
from mlflow.pyfunc import PythonModel

from src.mlops.model_registry import (
    LIFECYCLE_CANDIDATE,
    REGISTERED_MODEL_EFFICIENTNET_B0,
    REGISTERED_MODEL_RESNET18,
    build_model_metadata,
    configure_registry,
    register_candidate_model,
)

from src.models.vision.raptormaps_transfer import RaptorMapsTransferAdapter
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]

TRACKING_URI = "sqlite:///mlflow.db"

RESNET_CHECKPOINT = (
    PROJECT_ROOT
    / "models/raptormaps/resnet18_finetune/best_model.pt"
)

EFFICIENTNET_CHECKPOINT = (
    PROJECT_ROOT
    / "models/raptormaps/efficientnet_b0_finetune/best_model.pt"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "reports/results/mlflow/registry"
)

CLASS_NAMES = [
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


class RaptorMapsMLflowModel(PythonModel):
    """MLflow pyfunc wrapper around the exact RaptorMaps transfer model."""

    def __init__(
        self,
        model_family: str,
        checkpoint_path: str,
    ) -> None:
        self.model_family = model_family
        self.checkpoint_path = checkpoint_path
        self.model = None

    def load_context(self, context) -> None:
        if self.model_family == "resnet18":
            backbone = build_pretrained_resnet18(num_classes=12)
            classifier_module = backbone.fc

        elif self.model_family == "efficientnet_b0":
            backbone = build_pretrained_efficientnet_b0(num_classes=12)
            classifier_module = backbone.classifier

        else:
            raise ValueError(
                f"Unsupported model family: {self.model_family}"
            )

        self.model = RaptorMapsTransferAdapter(
            backbone=backbone,
            classifier_module=classifier_module,
            spatial_size=(160, 96),
        )

        checkpoint = torch.load(
            context.artifacts["checkpoint"],
            map_location="cpu",
            weights_only=False,
        )

        if "model_state_dict" not in checkpoint:
            raise RuntimeError(
                "Checkpoint does not contain model_state_dict."
            )

        self.model.load_state_dict(
            checkpoint["model_state_dict"],
            strict=True,
        )

        self.model.eval()

    def predict(self, context, model_input):
        import numpy as np

        if hasattr(model_input, "to_numpy"):
            array = model_input.to_numpy()
        else:
            array = np.asarray(model_input)

        tensor = torch.as_tensor(
            array,
            dtype=torch.float32,
        )

        if tensor.ndim == 3:
            tensor = tensor.unsqueeze(1)

        if tensor.ndim != 4:
            raise ValueError(
                "Expected input shape [B, 1, H, W]."
            )

        with torch.no_grad():
            logits = self.model(tensor)
            probabilities = torch.softmax(logits, dim=1)
            predictions = torch.argmax(
                probabilities,
                dim=1,
            )

        return predictions.cpu().numpy()


def load_checkpoint_metadata(checkpoint_path: Path) -> dict:
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=False,
    )

    required = {
        "best_validation_macro_f1",
        "epoch",
        "model_state_dict",
        "optimizer_state_dict",
    }

    missing = required - set(checkpoint)

    if missing:
        raise RuntimeError(
            f"{checkpoint_path}: missing checkpoint fields {sorted(missing)}"
        )

    return checkpoint


def verify_model_before_logging(
    model_family: str,
    checkpoint_path: Path,
) -> dict:
    checkpoint = load_checkpoint_metadata(checkpoint_path)

    if model_family == "resnet18":
        backbone = build_pretrained_resnet18(num_classes=12)
        classifier_module = backbone.fc
    elif model_family == "efficientnet_b0":
        backbone = build_pretrained_efficientnet_b0(num_classes=12)
        classifier_module = backbone.classifier
    else:
        raise ValueError(
            f"Unsupported model family: {model_family}"
        )

    model = RaptorMapsTransferAdapter(
        backbone=backbone,
        classifier_module=classifier_module,
        spatial_size=(160, 96),
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    model.eval()

    x = torch.zeros(
        2,
        1,
        24,
        40,
        dtype=torch.float32,
    )

    with torch.no_grad():
        logits = model(x)

    if tuple(logits.shape) != (2, 12):
        raise RuntimeError(
            f"{model_family}: unexpected output shape "
            f"{tuple(logits.shape)}"
        )

    if not torch.isfinite(logits).all():
        raise RuntimeError(
            f"{model_family}: non-finite output."
        )

    return {
        "checkpoint": str(checkpoint_path),
        "epoch": int(checkpoint["epoch"]),
        "best_validation_macro_f1": float(
            checkpoint["best_validation_macro_f1"]
        ),
        "state_dict_keys": len(checkpoint["model_state_dict"]),
        "output_shape": list(logits.shape),
    }


def log_and_register(
    *,
    model_family: str,
    registered_model_name: str,
    checkpoint_path: Path,
    test_metrics_path: Path,
    training_config_path: Path,
    model_metadata: dict[str, str],
    client: MlflowClient,
) -> dict:
    verification = verify_model_before_logging(
        model_family=model_family,
        checkpoint_path=checkpoint_path,
    )

    with mlflow.start_run(
        experiment_id=5,
        run_name=f"{model_family}_registry_packaging",
        tags={
            "project": "Solar_PV_Multimodal_Maintenance",
            "dataset": "RaptorMaps",
            "modality": "thermal",
            "task": "classification",
            "stage": "registry",
            "run_type": "model_packaging",
            "split_strategy": "frozen_raptormaps_manifest",
        },
    ) as run:

        mlflow.log_params(
            {
                "model_family": model_family,
                "registered_model_name": registered_model_name,
                "checkpoint_epoch": verification["epoch"],
                "spatial_height": 160,
                "spatial_width": 96,
                "num_classes": 12,
            }
        )

        mlflow.log_metric(
            "best_validation_macro_f1",
            verification["best_validation_macro_f1"],
        )

        mlflow.log_artifact(
            str(test_metrics_path),
            artifact_path="model_metadata",
        )

        mlflow.log_artifact(
            str(training_config_path),
            artifact_path="model_metadata",
        )

        model = RaptorMapsMLflowModel(
            model_family=model_family,
            checkpoint_path=str(checkpoint_path),
        )

        artifact_path = "raptormaps_model"

        mlflow.pyfunc.log_model(
            artifact_path=artifact_path,
            python_model=model,
            artifacts={
                "checkpoint": str(checkpoint_path),
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
                f"RaptorMaps {model_family} fine-tuned candidate "
                "registered from the validated project checkpoint."
            ),
            tracking_uri=TRACKING_URI,
        )

        result = {
            "model_family": model_family,
            "registered_model_name": registered_model_name,
            "version": int(version.version),
            "run_id": run.info.run_id,
            "model_uri": model_uri,
            "lifecycle_stage": LIFECYCLE_CANDIDATE,
            "verification": verification,
        }

        return result


def main() -> None:
    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    client = configure_registry(
        tracking_uri=TRACKING_URI,
    )

    resnet_verification = verify_model_before_logging(
        "resnet18",
        RESNET_CHECKPOINT,
    )

    efficientnet_verification = verify_model_before_logging(
        "efficientnet_b0",
        EFFICIENTNET_CHECKPOINT,
    )

    resnet_training_config_path = (
        PROJECT_ROOT
        / "reports/results/raptormaps/"
        "resnet18_finetune/training_config.json"
    )

    resnet_test_metrics_path = (
        PROJECT_ROOT
        / "reports/results/raptormaps/"
        "resnet18_finetune/test_overall_metrics.csv"
    )

    resnet_training_config = load_training_config(
        resnet_training_config_path
    )

    resnet_test_metrics = load_test_metrics(
        resnet_test_metrics_path
    )

    resnet_metadata = build_model_metadata(
        model_family="resnet18",
        source_run_id="a147e10469144f96a3fb552c4a0b1569",
        source_checkpoint=str(RESNET_CHECKPOINT),
        training_config=resnet_training_config,
        test_metrics=resnet_test_metrics,
        lifecycle_stage=LIFECYCLE_CANDIDATE,
    )

    efficientnet_training_config_path = (
        PROJECT_ROOT
        / "reports/results/raptormaps/"
        "efficientnet_b0_finetune/training_config.json"
    )

    efficientnet_test_metrics_path = (
        PROJECT_ROOT
        / "reports/results/raptormaps/"
        "efficientnet_b0_finetune/test_overall_metrics.csv"
    )

    efficientnet_training_config = load_training_config(
        efficientnet_training_config_path
    )

    efficientnet_test_metrics = load_test_metrics(
        efficientnet_test_metrics_path
    )

    efficientnet_metadata = build_model_metadata(
        model_family="efficientnet_b0",
        source_run_id="7227fbc89615451a9aa5030cf8d8c824",
        source_checkpoint=str(EFFICIENTNET_CHECKPOINT),
        training_config=efficientnet_training_config,
        test_metrics=efficientnet_test_metrics,
        lifecycle_stage=LIFECYCLE_CANDIDATE,
    )


    results = []

    results.append(
        log_and_register(
            model_family="resnet18",
            registered_model_name=REGISTERED_MODEL_RESNET18,
            checkpoint_path=RESNET_CHECKPOINT,
            test_metrics_path=(
                PROJECT_ROOT
                / "reports/results/raptormaps/"
                "resnet18_finetune/test_overall_metrics.csv"
            ),
            training_config_path=(
                PROJECT_ROOT
                / "reports/results/raptormaps/"
                "resnet18_finetune/training_config.json"
            ),
            model_metadata=resnet_metadata,
            client=client,
        )
    )

    results.append(
        log_and_register(
            model_family="efficientnet_b0",
            registered_model_name=REGISTERED_MODEL_EFFICIENTNET_B0,
            checkpoint_path=EFFICIENTNET_CHECKPOINT,
            test_metrics_path=(
                PROJECT_ROOT
                / "reports/results/raptormaps/"
                "efficientnet_b0_finetune/test_overall_metrics.csv"
            ),
            training_config_path=(
                PROJECT_ROOT
                / "reports/results/raptormaps/"
                "efficientnet_b0_finetune/training_config.json"
            ),
            model_metadata=efficientnet_metadata,
            client=client,
        )
    )

    report_path = (
        OUTPUT_DIR
        / "raptormaps_registry_registration.json"
    )

    report_path.write_text(
        json.dumps(
            results,
            indent=2,
        )
    )

    print("\nRaptorMaps registry registration complete.")
    print(f"Report: {report_path}")

    for result in results:
        print(
            f"{result['registered_model_name']} "
            f"version={result['version']} "
            f"stage={result['lifecycle_stage']} "
            f"run={result['run_id']}"
        )

def load_training_config(path: Path) -> dict:
    return json.loads(path.read_text())


def load_test_metrics(path: Path) -> dict[str, float]:
    dataframe = pd.read_csv(path)

    if len(dataframe) != 1:
        raise RuntimeError(
            f"Expected exactly one metrics row in {path}, "
            f"found {len(dataframe)}."
        )

    row = dataframe.iloc[0]

    metric_mapping = {
        "accuracy": "accuracy",
        "balanced_accuracy": "balanced_accuracy",
        "macro_precision": "macro_precision",
        "macro_recall": "macro_recall",
        "macro_f1": "macro_f1",
        "weighted_f1": "weighted_f1",
        "roc_auc_ovr_macro": "roc_auc_ovr_macro",
        "pr_auc_macro": "pr_auc_macro",
    }

    metrics = {}

    for source_name, target_name in metric_mapping.items():
        if source_name not in dataframe.columns:
            raise RuntimeError(
                f"Missing metric column '{source_name}' in {path}"
            )

        metrics[target_name] = float(row[source_name])

    return metrics


if __name__ == "__main__":
    main()