"""Register finalized RaptorMaps fine-tuned models in MLflow.

This script is retrospective: it reads existing checkpoints and evaluation
artifacts. It does not retrain models or regenerate predictions.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.mlops.mlflow_tracking import (
    EXPERIMENT_RAPTORMAPS_TRANSFER,
    build_standard_tags,
    log_artifact,
    log_dataset_metadata,
    log_metrics,
    log_params,
    start_run,
)


RAPTOR_ROOT = PROJECT_ROOT / "reports" / "results" / "raptormaps"
CHECKPOINT_ROOT = PROJECT_ROOT / "models" / "raptormaps"

MODELS = {
    "resnet18_finetune": {
        "model_family": "resnet18",
        "run_name": "raptormaps_resnet18_finetuned_retrospective",
    },
    "efficientnet_b0_finetune": {
        "model_family": "efficientnet_b0",
        "run_name": "raptormaps_efficientnet_b0_finetuned_retrospective",
    },
}


def read_metrics(path: Path) -> dict[str, float]:
    frame = pd.read_csv(path)

    if len(frame) != 1:
        raise ValueError(f"Expected exactly one metrics row: {path}")

    row = frame.iloc[0].to_dict()

    return {
        "test_accuracy": float(row["accuracy"]),
        "test_balanced_accuracy": float(row["balanced_accuracy"]),
        "test_macro_precision": float(row["macro_precision"]),
        "test_macro_recall": float(row["macro_recall"]),
        "test_macro_f1": float(row["macro_f1"]),
        "test_weighted_f1": float(row["weighted_f1"]),
        "test_roc_auc_ovr_macro": float(row["roc_auc_ovr_macro"]),
        "test_pr_auc_macro": float(row["pr_auc_macro"]),
    }


def log_model(model_dir_name: str, model_family: str, run_name: str) -> str:
    result_dir = RAPTOR_ROOT / model_dir_name
    checkpoint = CHECKPOINT_ROOT / model_dir_name / "best_model.pt"

    config_path = result_dir / "training_config.json"
    metrics_path = result_dir / "test_overall_metrics.csv"
    history_path = result_dir / "training_history.csv"
    confusion_path = result_dir / "test_confusion_matrix.csv"
    per_class_path = result_dir / "test_per_class_metrics.csv"
    class_weights_path = result_dir / "class_weights.csv"

    required = [
        checkpoint,
        config_path,
        metrics_path,
        history_path,
        confusion_path,
        per_class_path,
        class_weights_path,
    ]

    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Required artifact missing: {path}")

    with config_path.open() as handle:
        config = json.load(handle)

    metrics = read_metrics(metrics_path)

    tags = build_standard_tags(
        dataset="RaptorMaps",
        modality="thermal",
        task="classification",
        model_family=model_family,
        stage="finetuned",
        run_type="retrospective",
        split_strategy="frozen_raptormaps_manifest",
    )

    tags.update(
        {
            "dataset_name": "RaptorMaps InfraredSolarModules",
            "dataset_version": "project_locked",
            "fine_tuning": str(config["fine_tuning"]).lower(),
            "pretrained": str(config["pretrained"]).lower(),
            "best_epoch": str(config["best_epoch"]),
        }
    )

    params = {
        "model": config["model"],
        "pretrained": config["pretrained"],
        "fine_tuning": config["fine_tuning"],
        "trainable_backbone": config["trainable_backbone"],
        "seed": config["seed"],
        "batch_size": config["batch_size"],
        "num_workers": config["num_workers"],
        "device": config["device"],
        "spatial_size": config["spatial_size"],
        "classifier_learning_rate": config["classifier_learning_rate"],
        "backbone_learning_rate": config["backbone_learning_rate"],
        "weight_decay": config["weight_decay"],
        "max_epochs": config["max_epochs"],
        "early_stopping_patience": config["early_stopping_patience"],
        "scheduler_factor": config["scheduler_factor"],
        "scheduler_patience": config["scheduler_patience"],
        "min_learning_rate": config["min_learning_rate"],
        "best_epoch": config["best_epoch"],
        "best_validation_macro_f1": config["best_validation_macro_f1"],
        "class_count": len(config["class_names"]),
        "class_names": config["class_names"],
        "checkpoint": str(checkpoint.relative_to(PROJECT_ROOT)),
    }

    with start_run(
        experiment_name=EXPERIMENT_RAPTORMAPS_TRANSFER,
        run_name=run_name,
        tags=tags,
    ):
        log_params(params)

        log_metrics(
            {
                "best_validation_macro_f1": float(
                    config["best_validation_macro_f1"]
                ),
                **metrics,
            }
        )

        log_dataset_metadata(
            dataset_name="RaptorMaps InfraredSolarModules",
            dataset_version="project_locked",
            split_strategy="frozen_raptormaps_manifest",
            metadata={
                "class_count": len(config["class_names"]),
                "class_names": config["class_names"],
                "evaluation_scope": "existing_test_evaluation",
                "run_provenance": "retrospective_existing_results",
            },
        )

        artifact_paths = [
            config_path,
            metrics_path,
            history_path,
            confusion_path,
            per_class_path,
            class_weights_path,
            checkpoint,
        ]

        for artifact in artifact_paths:
            log_artifact(
                artifact,
                artifact_path="raptormaps_finetuned",
            )

    return run_name


def main() -> None:
    created_runs = []

    for model_dir_name, specification in MODELS.items():
        run_name = log_model(
            model_dir_name=model_dir_name,
            model_family=specification["model_family"],
            run_name=specification["run_name"],
        )
        created_runs.append(run_name)

    print("RaptorMaps MLflow retrospective runs created successfully.")
    for run_name in created_runs:
        print("Run name:", run_name)


if __name__ == "__main__":
    main()
