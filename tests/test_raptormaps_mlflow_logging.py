import json
from pathlib import Path

import pandas as pd
import pytest


def test_raptormaps_finetuned_artifacts_exist():
    root = Path("reports/results/raptormaps")

    required = [
        root / "resnet18_finetune/training_config.json",
        root / "resnet18_finetune/test_overall_metrics.csv",
        root / "resnet18_finetune/training_history.csv",
        root / "efficientnet_b0_finetune/training_config.json",
        root / "efficientnet_b0_finetune/test_overall_metrics.csv",
        root / "efficientnet_b0_finetune/training_history.csv",
    ]

    missing = [str(path) for path in required if not path.exists()]

    assert not missing, f"Missing RaptorMaps MLflow source artifacts: {missing}"


@pytest.mark.parametrize(
    "model_dir,expected_model,expected_epoch",
    [
        ("resnet18_finetune", "resnet18", 14),
        ("efficientnet_b0_finetune", "efficientnet_b0", 17),
    ],
)
def test_raptormaps_finetuned_config_contract(
    model_dir,
    expected_model,
    expected_epoch,
):
    path = Path("reports/results/raptormaps") / model_dir / "training_config.json"

    with path.open() as handle:
        config = json.load(handle)

    assert config["model"] == expected_model
    assert config["pretrained"] is True
    assert config["fine_tuning"] is True
    assert config["seed"] == 42
    assert config["batch_size"] == 64
    assert config["best_epoch"] == expected_epoch
    assert len(config["class_names"]) == 12


@pytest.mark.parametrize(
    "model_dir",
    [
        "resnet18_finetune",
        "efficientnet_b0_finetune",
    ],
)
def test_raptormaps_test_metrics_are_complete(model_dir):
    path = Path("reports/results/raptormaps") / model_dir / "test_overall_metrics.csv"

    metrics = pd.read_csv(path)

    expected_columns = {
        "accuracy",
        "balanced_accuracy",
        "macro_precision",
        "macro_recall",
        "macro_f1",
        "weighted_f1",
        "roc_auc_ovr_macro",
        "pr_auc_macro",
    }

    assert len(metrics) == 1
    assert expected_columns.issubset(metrics.columns)
    assert metrics[list(expected_columns)].notna().all().all()


def test_raptormaps_mlflow_logging_scripts_exist():
    required = [
        Path("scripts/log_raptormaps_finetuned_mlflow.py"),
        Path("src/mlops/mlflow_tracking.py"),
    ]

    missing = [str(path) for path in required if not path.exists()]

    assert not missing, f"Missing MLflow tracking implementation files: {missing}"
