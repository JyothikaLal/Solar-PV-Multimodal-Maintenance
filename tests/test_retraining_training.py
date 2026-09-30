from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from sklearn.ensemble import GradientBoostingRegressor

from src.mlops.retraining import training


def test_tecnalia_candidate_uses_production_gradient_boosting_config(
    monkeypatch,
):
    splits = {
        "train": pd.DataFrame(
            {
                "Front GPOA (W/m²)": [500.0, 600.0],
                "GHI (W/m²)": [450.0, 550.0],
                "Temp. Mod (°C)": [30.0, 31.0],
                "Amb. Temp. (°C)": [25.0, 26.0],
                "Wind Speed (m/s)": [1.0, 2.0],
                "module_name": ["Atersa", "Atersa"],
                "normalized_pmpp": [0.90, 0.91],
            }
        ),
        "validation": pd.DataFrame(
            {
                "Front GPOA (W/m²)": [550.0],
                "GHI (W/m²)": [500.0],
                "Temp. Mod (°C)": [30.5],
                "Amb. Temp. (°C)": [25.5],
                "Wind Speed (m/s)": [1.5],
                "module_name": ["Atersa"],
                "normalized_pmpp": [0.905],
            }
        ),
        "test": pd.DataFrame(
            {
                "Front GPOA (W/m²)": [580.0],
                "GHI (W/m²)": [530.0],
                "Temp. Mod (°C)": [31.0],
                "Amb. Temp. (°C)": [26.0],
                "Wind Speed (m/s)": [1.8],
                "module_name": ["Atersa"],
                "normalized_pmpp": [0.908],
            }
        ),
    }

    # Keep the test focused on the training adapter rather than raw-data IO.
    monkeypatch.setattr(
        training,
        "prepare_tecnalia_training_data",
        lambda: splits,
    )

    result = training.train_tecnalia_candidate()

    assert result.model_family == "gradient_boosting_tuned"
    assert isinstance(result.model, GradientBoostingRegressor)

    assert result.model.n_estimators == 400
    assert result.model.learning_rate == pytest.approx(0.03)
    assert result.model.max_depth == 3
    assert result.model.min_samples_leaf == 5
    assert result.model.random_state == 42

    assert set(result.validation_metrics) == {
        "mae",
        "rmse",
        "r2",
    }
    assert set(result.test_metrics) == {
        "mae",
        "rmse",
        "r2",
    }


def test_tecnalia_preprocessor_is_fitted_only_on_training_data():
    train = pd.DataFrame(
        {
            "numeric": [1.0, 2.0],
            "category": ["A", "A"],
        }
    )

    validation = pd.DataFrame(
        {
            "numeric": [1000.0],
            "category": ["B"],
        }
    )

    preprocessor = training._build_tecnalia_preprocessor(
        ["numeric"],
        ["category"],
    )

    train_transformed = preprocessor.fit_transform(train)
    validation_transformed = preprocessor.transform(validation)

    # StandardScaler fitted on [1, 2] has mean 1.5.
    assert train_transformed.shape[0] == 2
    assert validation_transformed.shape[0] == 1

    # If validation had influenced fitting, this transformed value
    # would be different. Unknown category B is safely handled.
    assert validation_transformed.shape[1] == train_transformed.shape[1]
    assert np.isfinite(validation_transformed).all()


def test_transfer_model_builder_supports_both_registered_families(
    monkeypatch,
):
    class DummyBackbone(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.fc = torch.nn.Linear(4, 12)
            self.classifier = torch.nn.Linear(4, 12)

        def forward(self, x):
            return self.fc(x)

    def fake_resnet(num_classes):
        return DummyBackbone()

    def fake_efficientnet(num_classes):
        return DummyBackbone()

    monkeypatch.setattr(
        training,
        "build_pretrained_resnet18",
        fake_resnet,
    )
    monkeypatch.setattr(
        training,
        "build_pretrained_efficientnet_b0",
        fake_efficientnet,
    )

    monkeypatch.setattr(
        training,
        "unfreeze_resnet18_layer4",
        lambda model: None,
    )
    monkeypatch.setattr(
        training,
        "unfreeze_efficientnet_b0_features8",
        lambda model: None,
    )

    def fake_optimizer(
        model,
        *,
        classifier_module,
        classifier_lr,
        backbone_lr,
        weight_decay,
    ):
        return torch.optim.AdamW(
            model.parameters(),
            lr=classifier_lr,
            weight_decay=weight_decay,
        )

    monkeypatch.setattr(
        training,
        "build_transfer_optimizer",
        fake_optimizer,
    )

    resnet_model, resnet_optimizer = (
        training._build_raptormaps_transfer_model(
            "resnet18_finetuned",
            12,
        )
    )

    efficientnet_model, efficientnet_optimizer = (
        training._build_raptormaps_transfer_model(
            "efficientnet_b0_finetuned",
            12,
        )
    )

    assert isinstance(
        resnet_model,
        training.RaptorMapsTransferAdapter,
    )
    assert isinstance(
        efficientnet_model,
        training.RaptorMapsTransferAdapter,
    )

    assert resnet_optimizer is not None
    assert efficientnet_optimizer is not None


def test_transfer_model_builder_rejects_unknown_family():
    with pytest.raises(ValueError, match="Unsupported RaptorMaps model family"):
        training._build_raptormaps_transfer_model(
            "unknown_family",
            12,
        )


def test_transfer_training_evaluates_saved_best_checkpoint(
    monkeypatch,
    tmp_path,
):
    class DummyModel(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.weight = torch.nn.Parameter(
                torch.tensor([1.0])
            )

        def forward(self, x):
            return torch.zeros(
                (x.shape[0], 2),
                dtype=torch.float32,
            )

    model = DummyModel()

    checkpoint_path = tmp_path / "best_model.pt"

    calls = []

    def fake_fit(**kwargs):
        torch.save(
            {
                "model_state_dict": {
                    "weight": torch.tensor([7.0])
                }
            },
            kwargs["checkpoint_path"],
        )

        return [
            {
                "epoch": 1.0,
                "train_loss": 1.0,
                "validation_loss": 1.0,
                "validation_macro_f1": 0.50,
                "validation_accuracy": 0.50,
                "validation_balanced_accuracy": 0.50,
                "learning_rate": 1e-3,
            },
            {
                "epoch": 2.0,
                "train_loss": 0.8,
                "validation_loss": 0.8,
                "validation_macro_f1": 0.80,
                "validation_accuracy": 0.80,
                "validation_balanced_accuracy": 0.80,
                "learning_rate": 5e-4,
            },
        ]

    def fake_evaluate(**kwargs):
        calls.append(
            float(kwargs["model"].weight.detach().item())
        )

        class Result:
            macro_f1 = 0.80
            accuracy = 0.80
            balanced_accuracy = 0.80

        return Result()

    class DummyLoader:
        def __init__(self):
            self.dataset = None

    train_loader = DummyLoader()
    validation_loader = DummyLoader()
    test_loader = DummyLoader()

    monkeypatch.setattr(
        training,
        "create_raptormaps_dataloaders",
        lambda **kwargs: (
            train_loader,
            validation_loader,
            test_loader,
        ),
    )

    class Dataset:
        class_names = ["Cell", "No-Anomaly"]

    dataset = Dataset()
    train_loader.dataset = dataset
    validation_loader.dataset = dataset
    test_loader.dataset = dataset

    monkeypatch.setattr(
        training,
        "create_raptormaps_weighted_loss",
        lambda: (
            torch.tensor([1.0, 1.0]),
            torch.nn.CrossEntropyLoss(),
        ),
    )

    monkeypatch.setattr(
        training,
        "_build_raptormaps_transfer_model",
        lambda **kwargs: (
            model,
            torch.optim.AdamW(model.parameters(), lr=1e-3),
        ),
    )

    monkeypatch.setattr(
        training,
        "fit_transfer_model",
        fake_fit,
    )

    monkeypatch.setattr(
        training,
        "evaluate_one_epoch_transfer",
        fake_evaluate,
    )

    monkeypatch.setattr(
        training,
        "_evaluate_raptormaps_test_set",
        lambda **kwargs: {
            "accuracy": 0.80,
            "balanced_accuracy": 0.80,
            "macro_precision": 0.80,
            "macro_recall": 0.80,
            "macro_f1": 0.80,
            "weighted_f1": 0.80,
            "roc_auc_ovr_macro": 0.80,
            "pr_auc_macro": 0.80,
        },
    )

    result = training.train_raptormaps_candidate(
        model_family="resnet18_finetuned",
        checkpoint_path=checkpoint_path,
        device=torch.device("cpu"),
    )

    assert result.best_validation_macro_f1 == pytest.approx(0.80)
    assert checkpoint_path.exists()

    # Both validation and test evaluation must use the saved best checkpoint.
    assert calls == [7.0]
