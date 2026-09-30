from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import random

import numpy as np
import pandas as pd
import torch
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline

from src.evaluation.classification import evaluate_classification

from src.data.tecnalia_regression import (
    MODULE_PATHS,
    RAW_ROOT,
    SPLIT_MANIFEST,
    attach_frozen_tecnalia_split,
    build_tecnalia_regression_frame,
    engineer_tecnalia_regression_modules,
    get_regression_feature_columns,
    load_tecnalia_regression_modules,
    split_tecnalia_regression_frame,
)
from src.data.raptormaps_torch import create_raptormaps_dataloaders
from src.models.vision.losses import create_raptormaps_weighted_loss
from src.models.vision.raptormaps_transfer import RaptorMapsTransferAdapter
from src.models.vision.transfer_models import (
    build_pretrained_efficientnet_b0,
    build_pretrained_resnet18,
    build_transfer_optimizer,
    unfreeze_efficientnet_b0_features8,
    unfreeze_resnet18_layer4,
)
from src.training.transfer_training import (
    TransferTrainingConfig,
    evaluate_one_epoch_transfer,
    fit_transfer_model,
)


@dataclass(frozen=True)
class RegressionTrainingResult:
    model_family: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    model: object
    preprocessor: object


@dataclass(frozen=True)
class ClassificationTrainingResult:
    model_family: str
    validation_metrics: dict[str, float]
    test_metrics: dict[str, float]
    best_validation_macro_f1: float
    checkpoint_path: Path
    history: list[dict[str, float]]
    class_names: list[str]


def set_reproducible_seed(seed: int) -> None:
    """Set the project-wide deterministic seed for retraining."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _regression_metrics(
    y_true: pd.Series | np.ndarray,
    y_pred: np.ndarray,
) -> dict[str, float]:
    return {
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(
            np.sqrt(mean_squared_error(y_true, y_pred))
        ),
        "r2": float(r2_score(y_true, y_pred)),
    }


def _build_tecnalia_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
) -> ColumnTransformer:
    """
    Build the same preprocessing contract used by the existing regression
    models: median/standardization for numeric features and most-frequent
    imputation + one-hot encoding for categorical features.
    """
    from sklearn.impute import SimpleImputer
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    numeric_pipeline = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        [
            (
                "imputer",
                SimpleImputer(strategy="most_frequent"),
            ),
            (
                "encoder",
                OneHotEncoder(
                    handle_unknown="ignore",
                    sparse_output=False,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        [
            ("numeric", numeric_pipeline, numeric_features),
            (
                "categorical",
                categorical_pipeline,
                categorical_features,
            ),
        ],
        remainder="drop",
    )


def prepare_tecnalia_training_data() -> dict[str, pd.DataFrame]:
    """
    Load, preprocess, engineer and attach the frozen TECNALIA split.
    """
    modules = load_tecnalia_regression_modules(
        raw_root=RAW_ROOT,
        module_paths=MODULE_PATHS,
    )

    modules = engineer_tecnalia_regression_modules(modules)

    frame = build_tecnalia_regression_frame(modules)

    split_manifest = pd.read_csv(SPLIT_MANIFEST)

    frame = attach_frozen_tecnalia_split(
        frame,
        split_manifest,
    )

    return split_tecnalia_regression_frame(frame)


def train_tecnalia_candidate() -> RegressionTrainingResult:
    """
    Retrain the production TECNALIA Gradient Boosting configuration.

    The estimator is fitted only on the frozen training split.
    Validation and test are evaluated independently.
    """
    splits = prepare_tecnalia_training_data()

    numeric_features, categorical_features = (
        get_regression_feature_columns()
    )

    train = splits["train"]
    validation = splits["validation"]
    test = splits["test"]

    preprocessor = _build_tecnalia_preprocessor(
        numeric_features,
        categorical_features,
    )

    x_train = preprocessor.fit_transform(
        train[numeric_features + categorical_features]
    )
    x_validation = preprocessor.transform(
        validation[numeric_features + categorical_features]
    )
    x_test = preprocessor.transform(
        test[numeric_features + categorical_features]
    )

    model = GradientBoostingRegressor(
        n_estimators=400,
        learning_rate=0.03,
        max_depth=3,
        min_samples_leaf=5,
        random_state=42,
    )

    model.fit(
        x_train,
        train["normalized_pmpp"],
    )

    validation_predictions = model.predict(x_validation)
    test_predictions = model.predict(x_test)

    return RegressionTrainingResult(
        model_family="gradient_boosting_tuned",
        validation_metrics=_regression_metrics(
            validation["normalized_pmpp"],
            validation_predictions,
        ),
        test_metrics=_regression_metrics(
            test["normalized_pmpp"],
            test_predictions,
        ),
        model=model,
        preprocessor=preprocessor,
    )


def _build_raptormaps_transfer_model(
    model_family: str,
    num_classes: int,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if model_family == "resnet18_finetuned":
        backbone = build_pretrained_resnet18(
            num_classes=num_classes,
        )
        unfreeze_resnet18_layer4(backbone)

        model = RaptorMapsTransferAdapter(
            backbone=backbone,
            classifier_module=backbone.fc,
            spatial_size=(160, 96),
        )

        optimizer = build_transfer_optimizer(
            backbone,
            classifier_module=backbone.fc,
            classifier_lr=1e-3,
            backbone_lr=1e-4,
            weight_decay=1e-4,
        )

        return model, optimizer

    if model_family == "efficientnet_b0_finetuned":
        backbone = build_pretrained_efficientnet_b0(
            num_classes=num_classes,
        )
        unfreeze_efficientnet_b0_features8(backbone)

        model = RaptorMapsTransferAdapter(
            backbone=backbone,
            classifier_module=backbone.classifier,
            spatial_size=(160, 96),
        )

        optimizer = build_transfer_optimizer(
            backbone,
            classifier_module=backbone.classifier,
            classifier_lr=1e-3,
            backbone_lr=1e-4,
            weight_decay=1e-4,
        )

        return model, optimizer

    raise ValueError(
        f"Unsupported RaptorMaps model family: {model_family}"
    )


@torch.no_grad()
def _evaluate_raptormaps_test_set(
    model: torch.nn.Module,
    loader,
    device: torch.device,
    class_names: list[str],
) -> dict[str, float]:
    model.eval()

    all_targets = []
    all_predictions = []
    all_probabilities = []

    for images, targets in loader:
        images = images.to(device, non_blocking=True)

        logits = model(images)

        if not torch.isfinite(logits).all():
            raise ValueError("Non-finite logits encountered during test evaluation.")

        probabilities = torch.softmax(logits, dim=1)
        predictions = torch.argmax(logits, dim=1)

        all_targets.append(targets.cpu().numpy())
        all_predictions.append(predictions.cpu().numpy())
        all_probabilities.append(probabilities.cpu().numpy())

    if not all_targets:
        raise ValueError("RaptorMaps test loader produced no samples.")

    y_true = np.concatenate(all_targets)
    y_pred = np.concatenate(all_predictions)
    y_proba = np.concatenate(all_probabilities)

    evaluation = evaluate_classification(
        y_true=y_true,
        y_pred=y_pred,
        class_names=class_names,
        y_proba=y_proba,
    )

    overall = evaluation["overall"]

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

    missing = required_metrics - set(overall)
    if missing:
        raise ValueError(
            f"Canonical RaptorMaps evaluation is missing metrics: {sorted(missing)}"
        )

    return {
        metric: float(overall[metric])
        for metric in required_metrics
    }


def train_raptormaps_candidate(
    model_family: str,
    checkpoint_path: Path,
    *,
    seed: int = 42,
    batch_size: int = 64,
    num_workers: int = 0,
    device: torch.device | None = None,
) -> ClassificationTrainingResult:
    """
    Retrain one registered RaptorMaps transfer-learning family.

    The frozen train/validation/test manifest remains authoritative.
    """
    set_reproducible_seed(seed)

    train_loader, validation_loader, test_loader = (
        create_raptormaps_dataloaders(
            batch_size=batch_size,
            num_workers=num_workers,
        )
    )

    class_names = train_loader.dataset.class_names

    _, loss_function = create_raptormaps_weighted_loss()

    model, optimizer = _build_raptormaps_transfer_model(
        model_family=model_family,
        num_classes=len(class_names),
    )

    if device is None:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )

    config = TransferTrainingConfig(
        max_epochs=30,
        early_stopping_patience=5,
        scheduler_factor=0.5,
        scheduler_patience=2,
        min_learning_rate=1e-6,
    )

    history = fit_transfer_model(
        model=model,
        train_loader=train_loader,
        validation_loader=validation_loader,
        loss_function=loss_function,
        class_names=class_names,
        optimizer=optimizer,
        device=device,
        config=config,
        checkpoint_path=checkpoint_path,
    )

    checkpoint = torch.load(
        checkpoint_path,
        map_location=device,
        weights_only=False,
    )

    if "model_state_dict" not in checkpoint:
        raise ValueError(
            "Transfer checkpoint is missing model_state_dict."
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True,
    )

    validation = evaluate_one_epoch_transfer(
        model=model,
        loader=validation_loader,
        loss_function=loss_function,
        device=device,
        class_names=class_names,
    )

    test_metrics = _evaluate_raptormaps_test_set(
        model=model,
        loader=test_loader,
        device=device,
        class_names=class_names,
    )

    best_validation_macro_f1 = max(
        record["validation_macro_f1"]
        for record in history
    )

    return ClassificationTrainingResult(
        model_family=model_family,
        validation_metrics={
            "macro_f1": float(validation.macro_f1),
            "accuracy": float(validation.accuracy),
            "balanced_accuracy": float(
                validation.balanced_accuracy
            ),
        },
        test_metrics=test_metrics,
        best_validation_macro_f1=float(
            best_validation_macro_f1
        ),
        checkpoint_path=checkpoint_path,
        history=history,
        class_names=class_names,
    )
